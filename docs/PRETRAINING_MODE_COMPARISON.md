# HDC pretraining: bit-tam hız ve controlled semantic-cohesion karşılaştırması

Bu belge `tools/compare_pretraining_modes.py` ile üretilen sınırlı ölçüm
protokolünü tanımlar. Araç gerçek `vegai_pretrain train` executable'ını çalıştırır
ve ölçümü yalnız o child process'in başlatılması ile bitişi arasında alır.
Compiler/configure, CI setup, fixture üretimi, checkpoint inspect, JSON yazımı
ve Python harness süresi **ölçüye dahil değildir**. Vocabulary pass, streaming
öğrenme ve checkpoint write ise `train` komutunun parçası oldukları için dahildir.

> Bu sonuçlar repository-authored, dengeli, sentetik ve corpus-içi controlled
> diagnostic'tir. Doğal dil zekâsı, held-out task accuracy, genel Türkçe anlama
> veya production throughput garantisi değildir. Gerçek corpus için ayrı
> held-out/de-duplicated evaluation manifesti gerekir.

## V3 protokolü: portable CPU provenance + gerçek train timing

```bash
make clean-all
make all                         # portable default: ./build/vegai_pretrain
make native                      # host-only: ./build-native/vegai_pretrain
python3 tools/hardware_compatibility.py --format json --require-recognized-os \
  --output hardware-profile.json
python3 tools/compare_pretraining_build_profiles.py \
  --portable-pretrain-bin ./build/vegai_pretrain \
  --native-pretrain-bin ./build-native/vegai_pretrain \
  --repetitions 3 --omp-threads 1
python3 tools/compare_pretraining_modes.py \
  --pretrain-bin ./build/vegai_pretrain --repetitions 3 --omp-threads 1
```

Her varyant üç bağımsız train process'i ile ölçülür; tablo değeri wall-clock
medyandır. Sabit `OMP_NUM_THREADS=1`, `max-vocab=128`, `min-frequency=8`,
window `2`, positional shift `41`, plasticity `25 → 1`, decay `8` kullanılır.
JSON provenance CPU modeli, OS, mimari, logical CPU sayısı ve GPU adapter
inventory'sini taşır; GPU compute kullanılmaz. Hardware discovery, compilation,
fixture setup, inspect ve harness süreleri timing kapsamı dışındadır.

### A. Portable/native build-profile diagnostic

Portable binary host-ISA flag'i taşımaz; native binary aynı host için opt-in
`-march=native` tuning kullanır. İki binary, aynı deterministic `hybrid` train
komutunda **checkpoint byte-tam eşitliği** üretmek zorundadır. Hız oranı
`portable_median / native_median` olarak raporlanır ve yalnız ölçülen CPU/OS
hostu için yorumlanır. Bu test iGPU/GPU performans ölçümü değildir.

### B. Base encoder diagnostic

`mini_turkish_ascii_pretrain.txt` sekiz disjoint topical kümeden oluşur:

```text
Corpus                    mini_turkish_ascii_pretrain.txt
Record length             8 active token
Training token/run        2,048
Epoch/run                 2
Ordered context pair/run  6,656
Vocabulary                64 active token
```

Burada `legacy`, bit-tam hızlı `positional` ve explicit local fused `hybrid`
çalıştırılır. Genel `train` varsayılanı artık `hierarchical`tir; bu 8-token
fixture hierarchy eşiğinin altında olduğundan local hybrid ile bit-tam aynıdır. Her checkpoint gerçek CLI üzerinden 64 self-excluded Top-1 query
ile okunur. `positional` nearest-neighbour çıktısının `legacy` ile tam eşit
olması gerekir; `hybrid` aynı-küme hit'i geriletmemeli ve mean Top-1 similarity
artışı `> 0.01` olmalıdır.

### C. Fused/reference semantic-preservation hız karşılaştırması

Aynı base corpus/config ile iki `hybrid` varyant çalışır:

- **reference:** `--reference-context-kernel` ile eski allocate/permute/iki-add
  yolu;
- **fused:** varsayılan raw SoA-word + fused positional/topic pair yolu.

İki checkpoint'in tüm byte dizisi ve SHA-256'sı **eşit** olmak zorundadır. Bu
kural Top-1 diagnostic eşitliğinden daha güçlüdür: vocabulary, base plane,
semantic plane, experience sayaçları, totals ve v5 checksum'ın tamamının aynı
olduğunu doğrular. Hız oranı `reference_median / fused_median` olarak verilir.

### D. Long-record hierarchical controlled diagnostic

Normal fixture kasıtlı olarak sekiz-token satırlıdır ve `hierarchical` eşiği
olan 12 active token'ın altında kalır. Araç geçici, ASCII-only bir fixture üretir:
her biri aynı disjoint topical kümenin sekiz token'ını iki kez içeren **16-token**
kayıtlar; küme başına 16 kayıt, toplam 128 kayıt. İki epoch için:

```text
Training token/run        4,096
Ordered context pair/run  14,848
```

`hierarchical`, local `hybrid` representation'ı korur ve yalnız bu uzun
kayıtlarda satır-base-vector majority özetini üç bounded katkı olarak ekler.
Bu nedenle comparison `hybrid` ile `hierarchical` arasında yapılır. Acceptance
signal: same-cluster Top-1 gerilemez ve mean Top-1 similarity gain `> 0.002`
olur. Bu bilerek tasarlanmış in-corpus cohesion testidir; held-out task
başarısı değildir.

## V4: deterministic single-core / multi-core context construction

[GitHub Actions run 39839135866](https://github.com/sswweett00/Vegai/actions/runs/39839135866)
(commit `085d2dc`) yeni immutable-context / source-ordered-commit yolunu gerçek
GNU Fortran/OpenMP binary ile doğruladı. Host AMD EPYC 7763 / Linux
`7.0.0-1017-azure` idi; test özellikle long-record fixture için yalnız
`OMP_NUM_THREADS=1` ve `2` arasını karşılaştırdı. Fixture 16 record,
record başına 256 token, 2 epoch, toplam 8.192 training token ve 64.896 ordered
context pair içerir.

| OpenMP thread | Median gerçek train süresi | Yaklaşık training token/s | Checkpoint |
|---:|---:|---:|---|
| 1 | 0.538901 s | ≈15,201.308 | regression arm |
| 2 | 0.465863 s | ≈17,584.569 | **byte-tam eşit** |

Aynı host/workload'da 2-thread yol **1.1568×** (yaklaşık `%15.68`) hızlandı;
checkpoint bytes ve SHA-256 tam eşitti. Context'ler paralel hazırlanırken
prototype/experience commit sırası korunduğu için bu, yalnız qualitative
neighbour çıktısı değil persisted model state için de deterministic bir
eşitliktir. Bu sonuç 64-core EPYC'nin tamamını kullandığı veya her corpus'un
lineer ölçekleneceği anlamına gelmez; yalnız istenen 2-thread armın gerçek
sonucudur. GPU/NPU compute kullanılmadı.

Aynı CI run'da controlled semantic signals korundu: base hybrid mean Top-1
similarity `0.137414`; long-record hybrid `0.168998`; hierarchical `0.826843`
ve gain `+0.657845`. Genel fresh `train` varsayılanı artık `hierarchical`tir;
bu quality default yalnız ≥12 active-token record'larda local hybrid'den ayrılır.
Bu yine sentetik/in-corpus diagnostic'tir, held-out doğal dil başarısı değildir.

## V5: OpenVINO immutable-context bridge — resident worker / CPU policy sonucu

[GitHub Actions run 39851363774](https://github.com/sswweett00/Vegai/actions/runs/39851363774)
(commit `750af27`) optional OpenVINO runtime'ı ile gerçek Fortran↔Python
lane-worker bridge'ini tekrar çalıştırdı. Worker artık bir train süresince
resident, `Core`/dynamic-batch compiled graph cache'i request'ler arasında
paylaşılıyor. Actual edge cardinality'leri seeded neutral `(tie, NOT tie)`
pair'leri ile iki reusable odd/even graph shape'ına normalize edilir;
source-order Hebbian commit Fortran CPU'da kalır. Strict CPU reference ve
provider checkpoint byte'ları tam eşitti.

| Arm | Gerçek `vegai_pretrain train` süresi | Dispatched context | Checkpoint |
|---|---:|---:|---|
| portable CPU reference | `0.021801 s` | 0 | regression arm |
| önceki process-per-line strict CPU bridge | `1.438524 s` | 30 | byte-tam eşit |
| resident/cached strict OpenVINO **CPU plugin** bridge | `0.400872 s` | 30 | **byte-tam eşit** |

Strict bridge ratio `67.9080×`den **`18.3878×`**e indi; same small fixture'da
bridge süresi yaklaşık **3.59×** azaldı. Bu, cold Python/OpenVINO startup
maliyetinin tamamını değil per-line startup ve repeated-shape compile maliyetini
ortadan kaldırır. Dolayısıyla strict CPU plugin armı hâlâ CPU hızlandırması
iddiası değildir.

Production CPU policy ayrıdır: fallback açıkken explicit `--openvino-device
CPU`, worker açmak yerine native fused/OpenMP context construction'a gider;
runtime output `native-CPU bypass` sayacını yükseltir, OpenVINO device execution
mask'i ve worker session/request sayaçları sıfır kalır. CI bu arm için de
checkpoint equality ve bounded child-process timing guard uygular. Ölçü yalnız
`vegai_pretrain train` child process'ini kapsar; fixture üretimi, byte compare
ve Python harness hariçtir.

CI hostunda iGPU/NPU yoktu. Bu nedenle gerçek Intel Core Ultra GPU/NPU speedup,
energy veya multi-device scaling değeri **henüz ölçülmüş değildir**. Target
benchmark yalnız `--openvino-require-accelerator` ile worker'ın GPU/NPU mask'i
doğrulandığında raporlanmalıdır.

## Güncel V3 sonucu — portable profile ile doğrulanmış GNU Fortran CI

[GitHub Actions run 39836218246](https://github.com/sswweett00/Vegai/actions/runs/39836218246)
(commit `db538b1`) Linux `7.0.0-1017-azure`, `x86_64`, Intel(R) Xeon(R)
`6973P-C` hostunda, `OMP_NUM_THREADS=1` ve üç repetition ile tamamlandı.
Checked CMake/OpenMP testleri, hardware reporter, portable/native byte-equality
benchmark, executable CLI doğrulamaları ve controlled semantic benchmark'ın
tümü geçti. GPU compute kullanılmadı.

### Portable varsayılan ile host-native profil: gerçek önce / sonra

Bu iki sonuç aynı hostta, aynı gerçek `hybrid` train corpus/config ile alındı;
derleme ve harness süresi hariçtir.

| Build profile | Median train süresi | Yaklaşık training token/s | Sonuç |
|---|---:|---:|---|
| portable, host-ISA yok | 0.126613 s | ≈16,175.274 | dağıtılabilir varsayılan |
| native, build-host tuned | 0.113506 s | ≈18,043.099 | **1.1155×** daha hızlı |

Portable/native checkpoint byte dizisi ve SHA-256'sı **tam eşit** bulundu
(`portable_native_bytes_identical=True`). Böylece portable default, semantic
ve persistent model state değiştirmeden bu hostta native profile göre yaklaşık
%10.35 daha düşük throughput ile çalıştı. Bu fark yalnız bu Xeon/Linux runner
içindir; Ultra 5 125H veya başka bir CPU'ya doğrudan taşınamaz.

### Portable binary üzerinde fused context / semantic diagnostic

| Encoder / kernel | Median train süresi | Yaklaşık training token/s | Karşılaştırma |
|---|---:|---:|---|
| `legacy` | 0.383280 s | ≈5,343.352 | 1.0000× |
| `positional` | 0.112116 s | ≈18,266.795 | 3.4186× legacy'ye göre |
| fused `hybrid` | 0.112756 s | ≈18,163.113 | 3.3992× legacy'ye göre |
| reference `hybrid` | 0.115806 s | ≈17,684.749 | fused'e göre 1.0271× daha yavaş |

Fused/reference `hybrid` checkpoint'leri yine byte-tam eşitti. Base fixture
fused hybrid mean Top-1 similarity değeri `0.137414` oldu. Long-record
controlled diagnostic'te local hybrid `0.168998`, hierarchical `0.826843`
mean Top-1 similarity üretti: **+0.657845** (+389.262%). Aynı-küme Top-1
hit gerilememe koşulu geçti. Bu son semantic satır yalnız tasarlanmış 16-token
in-corpus fixture için geçerlidir; doğal dil veya held-out task iddiası değildir.

## Önceki baseline (fused-pair değişikliğinden önce)

[GitHub Actions run 39826604165](https://github.com/sswweett00/Vegai/actions/runs/39826604165)
üzerindeki son önceki optimized GNU Make run'ı, o dönemin `legacy`, positional
ve pre-fused hybrid implementasyonunu şu şekilde raporladı:

| Encoder | Median train süresi | Training token/s | Legacy'ye göre | Mean Top-1 similarity |
|---|---:|---:|---:|---:|
| `legacy` | 0.466586 s | 4,389.331 | 1.0000× | — |
| `positional` | 0.160154 s | 12,787.702 | 2.9134× | 0.073368 |
| `hybrid` | 0.165548 s | 12,371.067 | 2.8184× | 0.137414 |

Base fixture'ın Top-1 cluster hit'i üç mode için `%100` idi; hybrid'in
positional'a göre similarity gain'i `+0.064046` idi. Bunlar tarihsel baseline
olarak saklanır; CPU/runner/derleyici farklı olduğunda mutlak saniye veya
token/s değerlerini doğrudan karşılaştırmayın.

## Önceki v2 sonucu — doğrulanmış GNU Fortran CI

[GitHub Actions run 39830774456](https://github.com/sswweett00/Vegai/actions/runs/39830774456)
(commit `515aec8`, optimized GNU Make binary, üç repetition, `OMP_NUM_THREADS=1`)
tüm checked native property/integration testleri, executable/continual CLI
verifier'ları ve yeni benchmark adımıyla başarılı tamamlandı.

### Base fixture: önce / sonra encoder süreleri

| Encoder | Median train süresi | Training token/s | Legacy'ye göre |
|---|---:|---:|---:|
| `legacy` | 0.467324 s | 4,382.399 | 1.0000× |
| `positional` | 0.160000 s | 12,800.000 | 2.9208× |
| fused `hybrid` | 0.162482 s | 12,604.473 | 2.8762× |

Current fused hybrid'in base controlled mean Top-1 similarity değeri `0.137414` oldu.
Bu, fused kernelin hybrid semantic contract'i değiştirmediğine dair ayrı
checkpoint-byte testinin yanında yalnız yardımcı controlled retrieval sinyalidir.

### Aynı hybrid semantiğinde gerçek before/after kernel ölçümü

| Uygulama | Median train süresi | Training token/s | Sonuç |
|---|---:|---:|---|
| reference rotate-then-add | 0.165942 s | 12,341.662 | regression oracle |
| fused raw SoA/pair | 0.162482 s | 12,604.473 | **1.0213×** daha hızlı |

İki run'ın checkpoint byte dizisi ve SHA-256'sı tam eşit bulundu
(`checkpoint_bytes_identical=True`). Bu, yalnız Top-1 output'u değil tüm
persisted model state'inin bit-tam korunduğu anlamına gelir.

### Long-record hierarchical controlled cohesion diagnostic

| Mode | Mean Top-1 similarity | Hybrid'e göre |
|---|---:|---:|
| local `hybrid` | 0.168998 | — |
| `hierarchical` | 0.826843 | **+0.657845** (+389.262%) |

Aynı-küme Top-1 hit gerilememe koşulu da geçti. Bu belirgin marj,
özellikle her record'un aynı disjoint topical kümeyi iki kez taşıdığı tasarlanmış
16-token fixture için geçerlidir; doğal metin, held-out task veya production
semantic-quality sonucu olarak genellenemez.

Her zaman aynı runner içindeki oranları yorumlayın. Cache, CPU frequency/termal
state, filesystem, compiler sürümü/flag'leri ve OpenMP ayarı mutlak ölçümü
değiştirir. Yukarıdaki V3 portable/native değerleri ile bu eski v2
native-default sonucu farklı CI hostlarında alındığı için doğrudan cross-run
throughput iddiası olarak karşılaştırılmamalıdır.

## Uygulama notları

- `hv_bitacc_add_words`, `hv_bitacc_add_permuted_words` ve
  `hv_bitacc_add_permuted_pair_words`, geçici `hv_t` kopyasını kaldırarak aynı
  bit-sliced population-counter arithmetic'ini doğrudan vocabulary'nin SoA
  column'unda uygular.
- Native property testleri negatif/pozitif shift, bütün counter plane'leri,
  sample count ve pair-overflow atomicity'sini kontrol eder.
- `--reference-context-kernel` checkpoint formatına yazılmaz; yalnız benchmark
  ve regression oracle'dır. Aynı semantic config ile fused/reference checkpoint
  byte eşitliği bu nedenle zorunludur.
- Checkpoint writer v5'tir; reader v4/v5 kabul eder ve checksum'ı dosyanın
  kendi version değeriyle doğrular.
- Hierarchical global signal satırda bir kez üretilir, yalnız ≥12 active token
  için etkinleşir ve local context accumulator kapasitesiyle bounded kalır.
