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

## V5 protokolü: portable CPU provenance + gerçek train timing

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

Burada `legacy`, bit-tam hızlı `positional`, explicit local fused `hybrid` ve
iki-epoch **`resonant`** çalıştırılır. Genel `train` varsayılanı artık
`hierarchical`tir; bu 8-token fixture hierarchy eşiğinin altında olduğundan
local hybrid ile bit-tam aynıdır. Her checkpoint gerçek CLI üzerinden 64
self-excluded Top-1 query ile okunur. `positional` nearest-neighbour çıktısının
`legacy` ile tam eşit olması gerekir; `hybrid` aynı-küme hit'i geriletmemeli ve
mean Top-1 similarity artışı `> 0.01` olmalıdır.

`resonant`in birinci epoch'u `hybrid` ile bit-tam aynıdır; ikinci epoch'ta yalnız
başlangıçta frozen learned semantic plane topical input olur. Comparator,
`resonant_mean_top1_similarity_gain_vs_hybrid` ile same-cluster hit delta'sını
ve `resonant_speedup_vs_hybrid` değerini raporlar. Bu repository fixture'ı için
mean Top-1 similarity gain **`> 0.02`** fail-closed acceptance koşuludur. Bu
delta controlled fixture için gerçek semantic-retrieval evidence'ı olsa da genel
corpus/LLM yeteneği veya her veri dağılımında pozitif gain garantisi olmaz.
Native testte fused/reference equality, checkpoint-resume equality ve v8 lineage
ayrı zorunlu invariant'lardır.

#### İlk doğrulanmış resonant controlled sonuç

[GitHub Actions run 40127066062](https://github.com/sswweett00/Vegai/actions/runs/40127066062)
(`995cb7e`, Intel Xeon Platinum 8573C hosted Linux, `OMP_NUM_THREADS=1`) gerçek
compiled Fortran binary, native property tests ve üçer child-process repetition
ile bu gate'i geçti. Aynı 64-query base fixture'da `hybrid` mean self-excluded
Top-1 similarity **`0.138813`**, `resonant` **`0.285841`** oldu; fark
**`+0.147028`** olup `+0.02` acceptance sınırının üzerindedir. Bu önce/sonra
karşılaştırmasında yalnız context mode farklıdır; compilation, fixture
construction, inspect ve Python harness timing'e dahil değildir. `resonant`
train-time/throughput değeri aynı workflow summary'sinde raporlanır ve sonraki
run'larda quality gate ile birlikte izlenir. Sonuç repository-authored
in-corpus cluster diagnostic'idir; held-out domain accuracy, doğal dil anlama
veya genel zekâ iddiası değildir.

### C. Fused/reference semantic-preservation hız karşılaştırması

Aynı base corpus/config ile iki `hybrid` varyant çalışır:

- **reference:** `--reference-context-kernel` ile eski allocate/permute/iki-add
  yolu;
- **fused:** varsayılan raw SoA-word + fused positional/topic pair yolu.

İki checkpoint'in tüm byte dizisi ve SHA-256'sı **eşit** olmak zorundadır. Bu
kural Top-1 diagnostic eşitliğinden daha güçlüdür: vocabulary, base plane,
semantic plane, experience sayaçları, totals ve mode'un seçtiği checkpoint
version checksum'ının tamamının aynı olduğunu doğrular. Yeni `hybrid` armı v7
Floyd/complement sampler lineage'i kullanır. Hız oranı
`reference_median / fused_median` olarak verilir.

### D. Long-record hierarchical / coherent controlled diagnostic

Normal fixture kasıtlı olarak sekiz-token satırlıdır ve `hierarchical` ile
`coherent` eşiği olan 12 active token'ın altında kalır. Araç geçici, ASCII-only
bir fixture üretir: her biri aynı disjoint topical kümenin sekiz token'ını iki
kez içeren **16-token** kayıtlar; küme başına 16 kayıt, toplam 128 kayıt. İki
epoch için:

```text
Training token/run        4,096
Ordered context pair/run  14,848
```

Araç `hybrid`, `hierarchical` ve `coherent` modlarını ayrı checkpoint'lerle
çalıştırır. `hierarchical`, local `hybrid` representation'ını korur ve uzun
kayıtlarda satır-base-vector majority özetini üç bounded katkı olarak ekler.
`coherent` aynı üç-item global bütçeyi `full-record × 1 + center'ın opposite-
parity fold'u × 2` olarak dağıtır; böylece focal occurrence iki fold katkısından
çıkar, ama full-record anchor'da bir kez kalır. Fold'lar record başına immutable
base vectors üzerinden hazırlanır.

`hierarchical` için acceptance signal: same-cluster Top-1 gerilemez ve mean
Top-1 similarity gain `> 0.002` olur. `coherent` için daha muhafazakâr guard:
same-cluster Top-1, `hybrid`e göre gerilememelidir; ölçülen similarity farkı
raporlanır fakat pozitif gain zorunlu değildir. Bu bilerek tasarlanmış in-corpus
cohesion testidir; held-out task başarısı veya doğal-dil anlama kanıtı değildir.
`coherent` CPU/OpenMP-only'dir; mevcut OpenVINO immutable-item ABI'si
center-dependent cross-fold context'i taşımaz.

### Coherent ilk doğrulanmış controlled sonuç

[GitHub Actions run 39980147491](https://github.com/sswweett00/Vegai/actions/runs/39980147491)
(AMD EPYC 7763 / Linux, `OMP_NUM_THREADS=1`) bu üç-arm diagnostic'i, native
Fortran testleri ve historical baseline byte-equality kontrolüyle birlikte geçti.
Aynı 16-token fixture'da mean Top-1 similarity `hybrid` için `0.168998`,
`hierarchical` için `0.826843`, `coherent` için `0.647824` oldu. Böylece
coherent'in hybrid'e karşı farkı **`+0.478826`** idi ve same-cluster Top-1
non-regression guard'ı geçti. Bu sonuç coherent'in bu kontrollü fixture'da
hierarchical'i geçtiği iddiası değildir; tasarımın amacı fixed global bütçede
self-echo'yu azaltmak ve measurable non-regression sağlamaktır. Süre, yalnız
`train` child-process'i için workflow summary'de raporlanır; herhangi bir
CPU/OS'a veya doğal corpus'a genellenmez.

### DEBIASED structural invariant protocol

`debiased`, `compare_pretraining_modes.py` içindeki cohesion score'a yeni bir
universal-quality arm olarak eklenmez: duplicate handling'in daha iyi olduğu
hedef domain'e bağlıdır ve repository'nin sentetik cluster fixture'ı bunu
kanıtlayamaz. Bunun yerine native property test iki fail-closed structural oracle
uygular:

1. 12 unique active slotlu long record için global item sırası aynı olduğundan
   `debiased` checkpoint-relevant base/semantic/experience planes'i
   `hierarchical` ile byte-tam eşit olmalıdır.
2. 12 raw occurrence fakat yalnız dört distinct active slotlu repeat-heavy
   long record için global channel kapalıdır; aynı planes `hybrid` ile byte-tam
   eşit olmalıdır.

Test ayrıca distinct/suppressed execution telemetry'yi, v7 checkpoint round-
trip'ini, legacy sampler lineage rejection'ını ve unsupported OpenVINO ABI
rejection'ını kontrol eder. Bu, controlled semantic quality gain değil;
repetition politikasının tanımlı ve denetlenebilir olduğu kanıtıdır. Deployment
kalitesi için operator-owned held-out cases/policy ile
`tools/pretraining_quality_evaluation.py` veya `tools/evidential_optimizer.py`
gate'i gerekir.

## Exact vocabulary lookup-cache child-process comparison

`tools/compare_pretraining_lookup_cache.py`, cache açık ve kapalı armı gerçek
`vegai_pretrain train` process'leriyle çalıştırır. Timer yalnız child-process
başlatma/bitiş aralığını kapsar; compilation, fixture generation, checkpoint
comparison ve JSON/harness dışarıdadır. Arm sırası her repetition'da değişir.

```bash
python3 tools/compare_pretraining_lookup_cache.py \
  --pretrain-bin ./build/vegai_pretrain --repetitions 3 \
  --output lookup-cache-comparison.json
```

Fixture 64 kayıt × 64 fixed-width active token × iki epoch kullanır. Tokenlar
bilerek aynı canonical vocabulary hash start indexini paylaşır, fakat direct-cache
indexleri distincttir; böylece collision-sensitive lookup hot-path görünür olur.
Bu natural-corpus throughput, full model training speed veya semantic quality
için benchmark değildir. Cache armında
`hits + misses = raw_tokens`, iki değerin de pozitif olması; cache-off armında
ikisi de sıfır olması ve iki checkpoint byte/SHA'sının eşit olması zorunludur.
Raporlanan `cache_speedup_vs_no_cache` minimum eşik değildir: CPU topology,
compiler, filesystem, process startup ve thermal durum etkileyebilir.

### İlk doğrulanmış lookup-cache CI gözlemi

[GitHub Actions run 40103290586](https://github.com/sswweett00/Vegai/actions/runs/40103290586)
(commit `28f0eaa`, AMD EPYC 7763 64-Core Processor, hosted Linux runner; üç
alternating repetition) gerçek child-process ölçümünü başarıyla tamamladı:

| Arm | Median train s | Training token/s | Cache hit / miss | Checkpoint |
|---|---:|---:|---:|---|
| canonical hash lookup (`--no-lookup-cache`) | 0.400487 | ≈20,455.096 | 0 / 0 | cache arm ile byte-tam eşit |
| exact-token direct-map (`--lookup-cache`) | 0.398502 | ≈20,556.986 | 8,064 / 128 (her run) | **byte-tam eşit** |

Bu controlled collision-sensitive fixture için ratio **1.0050×** oldu. Küçük
fark hosted-runner varyansından ayrı bir general speedup kanıtı değildir; CI'nın
asıl acceptance koşulları cache hit/miss partition ve semantic checkpoint
eşitliğidir. Farklı vocabulary load factor, record topology, CPU ve corpus'ta
ayrı ölçüm gereklidir.

## Aynı-run baseline → hot-path mimarisi karşılaştırması

`tools/compare_pretraining_revisions.py`, sabit pre-optimization commit
`e34fd22` ile aday binary'yi **aynı runner** üzerinde portable `make pretrain`
profiliyle build eder. Her iki binary için önce ölçüm-dışı warm-up yapılır,
sonra sıra her repetition'da ters çevrilir. Böylece cache/termal durumun yalnız
sonra çalışan binary'ye avantaj vermesi azaltılır. Her timed sample tam bir
`vegai_pretrain train` child process'idir; compilation, fixture üretimi,
checkpoint compare, JSON ve harness süresi hariçtir. Her workload için
varsayılan modda baseline/optimized checkpoint byte dizisinin ve her binary'nin
kendi tekrar çalışmalarının tam eşit olması zorunludur. Sadece versioned bir
algoritma/checkpoint lineage migration'ı ayrıca review edilmişse,
`--checkpoint-contract versioned-algorithm-transition` açıkça seçilir: iki
binary'nin kendi tekrarları byte-tam eşit kalır, cross-lineage checkpoint'ler
bilerek farklı olmalıdır ve candidate gerçek baseline checkpoint'ini byte-tam
resume etmelidir. Timing ratio semantic-equivalence speedup olarak sunulmaz.

CI'nın kullandığı iki bilinçli bottleneck fixture şunlardır:

| Fixture | Record | Epoch | Training token/run | Amaç |
|---|---:|---:|---:|---|
| `short_records` | 256 × 8 active token | 2 | 4.096 | bounded bitacc reset, staging wipe removal, kısa-record hierarchy skip ve adaptive serial scheduler |
| `repeated_centers` | 8 × 1.024 active token | 2 | 16.384 | repeated-center source-order planner'ın eski backward quadratic scan'ini kaldırma |

[GitHub Actions run 39974078629](https://github.com/sswweett00/Vegai/actions/runs/39974078629)
(commit `5848420`, AMD EPYC 9V45 96-Core / Linux `7.0.0-1017-azure`,
`OMP_NUM_THREADS=2`, üç timed repetition) aşağıdaki **aynı-run** sonucu verdi:

| Fixture | e34fd22 median train s | Optimize median train s | Önce token/s | Sonra token/s | Sonra/önce | Checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `short_records` | 0.061195 | 0.058502 | ≈66,933.573 | ≈70,014.700 | **1.0460×** | **byte-tam eşit** |
| `repeated_centers` | 0.129973 | 0.095075 | ≈126,056.950 | ≈172,327.110 | **1.3670×** | **byte-tam eşit** |

Bu, compiler/CI setup değil yalnız `train` process wall-clock ölçümüdür ve
checkpoint equality ile semantic preservation doğrulanmıştır. Fakat fixture'lar
özellikle mimari darboğazları görünür kılmak için sentetiktir; doğal corpus,
I/O, vocabulary cardinality, OpenMP runtime, SSD/cache ve termal koşullarda
aynı oran vaat edilmez. Kendi iki compiled binary'nizi aynı koşulda ölçmek için:

```bash
# Baseline source ve portable binary (repo root dışındaki geçici worktree önerilir)
git fetch origin e34fd22e485807108dd85957d869a2d880325dca
git worktree add --detach /tmp/vegai-e34 e34fd22e485807108dd85957d869a2d880325dca
make -C /tmp/vegai-e34 pretrain BUILD_DIR=/tmp/vegai-e34-build
make pretrain
python3 tools/compare_pretraining_revisions.py \
  --before-pretrain-bin /tmp/vegai-e34-build/vegai_pretrain \
  --after-pretrain-bin ./build/vegai_pretrain --repetitions 3
```

v5/v6 → v7 exact-mask sampler migration'ını ölçerken contract açık olmalıdır:

```bash
python3 tools/compare_pretraining_revisions.py \
  --before-pretrain-bin /tmp/vegai-e34-build/vegai_pretrain \
  --after-pretrain-bin ./build/vegai_pretrain --repetitions 3 \
  --checkpoint-contract versioned-algorithm-transition
```

Bu ikinci komut cross-lineage checkpoint farklılığını bekler; ortaya çıkan
runtime ratio ancak bu spesifik version transition için descriptive telemetry'dir.

### v7 sampler transition: gözlenen CI telemetry

[GitHub Actions run 40090215964](https://github.com/sswweett00/Vegai/actions/runs/40090215964)
(commit `fe184a9`, AMD EPYC 7763 64-Core Processor / Linux `7.0.0-1017-azure`,
`OMP_NUM_THREADS=2`, üç timed repetition) historical v5/v6 sampler lineage'i
ve v7 Floyd/complement lineage'i arasında aşağıdaki ölçümü verdi:

| Fixture | Baseline median train s | v7 median train s | Baseline token/s | v7 token/s | v7/baseline runtime ratio | Checkpoint contract |
|---|---:|---:|---:|---:|---:|---|
| `short_records` | 0.094152 | 0.092003 | ≈43,504.121 | ≈44,520.287 | 1.0234× | intentional versioned algorithm transition |
| `repeated_centers` | 0.202346 | 0.157675 | ≈80,970.219 | ≈103,909.941 | 1.2833× | intentional versioned algorithm transition |

Her taraf kendi repeat'inde deterministic kaldı. Transition contract ayrıca yeni
binary'nin gerçek baseline checkpoint'ini bir ek epoch için byte-tam resume
etmesini zorunlu kıldı ve CI bu gate'i geçti. Bununla birlikte fresh v7
checkpoint'i v5/v6 checkpoint'iyle bilerek aynı değildir; bu iki runtime oranı
semantic-equivalent optimizasyon veya genel throughput/capability iddiası olarak
okunmamalıdır. Compile, fixture, checkpoint comparison ve harness süreleri
ölçüye dahil değildir.

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

[GitHub Actions run 39851742002](https://github.com/sswweett00/Vegai/actions/runs/39851742002)
(commit `f5fc4ab`) optional OpenVINO runtime'ı ile gerçek Fortran↔Python
lane-worker bridge'ini tekrar çalıştırdı. Worker artık bir train süresince
resident, `Core`/dynamic-batch compiled graph cache'i request'ler arasında
paylaşılıyor. Actual edge cardinality'leri seeded neutral `(tie, NOT tie)`
pair'leri ile iki reusable odd/even graph shape'ına normalize edilir;
source-order Hebbian commit Fortran CPU'da kalır. Strict CPU reference ve
provider checkpoint byte'ları tam eşitti.

| Arm | Gerçek `vegai_pretrain train` süresi | Dispatched context | Checkpoint |
|---|---:|---:|---|
| portable CPU reference | `0.021218 s` | 0 | regression arm |
| normal explicit CPU fallback → native fused/OpenMP bypass | `0.020939 s` | 0 | **byte-tam eşit**, 0 worker session/request |
| önceki process-per-line strict CPU bridge | `1.438524 s` | 30 | byte-tam eşit |
| resident/cached strict OpenVINO **CPU plugin** bridge | `0.427948 s` | 30 | **byte-tam eşit**, 1 session / 2 request |

Strict bridge ratio `67.9080×` historical baseline'dan **`20.1696×`**e indi;
absolute bridge süresi yaklaşık **3.36×** azaldı. Bu, cold Python/OpenVINO
startup maliyetinin tamamını değil per-line startup ve repeated-shape compile
maliyetini ortadan kaldırır. Dolayısıyla strict CPU plugin armı hâlâ CPU
hızlandırması iddiası değildir.

Production CPU policy ayrıdır: fallback açıkken explicit `--openvino-device
CPU`, worker açmak yerine native fused/OpenMP context construction'a gider;
runtime output `native-CPU bypass` sayacını yükseltir, OpenVINO device execution
mask'i ve worker session/request sayaçları sıfır kalır. CI bu armda CPU reference
ile `0.9869×` timing ratio ve byte-eşit checkpoint doğruladı. Ölçü yalnız
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

- `hv_bitacc_add_words`, `hv_bitacc_add_permuted_words`,
  `hv_bitacc_add_permuted_pair_words` ve mixed
  `hv_bitacc_add_permuted_mixed_pair_words`, geçici `hv_t` kopyasını kaldırarak
  aynı bit-sliced population-counter arithmetic'ini doğrudan vocabulary'nin
  SoA column'unda uygular.
- Native property testleri negatif/pozitif shift, bütün counter plane'leri,
  sample count ve pair-overflow atomicity'sini kontrol eder.
- `--reference-context-kernel` checkpoint formatına yazılmaz; yalnız benchmark
  ve regression oracle'dır. Aynı semantic config ile fused/reference checkpoint
  byte eşitliği bu nedenle zorunludur.
- Checkpoint version sampler ve feedback lineage'ini de bağlar: ordinary fresh
  mode'lar v7 header+checksum ile Floyd/complement exact-mask sampler'ını yazar;
  `resonant` frozen-semantic feedback contract'i için v8 yazar. Reader v4/v5/v6'yı
  historical bounded-retry sampler olarak kabul eder ve bu lineage yeniden
  kaydedildiğinde v5 (non-coherent) veya v6 (coherent) sürdürülür. Coherent
  semantic mode taşıyan v4/v5 ve resonant taşıyan pre-v8 payload fail-closed
  reddedilir.
- Hierarchical global signal satırda bir kez üretilir, yalnız ≥12 active token
  için etkinleşir ve local context accumulator kapasitesiyle bounded kalır;
  eşik altındaki record için topic accumulator kurulmaz. Coherent aynı eşikte
  bir full-record ve iki parity fold özetini record başına bir kez üretir;
  center bundle'ı full özeti bir, opposite-parity fold özetini iki kez ekler.
- Bounded bit accumulator reset'i yalnız bilinen cardinality'nin erişebileceği
  carry plane'lerini yazar; daha yüksek plane'e geçiş lazy-clear ile güvenlidir.
  Odd majority finalization tie vector üretmez. Bu execution detayı
  fused/reference semantic equality sözleşmesini değiştirmez.
- Repeated-center experience planner'ı capacity-indexed occurrence scratch ve
  sparse touched-slot reset kullanır; eski per-center backward scan'in quadratic
  bölümünü kaldırır. Active-prefix count da bütün window neighbor cardinality'lerini
  O(token) belirler. Resource plan scratch için `4 × max-vocab` byte ayırır.
- CPU scheduler küçük record'larda OpenMP team açmak yerine serial immutable
  context path'ini seçer (`--parallel-min-contexts`, default 32). Controlled
  long-record timing harness'i output telemetry ile parallel branch'in gerçekten
  seçildiğini doğrular; commit yine source order'dadır.
