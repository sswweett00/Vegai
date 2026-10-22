# Corpus pretraining: deterministic HDC/VSA context learning

> **Terminoloji sınırı:** Bu modül Transformer, attention katmanları, token
> olasılıkları, backpropagation veya genel amaçlı bir LLM eğitmez. `pretraining`
> adı burada büyük bir corpus üzerinden tekrar kullanılabilir **dağıtımsal
> HDC/VSA semantic prototype** üretmek anlamındadır. Modelin kapasitesi,
> retrieval kalitesi ve yetenekleri bir dil modelinin kapasitesiyle eşdeğer
> değildir.

Vegai'nin corpus hattı `src/text_tokenizer.f90`, `src/vocabulary.f90` ve
`src/hdc_pretraining.f90` içinde yer alır. Hedef; bir corpus'u sabit bellek,
bit-tam determinism, açık veri-kayıp telemetrisi ve checkpoint/resume
sözleşmesiyle işlemektir.

## 1. Öğrenme semantiği

Her aktif sözlük token'ı `w` için iki adet 16.384-bit hiper-vektör vardır:

- `B(w)`: token metni + first-seen slot'tan türetilen, deterministik ve sabit
  **base identity**;
- `S(w)`: başlangıçta `B(w)` olan, stream boyunca güncellenen **semantic
  prototype**.

Bir satırdaki aktif center token `wᵢ` için, `r ∈ [-W, W] \ {0}` konumlarındaki
aktif komşular alınır. Konum bilgisi bir dairesel permütasyonla kodlanır.
Varsayılan **hierarchical** encoder, `hybrid`in positional syntax sinyalini ve
order-invariant topical co-occurrence sinyalini aynı deterministic bundle içinde
birleştirir; yalnız yeterince uzun records için ayrıca bounded whole-record
topical evidence ekler. Local hybrid çekirdeği şöyledir:

```text
Vpos = { Π^(r × P)(B(wᵢ₊ᵣ)) ; her geçerli r için }
Vtopic = { B(wᵢ₊ᵣ) ; her geçerli r için }
Cᵢ = majority_tie_seeded(Vpos ∪ Vtopic)
S(wᵢ) ← replace_exact_fraction(S(wᵢ), Cᵢ, learning_rate(experience(wᵢ)))
```

`hierarchical` modunda, bir satırda en az 12 active token olduğunda önce bir kez
`L = majority_tie_seeded({B(wⱼ)})` kurulur ve her center için `Cᵢ` bundle'ına
`L, L, L` eklenir. Bu üç katkı sabittir; context satır sınırını geçmez ve
bit-sliced accumulator kapasitesi tarafından üstten sınırlandırılır.

- `W`, context window; `P`, `position_shift_bits` değeridir.
- `hierarchical` varsayılandır: `hybrid`in left/right role bilgisini `Vpos`
  içinde ve topical neighbourhood sinyalini `Vtopic` içinde korur; yalnız ≥12
  active-token record için aşağıdaki bounded whole-record evidence'i ekler.
  Kısa record'larda bit-tam `hybrid` ile aynıdır.
- `hybrid`, explicit local-only baseline/reproducibility seçeneğidir: positional
  syntax ve order-invariant topical sinyali birleştirir, fakat whole-record
  channel eklemez.
- `positional`, yalnız `Vpos` kullanır. Bu modun bit-sliced implementation'ı,
  eski `legacy` signed-counter implementation'ı ile unit-weight context için
  bit-tam eşdeğerdir.
- `hierarchical`, `hybrid` yerel bundle'ını aynen korur; yalnız bir kayıtta
  en az 12 active token varsa bütün kaydın base-vector majority özetini üç
  bounded katkı olarak her eligible center'a ekler. Bu uzun-kayıt topical
  cohesion için kontrollü bir seçenektir; document understanding iddiası
  değildir. Sekiz-token örnek fixture bu eşiğin altında kalır ve etkilenmez.
- `legacy`, önceki positional signed-counter yolu için regression/benchmark
  referansıdır; yeni eğitimler için önerilmez.
- `majority_tie_seeded`, yeni modlarda word-parallel `hv_bitacc_t` binary
  population counter ve deterministic tie breaker kullanır. Bu, 16.384 bit
  üzerinde tek tek branch yerine packed XOR/AND carry ile çalışır.
- Güncelleme gradient değildir. `hv_hebbian_update_percent`, seçilecek
  **boyutların** cardinality'sini tam olarak belirler; eski prototype bitleri
  seçilmeyen boyutlarda korunur. Context ve eski bit aynıysa nihai Hamming
  değişiminin bundan küçük olması normaldir.
- Schedule deneyim `e` için `e < D` initial, `D <= e < 4D` yaklaşık yarı,
  `4D <= e < 16D` yaklaşık çeyrek ve `e >= 16D` minimum oranını kullanır
  (`D = plasticity_decay_events`). Varsayılan `25 → 13 → 7 → 1`dir.

Bu yapı aynı yerel bağlamda görülen token'ların prototype'larını korele
edebilir. Ancak negation, uzun menzilli syntax, factual recall, generation veya
kalibre edilmiş olasılık tahmini garantisi vermez. Semantic komşu çıktısı bir
**diagnostic/retrieval sinyali**dir; insan değerlendirmesi ve görev-özel test
olmadan doğruluk iddiası değildir.

## 2. İki geçişli veri hattı

### Geçiş 1 — vocabulary

1. Corpus fiziksel satır satır okunur.
2. ASCII tokenizer token'ları canonical lower-case biçiminde çıkarır.
3. Fixed-capacity open-addressing vocabulary her unique token'ın frequency'sini
   sayar.
4. Geçiş sonunda `frequency >= min_frequency` olan slotlar active yapılır.
5. Her active token'ın `B(w)` ve başlangıç `S(w)` hiper-vektörü kurulur.

Vocabulary sınırı aşıldığında işlem `PT_ERR_VOCABULARY` ile **durur**; yüksek
frequency token'ları sessizce düşüren approximate top-k politikası yoktur.
`--max-vocab` değeri corpus'un unique-token sayısını karşılayacak biçimde
seçilmelidir. Bu tercih, frequency sonuçlarının deterministik ve denetlenebilir
kalmasını sağlar.

Hash lookup fingerprint kullanır, fakat fingerprint eşitliğini token eşitliği
saymaz: her hit'te tam padded-string karşılaştırması yapılır. Böylece 64-bit
fingerprint collision yanlış token merge'ine neden olmaz. Checkpoint load
sonrasında transient hash table tokenlardan yeniden kurulur.

### Geçiş 2 — online context learning

Aynı corpus her epoch için yeniden stream edilir. Vocabulary dışı veya
min-frequency eşiğinin altında kalan token'lar `oov_tokens` olarak sayılır;
bunlar center ya da neighbor olarak prototype güncellemesine girmez. Context
satır sınırını geçmez: yeni satır yeni context segmentidir.

Varsayılan bellek maliyeti capacity başına yaklaşık şudur:

```text
base identity       2,048 byte
semantic prototype  2,048 byte
experience counter      8 byte
vocabulary + hash    token kapasitesine bağlı
```

Dolayısıyla 20.000 slotun yalnız iki HDC düzlemi yaklaşık 78.1 MiB'dir;
vocabulary/hash ve allocator overhead buna eklenir. Çalışma zamanında
`pretrain_model_footprint_bytes` tahmini resident payload'u raporlar.

## 3. Input contract ve veri kaybı telemetrisi

Tokenizer bilerek Unicode tokenizer değildir. Her byte ASCII olarak ele alınır:

- token karakterleri `A-Z`, `a-z`, `0-9`, `_`, `'`;
- `A-Z` ASCII lowercase'e dönüşür;
- UTF-8 multibyte byte'ları delimiter davranışı gösterir ve kısa kayıtlar için
  `non_ascii_bytes` telemetrisi artırılır;
- token uzunluğu en çok 48 byte, satırda en çok 1.024 token ve fiziksel satır
  en çok 8.192 byte olabilir.

Uzun token veya token-count limiti aşılırsa ilgili token **truncate edilmez**;
tamamen düşürülür ve `tokenizer_truncations` artırılır. 8.192 byte'tan uzun
fiziksel kayıt ilk 8.192 byte ile öğrenilmez: tamamı drain edilir, düşürülür ve
`overlong_records` artırılır. Bu davranış prefix-aliasing ve sessiz corpus
bozulmasını önler.

`--strict` etkinse üç durumda eğitim durur:

- tokenizer'ın token/token-count limitini aşması: `PT_ERR_TOKENIZATION`;
- `>127` non-ASCII byte görülmesi: `PT_ERR_TOKENIZATION`;
- fiziksel kayıt limiti aşması: `PT_ERR_RECORD_TOO_LONG`.

Türkçe veya başka Unicode dillerindeki corpus'lar için normalizasyon, segmentasyon
ve reversible ASCII-safe temsil (ör. ayrı sentence/token dosyası veya kontrollü
transliteration) **önceden** yapılmalıdır. Input record delimiter'ını da
LF'ye normalize edin; manifest preflight'i CRLF kayıtlarını telemetry olarak
raporlar ama onları sessizce normalize etmez. Hangi dönüşümün kullanıldığı
deney manifestinde yazılmalıdır. UTF-8 byte'larını bu tokenizer'a doğrudan
verip Unicode kelime semantiği beklemek geçerli değildir.

## 4. CLI

CMake hedefi `vegai_pretrain`, Make hedefi `make pretrain`dir.

```bash
cmake -S . -B build -DVEGAI_BUILD_TESTS=ON
cmake --build build --parallel

# Yeni iki-pass model
./build/vegai_pretrain train CORPUS.txt model.vgai \
  --max-vocab 50000 --min-frequency 3 --window 4 --position-shift 73 \
  --context-mode hierarchical --epochs 3 --initial-plasticity 25 --minimum-plasticity 1 \
  --decay-events 32 --query token

# Checkpoint semantic configuration ile uyumlu ek epoch (vocabulary değiştirmez)
./build/vegai_pretrain resume CORPUS.txt model.vgai --epochs 1 --query token

# Yeni immutable shard'dan vocabulary'yi genişletir, sonra sadece o shard'ı öğrenir
./build/vegai_pretrain continue NEW_SHARD.txt model.vgai --epochs 1 --strict --query newtoken

# Eğitim yapmadan metadata ve tek nearest-neighbour sorgusu
./build/vegai_pretrain inspect model.vgai --query token

# Tek checkpoint load ile dosyadaki her nonblank token için batch diagnostic
./build/vegai_pretrain inspect model.vgai --query-file evaluation-queries.txt
```

`train` options:

| Seçenek | Varsayılan | Anlam |
|---|---:|---|
| `--max-vocab N` | 20000 | fixed vocabulary capacity |
| `--min-frequency N` | 2 | active vocabulary threshold |
| `--window N` | 4 | her yönde context yarıçapı |
| `--position-shift N` | 73 | relative position başına bit shift |
| `--context-mode MODE` | `hierarchical` | `legacy`, bit-tam hızlı `positional`, local positional+topical `hybrid` veya `hybrid`e yalnız ≥12 active-token kayıtta bounded whole-record topical evidence ekleyen kalite-odaklı `hierarchical` |
| `--parallel-context-build` | açık | immutable per-record context construction için OpenMP; source-ordered prototype commit sayesinde thread count checkpoint semantiğini değiştirmez |
| `--serial-context-build` | kapalı | yalnız debug/regression için parallel context hazırlığını kapatır; checkpoint state değildir |
| `--epochs N` | 1 | ikinci corpus geçişi sayısı |
| `--initial-plasticity N` | 25 | başlangıç selected-dimension yüzdesi |
| `--minimum-plasticity N` | 1 | schedule alt sınırı |
| `--decay-events N` | 32 | token başına ilk schedule eşiği |
| `--strict` | kapalı | drop/truncation durumunu hata yap |
| `--reference-context-kernel` | kapalı | yalnız benchmark/regression için önceki rotate-then-add kernelini kullanır; varsayılan fused implementation ile checkpoint byte'ları bit-tam eşit olmalıdır |
| `--query TOKEN` | yok | eğitimden sonra veya `inspect` sırasında bir semantic neighbour göster |
| `inspect --query-file PATH` | yok | her nonblank satırdaki tek token için tek checkpoint load ile batch nearest-neighbour diagnostic |

`resume`, checkpoint'ten vocabulary threshold/capacity, `window`, shift,
context encoder, plasticity ve tie seed'ini geri alır; yalnız `--epochs`,
`--strict`, `--query` ve execution-only `--reference-context-kernel`,
`--parallel-context-build` / `--serial-context-build` seçeneklerini kabul eder
ve vocabulary'yi değiştirmez. Bu implementation/execution switches checkpoint
semantiği veya payload değildir; reference switch `continue` için de yalnız
benchmark/regression amacıyla kullanılabilir. `continue` aynı
uyumlu config'i geri alır, önce yeni shard'ın token'larını vocabulary'ye ekler,
yalnız newly-active slotları başlatır, ardından shard'ı train eder. Öğrenme
semantiğini checkpoint'ten farklı bir config ile sürdürme
`PT_ERR_INCOMPATIBLE_CONFIG` ile reddedilir. CLI query'si tokenizer üzerinden
canonicalize edilir ve yalnız bir token kabul eder.

Crash-safe append-only shard ledger, SHA-256, strict LF/ASCII contract ve
exactly-once durable generation semantics için
[CONTINUOUS_PRETRAINING.md](CONTINUOUS_PRETRAINING.md)'yi kullanın. Tek başına
`continue`, checkpoint'i verilen path üzerinde güncellediği için controller'ın
private candidate + journal transaction garantisinin yerini tutmaz.

Küçük smoke corpus ve dengeli Türkçe-ASCII fixture için
[`examples/`](../examples/README.md) altındaki örneğe bakın. İkinci fixture,
manifest digest'i ve gerçek `train → inspect → resume` CLI testiyle doğrulanır;
çalıştırılan counter, schedule ve checkpoint ayrıntıları
[EXAMPLE_PRETRAINING_RUN.md](EXAMPLE_PRETRAINING_RUN.md)'dedir. Hiçbiri
semantic kalite benchmarkı değildir.

## 5. Kaynak-duyarlı çalışma ve çok çekirdek

Corpus streaming olduğundan RAM gereksinimi corpus file boyutundan çok
`--max-vocab` ve HDC model capacity'sine bağlıdır. Eğitimden önce aynı corpus,
checkpoint path ve capacity ile aşağıdaki preflight'i çalıştırın:

```bash
python3 tools/training_resource_plan.py --corpus CORPUS.txt \
  --checkpoint checkpoints/model.vgai --max-vocab 50000 --require-ready
```

Araç RAM/storage lower-bound estimate'i, CPU topology, NUMA/RAM-channel
best-effort telemetry ve OpenMP placement önerisi üretir. SSD/HDD ve dual-RAM
channel seçimi OS/firmware tarafından yönetilir; Vegai bunları zorlamaz.
Deterministic OpenMP context construction, immutable base vectors üzerinde
paralel çalışır; model update'leri kaynak sırası ile commit edilir. Bu yüzden
`OMP_NUM_THREADS=1` ve çok-thread run checkpoint byte'ları eşit olmalıdır.

```bash
export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores
python3 tools/compare_pretraining_threads.py --pretrain-bin ./build/vegai_pretrain \
  --parallel-threads 2 --repetitions 3
```

Bu benchmark'ta compile/fixture/harness süresi hariçtir ve hız oranı yalnız
ölçülen host/workload için geçerlidir. GPU/NPU adapter inventory'si tek başına
compute kullanımı değildir. Intel Core Ultra için opt-in OpenVINO context
provider, immutable packed-HDC bundle/finalize aşamasını dispatch ederken
source-order update'i CPU'da tutar; worker bir train çağrısı boyunca resident
kalır ve item-cardinality graph'larını dynamic batch ile cache'ler. Explicit
CPU fallback, CPU plugin'i hızlandırma diye sunmak yerine native fused/OpenMP
kernel'ine bypass eder. Device mask'i runtime output'ta doğrulanır ve strict
CPU-reference checkpoint byte equality zorunludur. Ayrıntılı kaynak ve
accelerator sınırları için [RESOURCE_AWARE_TRAINING.md](RESOURCE_AWARE_TRAINING.md)'ye
bakın.

## 6. Metrikler ve yorum

CLI ve `pretrain_metrics_t` aşağıdaki sayıları verir:

| Metrik | Anlam |
|---|---|
| `corpus_lines` | işlenen fiziksel kayıt sayısı; overlong kayıtlar da dahil |
| `raw_tokens` | kabul edilen tokenizer token'ları |
| `active_tokens` | ikinci geçişte active vocabulary içinde olan token'lar |
| `oov_tokens` | vocabulary içinde olmayan veya inactive olan token'lar |
| `context_pairs` | position-permuted neighbor katkılarının sayısı |
| `center_updates` | en az bir active neighbor bulan center update sayısı |
| `tokenizer_truncations` | düşürülen long-token / excess-token kaydı |
| `non_ascii_bytes` | in-contract kayıtlarda gözlenen `>127` byte sayısı; non-strict telemetry'dir, `--strict` altında hatadır |
| `overlong_records` | tamamı düşürülen >8.192-byte fiziksel kayıt |

`active_tokens + oov_tokens = raw_tokens` ikinci geçiş için beklenen temel
invariant'tır. `center_updates` active token sayısından küçük olabilir; örneğin
satırda tek active token varsa context oluşmaz. Metrikler corpus kalitesini veya
semantic accuracy'yi tek başına ölçmez; ingest ve öğrenme kapsamını kanıtlayan
telemetridir.

## 7. Checkpoint ve resume sözleşmesi

`pretrain_checkpoint_save` stream-unformatted bir `.vgai` dosyası yazar;
`pretrain_checkpoint_load` magic/version, shape, counter sign, vocabulary
frequency/active invariantları, rebuilt hash index ve payload checksum kontrolü
yapar. Writer sürümü şu an **5**tir. v5, `hierarchical` context-mode enum'unu
kabul eder ve v5 header değeriyle checksum üretir; reader hem v4 hem v5
checkpoint'i, checkpoint'in kendi version değeriyle checksum hesaplayarak
okur. Böylece mevcut v4 payload layout'u okunabilir kalır; v3 ve daha eski
checkpoint'ler reddedilir ve corpus manifesti + source commit ile yeniden train
edilmelidir. `--reference-context-kernel` bilerek checkpoint state
olmadığından fused/reference run'ları aynı semantic model için byte-tam aynı
checkpoint üretmelidir. Checksum yalnız accidental corruption tespiti içindir;
kriptografik bütünlük/imza değildir.

Checkpoint aşağıdakileri taşır:

- capacity, token strings, frequencies, active mask ve min frequency;
- context mode/position/plasticity/tie-seed semantiği;
- completed epoch sayısı ve aggregate eğitim metrikleri;
- base/semantic hypervector planes ve token experience sayaçları.

Stream-unformatted dosya formatı aynı compiler/ABI/endianness ailesinde resume
edilmek üzere tasarlanmıştır; çapraz compiler, mimari veya gelecekteki format
sürümü arasında taşınabilir model interchange değildir. Uzun vadeli model
saklama için corpus manifesti + source commit + compiler bilgisi ile checkpoint
birlikte saklanmalıdır.

Kayıt esnasında I/O hatası sonrası model kısmen güncellenmiş olabilir; API bu
kısmi çalışmanın telemetrisini aggregate totals'a ekler, fakat
`training_epochs_completed` yalnız bütünüyle tamamlanan epoch'ları sayar.
Uygulamalar kesintisiz/atomik checkpoint ihtiyacı varsa dışarıda versioned
checkpoint path kullanmalı ve başarılı save sonrası manifest pointer'ını
atomik biçimde güncellemelidir.

## 8. Data governance ve deney manifesti

Vegai corpus indirme, scraping veya lisans doğrulama yapmaz. Kullanıcı input
verisi için sorumludur. Gerçek eğitim öncesi aşağıdaki minimum kontroller
uygulanmalıdır:

1. **Kaynak ve lisans:** URL/dataset sürümü, erişim tarihi, kullanım lisansı,
   ticari/redistribution koşulları ve attribution kaydedin.
2. **Rıza ve kişisel veri:** eğitim amacı için hukuki dayanak, consent/opt-out,
   çocuk verisi, hassas veri, PII/credential/secret taraması ve retention
   politikasını belgeleyin. Şüpheli veriyle eğitim yapmayın.
3. **Telif ve zararlı içerik:** hak sahibi kısıtları, takedown prosedürü,
   malware/prompt-injection/credential leak taraması ve erişim kontrolü
   uygulayın.
4. **Minimization:** yalnız gereken alanları alın; ham corpus, intermediate
   token dosyası ve checkpoint için erişim/şifreleme/retention sınırı tanımlayın.
5. **Değerlendirme ayrımı:** train/dev/test kaynak overlap'ini ve duplicate
   kayıtları denetleyin. Benchmark sonuçlarını training corpus'a sızdırmayın.

`tools/corpus_manifest.py` ağ erişimi yapmadan caller'ın sağladığı yerel
corpus için SHA-256, byte/record sayısı ve Vegai ASCII ingest preflight
telemetrisi üretir. Bu, hukuki/etik değerlendirmeyi otomatikleştirmez; manifest
içindeki source/license alanları kullanıcı beyanıdır.

```bash
python3 tools/corpus_manifest.py /secure/path/corpus.txt \
  --output run-manifest.json \
  --source 'dataset URI + version' --license 'license identifier' \
  --retrieved-at '2026-10-22T00:00:00Z' \
  --preprocessing 'normalizer 1.2; dedup 3' --strict-contract
```

`--strict-contract`, overlong record veya tokenizer drop bulursa nonzero çıkar.
`--expect-sha256` ile daha önce kaydedilmiş input digest'i yeniden doğrulanabilir.
Tool streaming scan yapar ve gerçek corpus'u manifest içine kopyalamaz.

Her run için version-control dışındaki bir manifestte en az şunlar bulunmalıdır:

```text
run_id, owner, purpose, source URIs/licenses, retrieval date,
corpus SHA-256 + byte/line count, preprocessing command + version,
Unicode/tokenization policy, dedup/PII filters, rejected-record metrics,
Vegai git commit, checkpoint SHA-256, compiler/version/flags,
CPU/OS, OMP_NUM_THREADS, full CLI config, evaluation protocol/results.
```

`.gitignore`, yerel `/corpora/`, `/checkpoints/` ve `.vgai` runtime çıktılarını
ignore eder. Lisansı ve dağıtım izni açık olmayan gerçek corpus/checkpoint'leri
bu depoya commit etmeyin.

## 9. Doğrulama ve kabul sınırları

Native testler şunları kontrol eder:

- exact-rate plasticity cardinality'si (complement input altında %25 = 4.096
  Hamming değişimi);
- bit-sliced `positional` bundle'ın legacy signed-counter bundle ile her
  cardinality'de bit eşitliği ve aynı corpus run'ında semantic plane eşitliği;
- fused raw-SoA/permuted/pair accumulator'ın reference eklemeyle her counter
  plane ve sample count için bit eşitliği; pair overflow'unda atomic state;
- fused `hybrid` corpus run'ının rotate-then-add reference kernel ile semantic
  plane, base plane ve learning schedule bakımından bit-tam eşitliği;
- repeated center token'lı record üzerinde serial ve OpenMP context construction
  run'larının base/semantic plane ve experience counter bakımından bit-tam eşitliği;
- ≥12 active-token controlled record üzerinde `hierarchical` replay
  determinismi, local-only hybrid'den ayrışması ve checkpoint round-trip'i;
- windowed context count ve active/OOV telemetry;
- checkpoint round-trip ve uninterrupted run ile checkpoint-resume run'ın
  bit-tam eşitliği;
- incompatible resume config, strict long token ve strict overlong record hata
  yolları.

CI ayrıca küçük corpus ile `train → inspect → resume` CLI smoke testini ve
optimized Make binary üzerinde `tools/compare_pretraining_modes.py` çalıştırır.
Araç, yalnız gerçek `vegai_pretrain train` child-process wall-clock'ını ölçer:
compiler/CI setup, fixture construction, inspect ve Python harness süreleri
hariçtir. Base 64-token fixture üzerinde `legacy → positional → fused hybrid`
oranlarını; aynı hybrid config için reference rotate-then-add ile fused kernel
checkpoint byte eşitliği + süre oranını; ayrıca 16-token-record kontrollü
fixture üzerinde `hybrid → hierarchical` Top-1 cohesion diagnostic'ini
raporlar. Bunlar kontrollü tanılardır; doğal dil yeteneği veya production
throughput garantisi değildir.

```bash
python3 tools/compare_pretraining_modes.py \
  --pretrain-bin ./build/vegai_pretrain --repetitions 3 --omp-threads 1
```

Kayıtlı optimized CI sonucu ve yorum için
[PRETRAINING_MODE_COMPARISON.md](PRETRAINING_MODE_COMPARISON.md)'ye; portable/native
CPU profilinin, iGPU fallback sınırlarının ve OS build yolunun ayrıntıları için
[HARDWARE_COMPATIBILITY.md](HARDWARE_COMPATIBILITY.md)'ye bakın.

Bu kontroller semantic usefulness, bias, güvenlik, gerçek dünya language quality
veya herhangi bir throughput iddiası değildir. Üretim kullanımı için kendi
lisanslı corpus'unuzda held-out retrieval, task accuracy, robustness, latency,
RAM ve privacy değerlendirmesi yapın.
