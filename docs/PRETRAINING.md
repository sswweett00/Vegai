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

`coherent` aynı uzun-record eşiği ve aynı üç global-item bütçesini kullanır,
fakat uzun satır için bir tam-record özeti `L` ile birlikte kaynak pozisyonunun
iki parity fold özetini bir kez üretir. 1-tabanlı kaynak pozisyonu `i` olan bir
center için `Cᵢ`ye `L` bir kez, **karşı parity** fold özeti iki kez eklenir.
Dolayısıyla center'ın o occurrence'ı iki cross-fold katkıda yer almaz; yalnız
tam-record anchor içinde bir kez görünür. Karşı fold boşsa güvenli fallback
`L, L, L`dir. Local positional+topic evidence aynen `hybrid`tir. Bu, bir
record'u insan gibi anlamak veya aynı token'ın bütün tekrarlarını dışlamak değil,
long-record self-identity echo'yu bounded ve deterministik biçimde azaltma
amaçlı yapısal bir HDC encoder seçeneğidir.

Opt-in **`debiased`** modunda local evidence yine bit-tam `hybrid`tir. Long-
record global channel ise raw occurrence listesiyle değil, source-order'da ilk
kez görülen **distinct active vocabulary slot** listesiyle kurulur:

```text
D = unique_source_order({ slot(wⱼ) | wⱼ active })
Ldistinct = majority_tie_seeded({ B(slot) | slot ∈ D })
Cᵢ = majority_tie_seeded(Vpos ∪ Vtopic ∪ { Ldistinct, Ldistinct, Ldistinct })
```

`|D| < 12` ise global üç katkı hiç eklenmez; sonuç local `hybrid` ile bit-
tam aynıdır. Bu nedenle çok tekrar eden boilerplate bir satırın yalnız occurrence
sayısıyla whole-record majority'yi ele geçirmesi engellenir. Bu bir data-quality
ve bounded aggregation seçeneğidir; tekrarın her domain'de yanlış olduğu veya
semantic quality'nin evrensel olarak arttığı iddiası değildir.

Opt-in **`resonant`** modunda position channel değişmez; epoch `e` başlamadan
önceki committed semantic plane bir kez kopyalanır ve bütün epoch boyunca salt-
okunur kalır:

```text
Ŝ_e(w) = S(w)  at the start of epoch e
Vpos      = { Π^(r × P)(B(wᵢ₊ᵣ)) }
Vfeedback = { Ŝ_e(wᵢ₊ᵣ) }
Cᵢ        = majority_tie_seeded(Vpos ∪ Vfeedback)
S(wᵢ)     ← replace_exact_fraction(S(wᵢ), Cᵢ, learning_rate(experience(wᵢ)))
```

Yeni modelde `Ŝ₁ = B` olduğundan ilk `resonant` epoch'u local `hybrid` ile
bit-tam aynıdır. İkinci ve sonraki epoch'lar yalnız önceki epoch'ta öğrenilmiş
neighbour prototype'larını topical channel'a taşır. Aynı epoch içinde yapılan
source-order commit'ler snapshot'a geri yazılmaz: serial/OpenMP scheduler,
record sırası veya checkpoint boundary'si context input'unu değiştiremez. Bu,
önceki learned structure'ın eğitim context'ine ölçülebilir biçimde katıldığı
bir semantic-retrieval iyileştirme mekanizmasıdır; doğal dil anlayışı, genel
zeka veya her corpus'ta kalite artışı garantisi değildir.

Snapshot checkpoint payload'u değildir; resume sonrası committed `S` plane'den
tekrar oluşturulur. Capacity başına tam `256 × 8 = 2,048` byte ek geçici RAM
kullanır (`max-vocab=20,000` için 40,960,000 byte) ve epoch sonunda serbest
bırakılır. `vegai_control resource-plan --context-mode resonant` bu reservation'ı
ve reserve fraction'ını fail-closed readiness hesabına ekler. Mevcut OpenVINO
immutable-item ABI'si semantic snapshot item'ı taşımadığından `resonant`
bilinçli olarak CPU/OpenMP-only'dir; OpenVINO'ya sessiz fallback yoktur.

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
- `coherent`, `hierarchical` ile aynı local evidence, ≥12-token eşiği ve üç
  global item sınırına sahiptir; global ağırlığı `full-record × 1 + opposite-
  parity fold × 2` olarak dağıtır. Fold'lar record başına bir kez immutable
  base vectors üzerinden hazırlanır; center başına quadratic scan veya mutable
  semantic-prototype okuması yoktur. Kısa record'larda bit-tam `hybrid` ile
  aynıdır. Mevcut OpenVINO worker ABI'si cross-fold items taşımadığından bu mod
  bilinçli olarak CPU/OpenMP ile sınırlıdır.
- `debiased`, local `hybrid` bundle'ını aynen korur; global majority'ye her
  active slotu en fazla bir kez verir ve yalnız ≥12 **distinct** active slotta
  üç `Ldistinct` katkısı ekler. Scratch tablosu capacity-indexed/touched-slot
  resetlidir; center başına duplicate scan, mutable prototype okuması veya
  corpus-size'a bağlı memory yoktur. Duplicate olmayan long record'larda
  `hierarchical` ile bit-tam eşdeğerdir; <12 distinct slotta `hybrid`e düşer.
  Distinct-slot item dizisi mevcut immutable OpenVINO ABI'sinde olmadığı için
  mod fail-closed olarak yalnız CPU/OpenMP'dir.
- `resonant`, local positional channel'ı `B` olarak sabit tutar ve topical
  neighbour channel'ında yalnız epoch-başında dondurulmuş `Ŝ` kullanır. İlk
  epoch `hybrid`le bit-tam aynıdır; sonraki epoch'larda semantic feedback başlar.
  2,048 byte/slot transient snapshot planlanmalıdır; snapshot checkpoint'e
  yazılmaz. Fused mixed-pair kernel ve rotate-then-add reference path her
  population-counter plane'inde eşit test edilir. Snapshot worker ABI'sinde
  olmadığı için mod fail-closed CPU/OpenMP-only'dir.
- `legacy`, önceki positional signed-counter yolu için regression/benchmark
  referansıdır; yeni eğitimler için önerilmez.
- `majority_tie_seeded`, yeni modlarda word-parallel `hv_bitacc_t` binary
  population counter ve deterministic tie breaker kullanır. Bu, 16.384 bit
  üzerinde tek tek branch yerine packed XOR/AND carry ile çalışır.
- Güncelleme gradient değildir. `hv_hebbian_update_tenths_percent`, seçilecek
  **boyutların** cardinality'sini yüzde puanının onda biri çözünürlüğünde sabit
  nokta aritmetiğiyle belirler; eski prototype bitleri seçilmeyen boyutlarda
  korunur. Context ve eski bit aynıysa nihai Hamming değişiminin bundan küçük
  olması normaldir.
- Schedule deneyim `e` için `e < D` initial, `D <= e < 4D` yaklaşık yarı,
  `4D <= e < 16D` yaklaşık çeyrek ve `e >= 16D` minimum oranını kullanır
  (`D = plasticity_decay_events`). Varsayılan `%25.0 → %12.5 → %6.3 → %1.0`dir.

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

Eğitim geçişinde varsayılan açık olan `--lookup-cache`, 4.096-entry bounded
direct-map execution cache'idir. Key; **fingerprint + tam 48-byte padded token**
şeklindedir: aynı cache index veya aynı fingerprint tek başına hit olamaz;
çakışma yalnız güvenli miss/overwrite üretir ve canonical vocabulary lookup'a
düşer. Active olmayan/OOV sonucu da exact token ile cache'lenebilir, ancak cache
checkpoint'e yazılmaz ve `--lookup-cache` / `--no-lookup-cache` seçimleri
semantic plane, experience counter veya resume sonucu değiştirmemelidir. CLI
`vocabulary lookup cache hits/misses` sayaçlarını yalnız execution telemetrisi
olarak raporlar.

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

### 2.1. Ölçülen hot-path mimarisi ve semantik sınırlar

Eğitim hot path'i compiler flag ile değil, aşağıdaki bounded execution değişimleri
ile azaltılır. Bunların hiçbiri checkpoint payload'una girmez veya source-order
öğrenme sırasını değiştirmez:

- `hv_bitacc_t`, bilinen maksimum bundle cardinality için yalnız gerekli binary
  carry plane'lerini resetler; daha yüksek bir plane ilk kez gerekli olduğunda
  güvenli biçimde lazy-clear edilir. Örneğin window-4 positional context dört,
  local hybrid beş plane ile tamamlanır; fiziksel üst sınır 2.047 item / 11
  plane olarak kalır. Odd cardinality majority'de tie oluşmayacağı için
  tie-break vector de üretilmez.
- Repeated center'ların `experience_before` planlaması artık aynı record'un
  önceki bütün center'larını tekrar taramaz. Capacity başına bir `int32`
  occurrence scratch tablosu ve en çok 1.024 touched-slot listesi ile
  source-order sayaçları O(update) hesaplanır; scratch yalnız touched slotlar
  için temizlenir. Active-token prefix sum ile bütün bounded-window neighbor
  cardinality'leri de O(token) hesaplanır; context bundle'ın kendisi doğal
  olarak her gerçek neighbor katkısını yine işler.
- 256 × 1.024 `int64` context staging matrisi her record öncesi sıfırlanmaz.
  Yalnız `update_required` kolonları yazılır ve daha sonra yalnız onlar okunur;
  böylece kısa record'larda gereksiz yaklaşık 2 MiB memory-bandwidth write
  kaldırılır.
- `hierarchical` whole-record accumulator yalnız ≥12 active token olduğunda
  kurulur. `coherent` aynı eşikte bir full-record ve en çok iki parity-fold
  accumulator'ını record başına bir kez kurar; `debiased` ise aynı touched-slot
  scratch ile distinct active slots'u bir kez toplar ve ancak ≥12 distinct slot
  varsa bir accumulator kurar. Hiçbiri center başına yeniden tarama yapmaz.
  Kısa/suppressed record'lar bit-tam local `hybrid` semantiğine sahip olduğu
  için bu modlarda global allocator/reset/bundle işi yapılmaz.
- 4.096-entry exact-token lookup cache, aynı fixed-width token'ın tekrar eden
  active/OOV probe'larını canonical hash-table walk olmadan karşılar. Full token
  equality veya fingerprint uyuşmazlığı miss'tir; cache state epoch sonunda
  bırakılır. `--no-lookup-cache` reference execution yolu için aynı bounded
  workspace ayrılır ama lookup shortcut'ı çalışmaz.
- Adaptive scheduler, varsayılan olarak bir record'da en az 32 eligible center
  varsa OpenMP worksharing seçer; daha küçük record'lar serial immutable path'e
  gider. CLI çıktıdaki `CPU serial/parallel context batches` ve `...contexts`
  sayaçları seçilen execution path'i raporlar. Bu telemetry throughput veya
  hardware utilisation iddiası değildir.

Bit-sliced bounded reset, planner scratch, fixed 4.096-entry lookup-cache
workspace ve staging allocation sabit capacity ile sınırlıdır. Resource preflight
planner scratch'ı (`4 × --max-vocab` byte) ve yaklaşık 272 KiB conservative
lookup-cache workspace'ini ayrıca RAM estimate'ine dahil eder. En önemlisi, context'ler immutable
`B(w)` üzerinden hazırlanır fakat `S(w)` ve experience sayaçları sonra aynı
orijinal token sırası ile commit edilir; serial/parallel veya threshold seçimi
checkpoint byte'larını değiştirmemelidir.

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
  --context-mode hierarchical --epochs 3 --initial-plasticity 25.0 --minimum-plasticity 1.0 \
  --decay-events 32 --query token

# Long-record global self-echo'yu cross-fold olarak azaltan CPU-only encoder.
# Kısa record'larda hybrid ile bit-tam aynıdır.
./build/vegai_pretrain train LONG_RECORDS.txt coherent.vgai \
  --max-vocab 50000 --min-frequency 3 --window 4 --position-shift 73 \
  --context-mode coherent --epochs 3 --strict

# Tekrarlı boilerplate'in global majority'yi occurrence ile domine etmesini
# önleyen CPU/OpenMP-only encoder; <12 distinct active slotta hybrid'e düşer.
./build/vegai_pretrain train REPEATED_LONG_RECORDS.txt debiased.vgai \
  --max-vocab 50000 --min-frequency 3 --window 4 --position-shift 73 \
  --context-mode debiased --epochs 3 --strict

# Epochlar arasında learned semantic neighbourhood feedback'i kullanan
# CPU/OpenMP-only encoder. Resource preflight'ta 2,048 byte × max-vocab
# snapshot reservation'ını ayrıca bildirin.
./build/vegai_pretrain train CORPUS.txt resonant.vgai \
  --max-vocab 50000 --min-frequency 3 --window 4 --position-shift 73 \
  --context-mode resonant --epochs 3 --strict
./build/vegai_control resource-plan CORPUS.txt --max-vocab 50000 \
  --context-mode resonant --available-memory-bytes 17179869184 \
  --available-storage-bytes 2147483648 --require-ready

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
| `--context-mode MODE` | `hierarchical` | `legacy`, bit-tam hızlı `positional`, local positional+topical `hybrid`, ≥12 active-token kayıtta `full-record × 3` global evidence ekleyen `hierarchical`, `full-record × 1 + opposite-parity fold × 2` kullanan CPU-only `coherent`, ≥12 **distinct** active slotta unique-slot `full-record × 3` kullanan CPU/OpenMP-only `debiased`, veya epoch-başı frozen `Ŝ` topical feedback kullanan CPU/OpenMP-only `resonant` |
| `--lookup-cache` / `--no-lookup-cache` | açık / kapalı | execution-only 4.096-entry fingerprint + full padded-token direct-map cache'i aç/kapat; collision miss'tir, checkpoint state değildir |
| `--parallel-context-build` | açık | immutable per-record context construction için OpenMP; yalnız record update sayısı eşikteyse team açılır, source-ordered prototype commit sayesinde thread count checkpoint semantiğini değiştirmez |
| `--parallel-min-contexts N` | 32 | execution-only adaptive scheduler eşiği; N'den az eligible center içeren kısa record'lar team-launch/staging maliyetinden kaçınmak için serial hazırlanır |
| `--serial-context-build` | kapalı | yalnız debug/regression için parallel context hazırlığını kapatır; checkpoint state değildir |
| `--epochs N` | 1 | ikinci corpus geçişi sayısı |
| `--initial-plasticity PCT` | 25.0 | başlangıç selected-dimension yüzdesi; tam olarak bir ondalık basamak desteklenir |
| `--minimum-plasticity PCT` | 1.0 | schedule alt sınırı; ör. `0.6` tam fixed-point olarak saklanır |
| `--decay-events N` | 32 | token başına ilk schedule eşiği |
| `--strict` | kapalı | drop/truncation durumunu hata yap |
| `--reference-context-kernel` | kapalı | yalnız benchmark/regression için önceki rotate-then-add kernelini kullanır; varsayılan fused implementation ile checkpoint byte'ları bit-tam eşit olmalıdır |
| `--query TOKEN` | yok | eğitimden sonra veya `inspect` sırasında bir semantic neighbour göster |
| `inspect --query-file PATH` | yok | her nonblank satırdaki tek token için tek checkpoint load ile batch nearest-neighbour diagnostic |

`resume`, checkpoint'ten vocabulary threshold/capacity, `window`, shift,
context encoder, plasticity ve tie seed'ini geri alır; yalnız `--epochs`,
`--strict`, `--query` ve execution-only `--reference-context-kernel`,
`--lookup-cache` / `--no-lookup-cache`, `--parallel-context-build` /
`--serial-context-build` / `--parallel-min-contexts` seçeneklerini kabul eder
ve vocabulary'yi değiştirmez.
Bu implementation/execution switches checkpoint
semantiği veya payload değildir; reference switch `continue` için de yalnız
benchmark/regression amacıyla kullanılabilir. `continue` aynı
uyumlu config'i geri alır, önce yeni shard'ın token'larını vocabulary'ye ekler,
yalnız newly-active slotları başlatır, ardından shard'ı train eder. Öğrenme
semantiğini checkpoint'ten farklı bir config ile sürdürme
`PT_ERR_INCOMPATIBLE_CONFIG` ile reddedilir. CLI query'si tokenizer üzerinden
canonicalize edilir ve yalnız bir token kabul eder.

### 4.1 Tek bir doğrudan train için manifest-bound attestation

Çıplak `vegai_pretrain train` komutu checkpoint formatını ve deterministic
öğrenme çekirdeğini denetler; ancak tek başına hangi corpus/manifest/binary ile
çalıştığını, process sırasında input'un değişmediğini veya yarım checkpoint'in
publication'a ulaşmadığını kaydetmez. Yeni bir **tek corpus** direct-run için
`tools/attested_pretrain.py` bu operasyonel sınırı fail-closed olarak ekler.
Bu araç `resume`/`continue` transaction controller'ı değildir: immutable shard
ve exactly-once generation için hâlâ
[CONTINUOUS_PRETRAINING.md](CONTINUOUS_PRETRAINING.md), packaged production
release için [PRODUCTION_PRETRAINING.md](PRODUCTION_PRETRAINING.md) kullanılmalıdır.

Önce operator corpus'un kaynağını, lisansını, retrieval zamanını ve canonical
preprocessing'ini gözden geçirir; araç bu insan incelemesini otomatik olarak
kanıtlayamaz. Ardından aynı strict LF/ASCII ingest contract'i ile manifest
oluşturulur ve yeni checkpoint/report hedefleri seçilir:

```bash
python3 tools/corpus_manifest.py DOMAIN_CORPUS.txt \
  --output runs/domain.manifest.json \
  --source 'operator-recorded dataset identifier or URI' \
  --license 'operator-verified usage terms' \
  --retrieved-at '2026-10-24T00:00:00Z' \
  --preprocessing 'documented ASCII-safe preprocessing/v1' \
  --strict-contract

python3 tools/attested_pretrain.py train \
  --pretrain-bin ./build/vegai_pretrain \
  --corpus DOMAIN_CORPUS.txt --manifest runs/domain.manifest.json \
  --checkpoint checkpoints/domain.vgai --report runs/domain.attestation.json \
  --max-vocab 50000 --min-frequency 3 --window 4 --position-shift 73 \
  --context-mode hierarchical --epochs 3 --omp-threads 8

# Retraining yapmadan corpus, manifest, binary ve published checkpoint'i yeniden bağlar.
python3 tools/attested_pretrain.py verify --report runs/domain.attestation.json
```

`train` yalnız `vegai-corpus-manifest/v1` kabul eder, manifest telemetry'sini
corpus'u stream ederek train öncesi ve sonrasında yeniden hesaplar ve native
child'a zorunlu `--strict` geçirir. Manifest/binary/corpus değişirse, strict
telemetry kaybı görülürse, output formatı/metric/inspect configuration binding'i
bozulursa veya resource plan hazır değilse candidate yayınlanmaz. Native train
çıktısındaki `active_tokens + oov_tokens = raw_tokens` partition'ı ve raw-token/
record count'ları manifest'e bağlanır. `--min-frequency` bilinçli olarak
inactive token üretebilir; dolayısıyla generic wrapper **OOV=0 vaat etmez**.
OOV miktarı report'ta görünür ve operator kalite/vocabulary policy'sinin ayrı
bir kararıdır. `--context-provider openvino` seçilirse wrapper worker script'i
ve PATH'ten çözümlenmiş Python interpreter binary'sini de pre/post SHA-256 ile
bağlar; report bunları `inputs.openvino_worker` ve `inputs.openvino_python`
altında taşır. Bu, Python package seti, OpenVINO/runtime driver, firmware veya
device state için tam environment snapshot değildir; accelerator release'i için
container/image digest'i ya da operator-owned immutable environment manifest'i
ayrıca saklanmalıdır.

Varsayılan RAM/storage preflight zorunludur; yalnız izole contract testleri için
`--skip-resource-preflight` report'a açık bypass olarak yazılır. Stopwatch tam
olarak bir native `vegai_pretrain train` child process'i çevreler; manifest
scan, preflight, binary hash, inspect, publication, compile ve CI orchestration
süreleri `measurement.excluded` içindedir. Report semantic-quality, dünya
bilgisi, güvenlik veya genel yetenek kanıtı değildir.

Checkpoint önce **aynı checkpoint directory/filesystem** içinde private
`.partial` candidate'a yazılır, inspect edilir, sonra `os.replace` ile final
checkpoint'e alınır. JSON report kendi directory'sinde staged edilir ve
checkpoint'ten **sonra** publish edilir; böylece yeni bir report normal akışta
mevcut bir checkpoint olmadan görünmez. Checkpoint **ve** report final path'i
ayrı ayrı conservative `O_EXCL` lock alır; böylece yalnız bir path'i yeniden
kullanan iki invocation da interleave edemez. Crash sonrası lock kalırsa owner'ı
inceleyin; yalnız writer'ın çalışmadığından eminseniz yeni publish için
`train --recover-lock`, yalnız mevcut pair'i okumak için `verify --recover-lock`
kullanın. Araç iki ayrı dosyayı tek filesystem transaction'ında atomik değiştiremez: özellikle
`--overwrite` sırasında checkpoint replace ile report replace arasındaki host
crash'i eski report/yeni checkpoint mismatch'i bırakabilir. Bu durumda
`verify` fail-closed reddeder; report'u veya checkpoint'i elle "başarılı" kabul
etmek yerine tekrar train/publish edin. Başarısız train, validation veya inspect
eski published çifti değiştirmez ve private candidate'lar temizlenir.

`verify` report schema/configuration/metric arithmetic/staged-command contract'ini
kontrol eder; aktif veya stale output lock varken olası in-progress publication'ı
okumayı reddeder, güncel manifest, corpus, binary ve checkpoint SHA-256
identity'lerini tekrar hesaplar ve gerçek `inspect` alanlarını recorded inspection
ile eşitler. SHA-256 accidental corruption/substitution tespitidir, saldırgana karşı imzalı
attestation değildir; araç pre/post scan yapar ama hostile bir filesystem writer'a
karşı immutable snapshot/isolation primitive'i değildir. Corpus, manifest ve
binary'yi run boyunca erişim kontrollü/read-only storage'da tutun; report/manifest/checkpoint
digest'ini güvenilir dış release registry veya imza mekanizmasında saklayın. Bu doğrulama yeniden train
etmez ve o anki RAM/CPU availability'sini yeniden quality/release kararı olarak
ölçmez.

Crash-safe append-only shard ledger, SHA-256, strict LF/ASCII contract ve
exactly-once durable generation semantics için
[CONTINUOUS_PRETRAINING.md](CONTINUOUS_PRETRAINING.md)'yi kullanın. Tek başına
`continue`, checkpoint'i verilen path üzerinde güncellediği için controller'ın
private candidate + journal transaction garantisinin yerini tutmaz.

Küçük smoke corpus ve dengeli Türkçe-ASCII fixture için
[`examples/`](../examples/README.md) altındaki örneğe bakın. İkinci fixture,
manifest digest'i, checksum-bit corruption rejection ve gerçek `train → inspect → resume`
CLI testiyle doğrulanır; çalıştırılan counter, schedule ve checkpoint ayrıntıları
[EXAMPLE_PRETRAINING_RUN.md](EXAMPLE_PRETRAINING_RUN.md)'dedir. Hiçbiri
semantic kalite benchmarkı değildir. Gerçek target-domain checkpoint quality
kararı için repository fixture'ı yerine operator-owned held-out
semantic-neighbour case manifest'i ve immutable policy ile
`tools/pretraining_quality_evaluation.py` kullanın. Araç mevcut batch
`inspect --query-file` ABI'sini strict parse eder, real checkpoint/binary/case/
policy SHA'larını bağlar ve HMAC-redacted report ile threshold gate uygular;
kapsamı yalnız bu retrieval ölçüsüdür. Tam schema/production integration ve
non-claims: [PRETRAINING_QUALITY_GATE.md](PRETRAINING_QUALITY_GATE.md).

## 5. Kaynak-duyarlı çalışma ve çok çekirdek

Corpus streaming olduğundan RAM gereksinimi corpus file boyutundan çok
`--max-vocab` ve HDC model capacity'sine bağlıdır. Eğitimden önce aynı corpus,
checkpoint path ve capacity ile aşağıdaki preflight'i çalıştırın:

```bash
python3 tools/training_resource_plan.py --corpus CORPUS.txt \
  --checkpoint checkpoints/model.vgai --max-vocab 50000 --require-ready
```

Araç RAM/storage lower-bound estimate'i, CPU topology, NUMA/RAM-channel
best-effort telemetry ve OpenMP placement önerisi üretir; ayrıca capacity başına
4-byte source-order/de-biased planner scratch'ını ve sabit exact-token lookup-
cache workspace'ini de dahil eder. SSD/HDD ve dual-RAM channel seçimi
OS/firmware tarafından yönetilir; Vegai bunları zorlamaz.
Deterministic OpenMP context construction, immutable base vectors üzerinde
paralel çalışır; adaptive scheduler küçük record'larda serial kalabilir, model
update'leri ise her durumda kaynak sırası ile commit edilir. Bu yüzden
`OMP_NUM_THREADS=1` ve çok-thread run checkpoint byte'ları eşit olmalıdır.

```bash
export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores
python3 tools/compare_pretraining_threads.py --pretrain-bin ./build/vegai_pretrain \
  --parallel-threads 2 --repetitions 3
```

Bu benchmark'ta compile/fixture/harness süresi hariçtir ve hız oranı yalnız
ölçülen host/workload için geçerlidir. Exact vocabulary-cache yolu için ayrıca:

```bash
python3 tools/compare_pretraining_lookup_cache.py \
  --pretrain-bin ./build/vegai_pretrain --repetitions 3
```

Bu comparator cache açık/kapalı gerçek `train` child process wall-clock
örneklerini ayrı ölçer; checkpoint SHA/byte eşitliği ve full-token hit/miss
telemetrisi zorunludur. Collision-heavy fixed-width token-set fixture yalnız lookup hot-path
kanıtıdır, doğal corpus veya genel eğitim hız iddiası değildir. Sabit `e34fd22` baseline ile aynı-run
mimari önce/sonra ölçümü için `tools/compare_pretraining_revisions.py` kullanın;
araç real `train` process'lerini timing eder, warm-up/order alternation uygular
ve varsayılan olarak checkpoint byte equality zorunlu kılar. Yalnız açıkça
incelenmiş checkpoint-algoritma migration'ında
`--checkpoint-contract versioned-algorithm-transition` kullanılabilir; bu modda
her binary'nin self-determinism'i zorunlu kalır ve candidate gerçek baseline
checkpoint'ini byte-tam resume etmek zorundadır, fakat cross-lineage runtime
oranı semantic-equivalence speedup iddiası değildir. Controlled sonuçlar ve
fixture sınırları [PRETRAINING_MODE_COMPARISON.md](PRETRAINING_MODE_COMPARISON.md)'de
aynı-run baseline bölümünde belgelenir. GPU/NPU adapter inventory'si tek başına compute kullanımı
değildir. Intel Core Ultra için opt-in OpenVINO context
provider, immutable packed-HDC bundle/finalize aşamasını dispatch ederken
source-order update'i CPU'da tutar; worker bir train çağrısı boyunca resident
kalır ve item-cardinality graph'larını dynamic batch ile cache'ler. `coherent`
mode'un center-dependent cross-fold item dizisi, `debiased` mode'un distinct-
slot global item dizisi ve `resonant` mode'un epoch-frozen semantic snapshot
item dizisi bu worker ABI'sinde yoktur: `--context-provider openvino` ile bu
üç mode validation aşamasında reddedilir; sessiz CPU/OpenVINO route değişimi
veya mutable snapshot substitution yapılmaz. Explicit CPU fallback,
CPU plugin'i hızlandırma diye sunmak yerine native fused/OpenMP kernel'ine bypass eder. Device mask'i runtime output'ta doğrulanır ve strict
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
| `vocabulary lookup cache hits/misses` | checkpoint dışı execution telemetry; cache açıkken raw lookup partition'ı, kapalıyken ikisi de sıfır |
| `de-biased global-topic/suppressed records` | yalnız `debiased` modda ≥12 distinct slotla global summary alan veya repetition nedeniyle local-hybrid'e düşen long record sayısı |
| `de-biased distinct topic tokens` | global-channel eligibility kontrolü yapılan long record'lardaki distinct active slot toplamı |
| `resonant frozen-snapshot contexts` | yalnız `resonant` modda epoch-başında frozen `Ŝ` topical channel'ını gerçekten tüketen prepared center context sayısı |
| `resonant epoch snapshot bytes` | yalnız `resonant` modda geçici snapshot'ın gerçek capacity için byte reservation'ı; checkpoint payload'una dahil değildir |

`active_tokens + oov_tokens = raw_tokens` ikinci geçiş için beklenen temel
invariant'tır. `center_updates` active token sayısından küçük olabilir; örneğin
satırda tek active token varsa context oluşmaz. Metrikler corpus kalitesini veya
semantic accuracy'yi tek başına ölçmez; ingest ve öğrenme kapsamını kanıtlayan
telemetridir.

## 7. Checkpoint ve resume sözleşmesi

`pretrain_checkpoint_save` stream-unformatted bir `.vgai` dosyası yazar;
`pretrain_checkpoint_load` magic/version, shape, counter sign, vocabulary
frequency/active invariantları, rebuilt hash index ve payload checksum kontrolü
yapar. Sürüm, sadece wire parser değil **gelecek Hebbian seçim algoritmasının
continuation contract'i**dir:

- Modern Floyd/complement sampler kullanan yeni context mode checkpoint'leri
  **v9** header/checksum'ıyla yazılır. v9, plasticity schedule değerlerini
  yüzde puanının onda biri cinsinden saklar; böylece `0.6%` float yuvarlaması
  olmadan ifade edilir. Exact-rate update, HV dimension sayısını sabit nokta
  aritmetiğiyle en yakın tam bite yuvarlar.
- v4-v8 checkpoint'leri okunabilir: eski whole-percent alanları bellekte
  tenths-of-a-percent ölçeğine taşınır; checksum eski biçimde doğrulanır. v7
  unbiased Floyd/complement sampler lineage'ini, v8 ise RESONANT epoch-başı
  frozen semantic-context sözleşmesini taşır; yeni v9 bunların ikisini de
  sürdürürken sadece plasticity alanlarının birimini günceller.
- Reader v4, v5 ve v6'yı historical bounded-retry sampler lineage'i olarak
  yükler. Bu eski sampler'lı checkpoint'ler tekrar kaydedildiğinde non-coherent
  ise v5, coherent ise v6 biçimini ve whole-percent alanlarını korur; böylece
  geçmiş deterministic continuation stream'i değişmez.
- Coherent payload v4 veya v5 ise fail-closed reddedilir; v6+ coherent için
  geçerlidir. `debiased` payload yalnız v7+ Floyd lineage'inde, `resonant`
  payload yalnız v8+ içinde kabul edilir. Legacy sampler'a ait olmayan
  fractional plasticity değeri eski checkpoint'e yazılmaya çalışılırsa kayıplı
  dönüşüm yerine fail-closed reddedilir.

`vegai_pretrain inspect`, `plasticity sampler` alanında `floyd-complement` veya
`legacy-bounded-retry` yazar; operatör checkpoint lineage'ini context mode'dan
ayrı görmelidir. Böylece eski checkpoint resume'i sessizce farklı bir mask
sequence'ine geçmez;
aynı eski lineage'de uninterrupted/resumed run bit-tam kalır. v3 ve daha eski
checkpoint'ler reddedilir ve corpus manifesti + source commit ile yeniden train
edilmelidir. `--reference-context-kernel` bilerek checkpoint state olmadığından
fused/reference run'ları **aynı sampler lineage'inde** byte-tam aynı checkpoint
üretmelidir. Farklı sampler lineage veya encoder mode checkpoint'lerinin
byte'larının eşit olması beklenmez. Checksum yalnız accidental corruption
tespiti içindir; kriptografik bütünlük/imza değildir.

Checkpoint aşağıdakileri taşır:

- capacity, token strings, frequencies, active mask ve min frequency;
- context mode/position/plasticity/tie-seed ve version-selected sampler lineage semantiği;
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

Rich JSONL/JSON/Parquet/Arrow kaynakları için önce
`tools/training_dataset.py` ile `vegai-training-record/v1` schema/quality audit'i
çalıştırın; exact text dedupe, label imbalance ve deterministic train/validation/test
hash split'i bu aşamada görünür olur. Native eğitim doğrudan bu rich alanları
okumaz: yalnız explicit strict text export veya `production_dataset.py
--input-format ...` adapter'ı `record.text` alanını ASCII/LF corpus'a projection
yapar. Tam schema ve format-I/O sözleşmesi
[TRAINING_DATA_FORMAT.md](TRAINING_DATA_FORMAT.md)'dedir.

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

Strict ASCII corpus'u content-addressed immutable shard'lara bölmek, disk-backed
exact record dedupe uygulamak, deterministic held-out split yayınlamak ve run
öncesi exact record-level train/validation leakage audit yapmak için
`tools/production_dataset.py` + `tools/production_pretrain.py` kullanılabilir.
Bu production HDC control plane'in atomic publish/exactly-once/provenance
sözleşmesi [PRODUCTION_PRETRAINING.md](PRODUCTION_PRETRAINING.md)'de tanımlıdır.
Bu araçlar LLM/LMM runtime, semantic near-duplicate detection veya lisans/PII
uyumluluğunun yerine geçmez.

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
  plane ve sample count için bit eşitliği; mixed base-position/semantic-topic
  pair kernelinin de iki explicit add ile eşitliği ve pair overflow'unda atomic
  state;
- fused `hybrid` corpus run'ının rotate-then-add reference kernel ile semantic
  plane, base plane ve learning schedule bakımından bit-tam eşitliği;
- repeated center token'lı record üzerinde serial ve OpenMP context construction
  run'larının base/semantic plane ve experience counter bakımından bit-tam eşitliği;
- ≥12 active-token controlled record üzerinde `hierarchical` replay
  determinismi, local-only hybrid'den ayrışması ve checkpoint round-trip'i;
- aynı long-record fixture üzerinde `coherent`in hierarchy'den ayrışması,
  fused/reference local-kernel eşitliği, kısa-record `hybrid` eşitliği ve v9
  checkpoint round-trip'i; synthetic v5 ve coherent-v6 lineage'lerinin
  historical sampler ile uninterrupted run'a bit-tam resume edilmesi;
- duplicate olmayan long record'da `debiased = hierarchical` bit eşitliği;
  raw uzun ama <12 distinct slotlu repeat record'da `debiased = hybrid` bit
  eşitliği, distinct/suppressed telemetry, v9 checkpoint round-trip'i ve
  legacy sampler save rejection; ayrıca OpenVINO immutable ABI'sinin explicit
  `debiased` rejection'ı;
- `resonant` ilk epoch'unun local `hybrid`le base/semantic/experience düzleminde
  bit-tam eşitliği, ikinci epoch'ta learned snapshot ile hybrid'den semantic
  olarak ayrışması, fused/reference eşitliği, v9 checkpoint round-trip +
  uninterrupted/resumed third epoch byte eşitliği, snapshot telemetry/reservation
  ve OpenVINO ABI'sinin explicit rejection'ı;
- cache açık/kapalı run'larının base/semantic/experience planes ve checkpoint
  continuation bakımından bit-tam eşitliği; hit/miss partition telemetry ve
  fingerprint'e rağmen full padded-token equality;
- windowed context count ve active/OOV telemetry;
- checkpoint round-trip ve uninterrupted run ile checkpoint-resume run'ın
  bit-tam eşitliği;
- incompatible resume config, strict long token ve strict overlong record hata
  yolları;
- `vegai_attested_pretraining_real_binary` CTest'i: gerçek compiled binary ile
  generic manifest-bound strict train, same-filesystem candidate inspect/publish,
  report verify, corpus/manifest/checkpoint/stored-inspection tamper rejection ve
  failed `--overwrite` altında önceki published pair'in byte korunumu; ayrıca
  hermetic fake CLI ile OpenVINO worker/interpreter identity branch'i (bu ikinci
  subcase gerçek device/runtime dispatch iddiası değildir).

CI ayrıca küçük corpus ile `train → inspect → resume` CLI smoke testini ve
optimized Make binary üzerinde `tools/compare_pretraining_modes.py` çalıştırır.
Araç, yalnız gerçek `vegai_pretrain train` child-process wall-clock'ını ölçer:
compiler/CI setup, fixture construction, inspect ve Python harness süreleri
hariçtir. Base 64-token fixture üzerinde `legacy → positional → fused hybrid →
resonant` oranlarını ve resonant'ın hybrid'e göre controlled Top-1 similarity
farkını; aynı hybrid config için reference rotate-then-add ile fused kernel
checkpoint byte eşitliği + süre oranını; ayrıca 16-token-record kontrollü fixture
üzerinde `hybrid → hierarchical → coherent` Top-1 cohesion diagnostic'ini
raporlar. Coherent arm için kabul koşulu hybrid same-cluster Top-1 sonucunu
geriletmemektir; resonant arm için controlled mean Top-1 similarity farkı
`> 0.02` fail-closed şarttır. Bu fixture-bound semantic-retrieval evidence'ı,
pozitif farkın her doğal-dil corpus'una genellenmesi değildir. Bunlar kontrollü
tanılardır; doğal dil yeteneği veya production throughput garantisi değildir. CI ayrıca `tools/compare_pretraining_lookup_cache.py`
ile cache açık/kapalı, checkpoint-byte-identical real child-process comparator'ı
çalıştırır. Bu fixture canonical vocabulary hash start indexi aynı, direct-cache
indexleri farklı fixed-width token setiyle lookup probe overhead'ini izole eder;
hız oranı host/workload gözlemidir ve minimum speedup acceptance gate'i değildir.

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
