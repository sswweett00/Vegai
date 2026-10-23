# Vegai HDS-Fortran mimari sözleşmesi

Bu belge, Fortran 2023 hedefli kaynak kodun koruması gereken semantik ve bellek
invariantlarını tanımlar. API yorumlarıyla çelişki halinde kaynak kod + native testler
referanstır; performans iddiaları ise hedef makinede ölçülmelidir.

## 1. Temsil katmanı

### 1.1 Boyut ve bit/bipolar eşlemesi

```text
D          = 16,384 bit
HV_WORDS   = 256
WORD_BITS  = 64
HV_BYTES   = 2,048 byte
```

`hv_t%bits(1:256)` bir bit dizisini taşır. Fiziksel bit `0`, bipolar `−1`;
fiziksel bit `1`, bipolar `+1` olarak yorumlanır. Bu bir temsil eşlemesidir:
`F₂^D` ile `{-1,+1}^D` aynı cebirsel nesne değildir, fakat aşağıdaki binding
işlemi altında isomorfik davranır:

```text
binary:  a XOR b
bipolar: sign(a) × sign(b)
```

Bir `hv_t` payload'u 2 KiB'dir. Derived type padding'i bir ABI detayı
olabileceğinden API, `storage_size(hv_t)` için ayrıca söz vermez; veri düzlemi
her zaman 256 `int64` kelimesidir.

### 1.2 Rastgelelik ve reproducibility

`hv_from_seed(seed)` bir seed için bit-tam deterministiktir. Uygulama
SplitMix64'in standart dönüşümlerini kullanır. Önemli ayrıntı, normal
Fortran'daki signed overflow'a yaslanmamasıdır:

- `add_mod64`: taşıma yayılımını `AND/XOR/ISHFT` ile yaparak 2⁶⁴ modunda toplar;
- `mul_mod64`: dört 16-bit limb ile 2⁶⁴ modunda çarpar;
- sabitler BOZ literal olarak yüklenir, böylece bit desenleri korunur.

Bu sayede `-ftrapv`, farklı optimizasyon düzeyleri veya derleyici signed
overflow yorumları PRNG dizisini değiştirmemelidir. `tools/verify_reference.py`
standard SplitMix64'ün seed-0 ilk değerini (`E220A8397B1DCDAF`) oracle olarak
kontrol eder.

`hv_seed_from_id` bir `int32` ID'yi `int64` seed'e injective olarak taşır.
Bellek hash tablosu da negatif ID'leri geçerli kabul eder.

### 1.3 Tokenless raw-byte Signal Field

`tokenless_field` normal `text_tokenizer` ve external LMM tokenizer plane'inden
ayrı bir raw-octet (`int8`) giriş yoludur. BPE/subword vocabulary, merge,
normalization veya learned lookup çalıştırmaz. Bir byte stream'in her event'i
sabit domain-separated atomla temsil edilir; `order_field` pozisyonu XOR
kompozisyonuna dahil eder, `distribution_field` ise deterministic majority ile
sıra-duyarsız dağılım özetidir. Bu iki field de sabit `D=16384` bitlik sketch'tir;
arbitrary uzun stream için injective/lossless değildir. Raw source digest ve
provenance field dışında tutulmalıdır.

Modül yalnız `1…32767` raw byte kabul eder. Limit, `int16` bundling counter'ın
wraparound'sız `±32767` unit vote sınırından türetilir; empty veya oversized
input fail-closed hata döndürür. `D=2^14` sayesinde unnormalised
Walsh-Hadamard dönüşümü tam `int64` sınırlarıyla kontrol edilir:
`HᵀH = D I`, valid bipolar field için `||Hx||² = D²` ve inverse
`H(Hx)/D = x` bit-tam sınanır. Bu, semantic veya fiziksel dünya doğruluğu
iddiası değil, sonlu representation aritmetiği invariantıdır. Tam contract,
integer bounds, independent oracle ve deployment sınırları için
[TOKENLESS_SIGNAL_FIELD.md](TOKENLESS_SIGNAL_FIELD.md)'ye bakın.

## 2. Cebirsel operator invariantları

### 2.1 Binding / unbinding

```fortran
c = hv_bind(a, b)
```

Her kelimede `c = ieor(a,b)` yapılır. Özellikler:

```text
A ⊗ 0 = A
A ⊗ A = 0
A ⊗ (A ⊗ B) = B
A ⊗ B = B ⊗ A
```

Buradaki `0`, binary all-zero vektördür (bipolar all-`−1`), lineer cebirdeki
sıfır vektörü değildir. Bu ayrım özellikle açıklama ve dokümantasyonda
önemlidir.

### 2.2 Bundling

`hv_bundle3(a,b,c)` şu boolean majority ifadesidir:

```text
(a AND b) OR (b AND c) OR (a AND c)
```

Genel veya weighted bundling `hv_acc_t` kullanır. `counts(bit, word)` o
boyuttaki bipolar oy toplamıdır. `hv_acc_add` iki fazlıdır:

1. bütün 16.384 counter için güncellemenin `int16` sınırına sığıp sığmadığını
   kontrol eder;
2. yalnızca kontrol başarıyla bittiyse counter'ları ve `sample_count` değerini
   mutasyona uğratır.

Dolayısıyla taşma başarısızlığı **atomiktir**. `hv_acc_finalize`, pozitif
sayacı bit-1, negatif sayacı bit-0 yapar; sıfırda deterministik `tie_seed`
hiper-vektörünün aynı bitini kullanır. Boş accumulator bilerek all-zero
döndürür; rastgele bir “boş bundle” üretmez.

Unit-weight corpus context'leri için `hv_bitacc_t`, her 64-bit word içindeki
64 ayrı population counter'ı 11 binary carry plane'de tutar. Böylece 16.384
ayrı bit branch'i yerine word-level XOR/AND ripple carry uygulanır. Eşit
kardinalitede aynı seeded tie policy kullanıldığından `positional` pretraining
mode'u `hv_acc_t` legacy yoluyla bit-tam eşdeğerdir; native ve bağımsız Python
reference testleri bu eşdeğerliği sınar.

### 2.3 Permütasyon

`hv_permute(v,k)` 16.384 bitlik tek bir halkadır, 256 bağımsız word halkası
değildir. Pozitif `k` için:

```text
out[(p + k) mod D] = in[p]
```

Negatif shift önce `modulo(k,D)` ile normalize edilir. Bu, Fortran'ın negatif
sayılar için `mod` ve `modulo` farkı nedeniyle kritiktir. Sınanan invariant:

```text
Π^(-k)(Π^k(v)) = v
Π^(D+k)(v) = Π^k(v)
```

### 2.4 Hamming similarity

```text
H(A,B)   = Σ popcnt(A_word XOR B_word)
sim(A,B) = 1 − 2H(A,B)/D
```

`H` en fazla 16.384 olduğundan `int32` güvenlidir. `hv_hamming_zscore` bağımsız
rastgele vektör hipotezinde `H ~ Binomial(D, 1/2)` için `(H−8192)/64` döndürür.
Bu bir diagnostic'tir, dağılımsal garanti değildir.

### 2.5 Exact noise test araçları

`hv_inject_noise_exact(v,n,seed,...)` PRNG ile benzersiz bit konumları seçer,
bir bitmask kurar ve XOR uygular. Sonuç için kesin olarak `H(v,noisy)=n`dir.
Rastgele çakışmalar çok uzarsa deterministik lineer fallback ile terminasyon
garantilidir. Yüzde wrapper'ı integral yüzdeyi en yakın bit sayısına çevirir;
`25% × 16384 = 4096` tamdır.

## 3. Bellek düzeni ve lifecycle

### 3.1 Sabit kapasite

`mem_init(memory, max_symbols, status)` aşağıdaki alanları bir kez allocate
der:

```text
vector_bits(HV_WORDS, capacity)   int64  packed symbol columns
ids(capacity)                     int32
labels(capacity)                  char(len=48)
experience_count(capacity)        int32
hash_used/hash_keys/hash_slots    open-addressed ID index
lsh_table(8,4096), lsh_counts     fixed candidate index
```

Fortran column-major düzeninde `vector_bits(:,slot)` tek bir hiper-vektörün
256 bitişik kelimesidir. Aynı word ofsetinin tüm sembollerdeki değerleri de
bitişiktir. Bu, array-of-derived-types kullanmaktan daha açık ve gerçek bir
SoA-benzeri data plane'dir. Kayıttan sonra normal işlem yolunda allocate veya
deallocate yoktur.

`mem_footprint_bytes` derleyicinin bildirdiği `storage_size` ile alanların
payload toplamını hesaplar. Allocator metadata, stack, instruction cache ve
OpenMP runtime bu ölçüye dahil değildir.

### 3.2 ID index

ID hash'i bit-mixing ve linear probing kullanır. `hash_used` ayrı dizi olduğu
ıçin `-1` gibi “empty sentinel” gerekmez. Register iki kez aynı ID ile
çağrılırsa `MEM_ERR_DUPLICATE_ID` verir; önceki LSH entry'si bozulmaz.

### 3.3 LSH ve cleanup ayrımı

12 seçili bit `1..4096` arası bir bucket verir. Her bucket sadece ilk sekiz
slot tutar. Bu tasarım LSH'nin *candidate filter* olduğunu kabul eder:

- `mem_cleanup`: global en yakın komşu; LSH yalnız distance-0 güvenli kısa yol
  için kullanılır. Paralel minimum bulur, sonra seri ikinci geçişte en küçük
  slot tie-break uygular.
- `mem_cleanup_lsh`: sadece bucket adayları; `exact_search=.false.` döner.
  Gerçek nearest neighbour garantisi yoktur.

Bu ayrım, sık görülen “LSH candidate eşittir gerçek en yakın” hatasını API
seviyesinde görünür kılar.

### 3.4 Plasticity

`mem_associate_hebbian` hedef prototipin yaklaşık %25 maskelenmiş bitini yeni
context ile değiştirir. Güncellemeden sonra `mem_rebuild_lsh` çağrılır; aksi
takdirde approximate sorgu yeni vektörün eski bucket'ına bakabilirdi.

## 4. Akıl yürütme katmanı

### 4.1 Role–filler frame

```text
Frame = bundle_i(Role_i XOR Filler_i)
Probe = Frame XOR RequestedRole
Filler = exact_cleanup(Probe)
```

Çoklu filler'larda probe tam filler değildir: diğer role/filler çiftleri
crosstalk oluşturur. Cleanup sözlüğü, yeterli Hamming marjı varsa istenen
filleri seçer.

### 4.2 Sıralı n-gram

```text
NGram = XOR_i Π^(N-i)(Token_i)
```

Bağlama işleminin değişmeli olmasına rağmen pozisyon permütasyonu sıralamayı
korur. `[A,B]` ve `[B,A]` bu nedenle farklıdır.

### 4.3 Causal bank

```text
Rule_i      = Cause_i XOR Π(Effect_i)
RuleBank    = bundle_i(Rule_i)
NoisyEffect = Π^(-1)(RuleBank XOR ObservedCause)
NextState   = exact_cleanup(NoisyEffect)
```

`infer_causal_step` cleanup'tan dönen **kayıtlı** hiper-vektörü ayrıca
`mem_get_by_id` ile alır. Bir önceki noisy probe'u zincire sokmaz; bu seçim
çok-adımlı drift'i sınırlar.

### 4.4 Aligned analogy

Yanlış fakat çekici formül:

```text
(Frame_source XOR Frame_target) XOR Query
```

önceden bundle edilmiş frame'lerde tek rolün eşini genel olarak ayırmaz.
Vegai'nin doğrulanmış formülü hizalanmış çiftleri kullanır:

```text
Mapping = bundle_i(SourceValue_i XOR TargetValue_i)
Answer  = exact_cleanup(Mapping XOR QuerySourceValue)
```

Bu “zero-shot” transfer, sorgu çiftinin kendisi ayrıca öğretilmeden ortak
structural alignment üzerinden olur. Ancak alignment listesi bilinmelidir;
hiçbir VSA cebiri semantik correspondence'i yoktan çıkarmaz.

## 5. Resonator Network factorization

`resonator_network` modülü, farklı semantic slot'lar için ayrı
`hds_memory_t` codebook'ları kabul eder. Hedef şu formdadır:

```text
T = F_1 XOR F_2 XOR ... XOR F_N
```

Her resonator turu, bir slotun güncellemesini diğer güncel tahminlerle
hesaplar:

```text
Q_i       = T XOR product(F_j, j != i)
F_i(next) = exact_cleanup(Codebook_i, Q_i)
```

`resonator_config_t%synchronous_updates=.true.` iken bütün `Q_i` değerleri
aynı eski state'ten hesaplanır. `false` değeri ise Gauss-Seidel tarzı in-place
asynchronous güncelleme kullanır. Her restart için başlangıç slotları seed'den
üretilir; en düşük composite Hamming residual saklanır.

Binary cleanup seçimi, sürekli-valued klasik resonator ağlardaki projection
operatorünün ayrık karşılığıdır. Bu nedenle iteratif ağ bir heuristic'tir;
`network_converged` yalnız network'ün target'a tam ulaşıp ulaşmadığını bildirir.
Varsayılan bounded exact verifier daha sonra sonucu tam olarak kontrol eder.
Verifier, factor'ları iki gruba bölüp sağ grup composite'lerini chained
fingerprint hash'inde saklar. Bir eşleşme ancak tüm 16.384 bit `hv_equal` ile
doğrulanınca kabul edilir. Fingerprint collision bu yüzden false-positive
üretemez.

Sonuçta `resonator_result_t` şunları birbirinden ayırır:

- `network_best_distance`: iteratif Resonator fazının en iyi residual'ı;
- `exact` ve `hamming_distance`: nihai doğrulanmış tuple'ın durumu;
- `used_exact_verifier`: bounded meet-in-the-middle certificate kullanıldı;
- `ambiguous` / `exact_solution_count`: en az iki exact tuple bulundu;
- `search_limit_reached`: verifier'ın her yarı için combination sınırı aşıldı.

Kapsamlı API, complexity ve çalışma örneği için
[RESONATOR_NETWORK.md](RESONATOR_NETWORK.md) belgesine bakın.

## 6. Corpus pretraining katmanı

`text_tokenizer`, `vocabulary` ve `hdc_pretraining`, mevcut sembolik HDC
çekirdeğinin üstünde iki-pass, bounded-memory bir distributional learning
katmanı oluşturur. Bu katman Transformer/attention/backpropagation içermez.

İlk pass token frequency'lerini fixed-capacity sözlükte sayar. Hash
fingerprint yalnız probe hızlandırıcısıdır; fingerprint hit'i tam token string
eşitliği ile doğrulanır. Frequency eşiği active mask'i belirler. Capacity
taşması, approximate eviction yerine deterministik hata üretir.

İkinci pass'te token identity hiper-vektörleri relative-position permutation
ile bundle edilir. Varsayılan `hybrid` context encoder buna ek olarak her
komşunun unpermuted identity görünümünü de aynı deterministic majority bundle'a
katar; positional role sinyali korunurken topical co-occurrence semantic
prototype'ları daha güçlü yakınlaştırır. `positional` encoder aynı positional
semantiği word-parallel bit-sliced binary counter ile yürütür ve eski `legacy`
signed-counter yolu ile bit-tam eşdeğerdir. `base_bits(HV_WORDS, capacity)` ve
`semantic_bits(HV_WORDS, capacity)` ayrı tutulur: identity değişmez, semantic
iz online güncellenir. Aynı corpus, config ve source commit için işlem sırası
bit-tam sonuç üretmelidir.

Ingest boundary'si de mimari sözleşmesidir: tokenizer ASCII-byte semantiklidir;
long token truncate edilmez, >8.192-byte fiziksel record prefix olarak
öğrenilmez ve tamamen düşürülür. Drop'lar metrik olur; strict modda hata olur.
Checkpoint'te vocabulary hash table saklanmaz, tokenlardan rebuild edilir;
context encoder dahil semantic config, planes, counters ve accidental-corruption
checksum saklanır.
Ayrıntılı contract ve data governance için [PRETRAINING.md](PRETRAINING.md)'ye
bakın.

Native corpus sınırının **önünde**, `tools/training_data_io.py` ve
`tools/training_dataset.py` schema-versioned `vegai-training-record/v1` data
boundary'sini sağlar. JSONL/JSON/TXT/Parquet/Arrow reader'ları lazy/generator
bazlıdır; rich `label`/`source`/`metadata` alanları native text'e otomatik
katılmaz. SQLite-backed exact text-dedupe, label imbalance audit ve SHA-256
hash split bu sınırda yapılabilir; yalnız explicit `write_hdc_corpus` projection'ı
strict ASCII/LF corpus üretir. Böylece Fortran tokenizer sözleşmesi zayıflatılmaz
ve rich source provenance canonical HDC text snapshot'a bağlanır. Ayrıntılı
schema, compression/memory sınırı ve operasyon örnekleri için
[TRAINING_DATA_FORMAT.md](TRAINING_DATA_FORMAT.md)'ye bakın.

Bu Fortran data plane'in üstünde Python production control plane bulunur:
`production_dataset.py`, strict ASCII/LF record'ları disk-backed exact SHA-256
dedupe, exact min-frequency-1 train vocabulary count ve deterministic held-out
split ile immutable shard package olarak atomik yayınlar.
`production_pretrain.py` dataset/binary/policy identity'sini kaydeder, run
öncesi exact record-level train/validation leakage + max-vocabulary capacity
audit ile resource plan yapar ve var olan
`continuous_pretrain.py` candidate-generation/state-journal protokolünü çağırır. Kontrol katmanı model semantiğini veya checkpoint data
plane'ini değiştirmez; Transformer/LMM runtime değildir. Tam operation/failure
contract için [PRODUCTION_PRETRAINING.md](PRODUCTION_PRETRAINING.md)'ye bakın.

## 7. Grounded diyalog retrieval katmanı

`dialogue_retrieval`, checkpoint semantic prototype'larini operator-supplied
`prompt<TAB>response` bilgi bankasi ile birlestiren ayri bir katmandir. Index
prompt'lari active semantic hypervector'lerin deterministic weighted majority
bundle'i olarak saklanir; response metni embedding, tokenizer input'u veya
executable instruction olmaz. `vegai_chat`, yalnizca secilen TSV response'u
aynen yayinlar.

Canli prompt bilinmeyen filler token'lar icerebilir; bunlar sayilir ve policy
izin verirse atlanir. Buna karsin publish edilen knowledge-base prompt'unun tum
token'lari checkpoint vocabulary'sinde active olmak zorundadir. Exact Hamming
tarama en iyi ve runner-up sonucu hesaplar; similarity veya margin esigi
gecilmezse result `found=.false.` kalir. Bu abstention mekanizmasi, zayif
semantic eslesmeyi dogal gorunen hardcoded/generative cevapla gizlemez.

Session yalniz son `max_history_turns` user-prompt HV'sini fixed-capacity ring
dizisinde tutar. Mevcut prompt ile bounded history tekrar bundle edilir; index,
checkpoint ve training state mutasyona ugramaz. Exact scan `O(N)` oldugu icin
module tek-host, bounded knowledge base contract'idir; buyuk-scale ANN/remote
search bir adapter olarak ayri tasarlanmalidir.

Bu retrieval plane'in ustunde `tools/dialogue_evaluation.py` ayri bir release
quality plane saglar: history kapali held-out JSONL case'lerini gerçek
`vegai_chat` ile calistirir, exact expected-prompt grounding ile expected
abstention'i olcer, raw prompt yerine hash raporlar ve checkpoint/KB/binary
kimliklerini baglar. Bu tool response uretmez veya HDC semantics'ini degistirmez;
operatore threshold calibration ve publish gate verir. Ayrintili veri,
confidence, evaluation ve operasyon contract'i için
[DIALOGUE_RETRIEVAL.md](DIALOGUE_RETRIEVAL.md) ve
[QUALITY_ASSURANCE.md](QUALITY_ASSURANCE.md)'ye bakın.

## 8. Optional gerçek LMM integration plane

Fortran HDC core'a Transformer olarak davranan yapay bir wrapper eklenmemistir.
Bunun yerine `tools/lmm_contract.py` etrafinda ayri bir model-runtime boundary
vardir. `vegai-lmm-model/v1`, gerçek backend engine/capability/context window,
model artifact ve tokenizer identity'sini baglar; `tokenizers-json` kullanilirsa
Unicode BPE/WordPiece/Unigram semantics modelin kendi tokenizer JSON'undan gelir.
Native HDC ASCII tokenizer'i bu plane tarafindan degistirilmez.

`lmm_session.py` hash-chain mesaj history'sini, `lmm_knowledge.py` source/license
bagli Unicode FTS evidence'i, `lmm_assets.py` ise content-addressed image/audio/
video referanslarini üretir. `lmm_gateway.py`, tam token budget'ine sigan pinned
policy/evidence + complete recent turns'i seçer, asset SHA'larini dogrular ve
yalniz capability ilan eden actual backend'e dispatch eder. OpenVINO GenAI text
adapter'i, HTTPS OpenAI-compatible endpoint'i ve local multimodal worker protocol'u
ayri seçeneklerdir. Missing backend/model/capability fail-closed olur.
`lmm_quality_evaluation.py`, operator-owned held-out request/evidence case'lerini
aynı gerçek gateway command'i üzerinden çalıştırır; modelin kendi ürettiği
citation/abstention marker'larını ölçer ve release threshold'unu fail-closed
uygular. Report ham prompt/response taşımadan provenance hash'lerini korur.
`lmm_portfolio.py`, bu quality report'u model manifest ve target-domain
measurement card ile bağlayarak domain/residency/safety/modality/SLO koşullarını
geçen en iyi expert'i seçer veya farklı provider/family için bağımsız panel planı
üretir. Uygun candidate yoksa cevap template'i üretmek yerine fail-closed olur.

Bu plane serbest generation'i gerçek modelin output'u olarak mümkün kilar;
factuality, safety, neural training, model quality veya GPU/NPU speed iddiasini
tek basina kanitlamaz. Tam package/worker, privacy, long-context, held-out gate,
measured expert routing ve benchmark contract'i için [LMM_RUNTIME.md](LMM_RUNTIME.md)
ile [LMM_PORTFOLIO_ROUTING.md](LMM_PORTFOLIO_ROUTING.md)'ye bakın.

## 9. Paralellik ve performans sınırları

`hv_bind`, `hv_bundle3`, `hv_hamming` ve mask işlemlerinde `!$omp simd`
ipucu bulunur. `mem_cleanup` global taraması `!$omp parallel do reduction(min:...)`
kullanabilir. Pretraining'de bir record'un immutable base-vector context'leri
OpenMP ile paralel hazırlanır; semantic prototype ve experience-counter
commit'i ise source-order'da kalır. Böylece repeated token tie seed'leri
korunur ve serial/multi-core run checkpoint byte'ları aynı kalır. OpenMP
kapalıysa direktifler yorum satırıdır ve semantik aynı kalır.

Derleyicinin popcount/SIMD seçimi hedefe özgüdür. Dağıtılabilir varsayılan
profil host-ISA flag'i kullanmaz; CPU-specific inspection yalnız native host
binary üzerinde yapılmalıdır:

```bash
make native
objdump -d build-native/vegai_pretrain | grep -Ei 'popcnt|vpopcnt|xor'

# GNU Fortran vectorization raporu için, sadece build hostunda:
gfortran -O3 -std=f2023 -march=native -fopenmp -fopt-info-vec-optimized \
  src/hyper_types.f90 src/symbolic_memory.f90 src/resonator_network.f90 \
  src/reasoning_core.f90 app/main.f90
```

`VEGAI_CPU_TARGET=portable` / varsayılan `make all` Intel Core Ultra 5 125H
dahil compatible CPU'lar için CPU fallback'tir; GPU/iGPU/NPU API'sine bağlı
değildir. GPU/NPU inventory compute dispatch anlamına gelmez. Portable/native
profile sözleşmesi, OS build yolları ve checkpoint ABI sınırları için
[HARDWARE_COMPATIBILITY.md](HARDWARE_COMPATIBILITY.md)'ye; RAM/SSD/NUMA,
OpenMP ve accelerator provider sınırları için
[RESOURCE_AWARE_TRAINING.md](RESOURCE_AWARE_TRAINING.md)'ye bakın.

Performans ölçümü için warm-up, CPU frekans ölçekleme kontrolü, `OMP_NUM_THREADS`
sabitleme ve gerçek sözlük/workload dağılımı gereklidir. Tek bir clock-cycle
sayısı veya belirli bir SIMD instruction count API garantisi değildir.
