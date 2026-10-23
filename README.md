# Vegai — doğrulanmış HDS-Fortran VSA/HDC motoru

[![Fortran verification](https://github.com/sswweett00/Vegai/actions/workflows/ci.yml/badge.svg)](https://github.com/sswweett00/Vegai/actions/workflows/ci.yml)

Vegai, **16.384 boyutlu ikili/bipolar hiper-vektörler** üzerinde çalışan,
**Fortran 2023** hedefli, küçük ve denetlenebilir bir **Vector Symbolic
Architecture (VSA) / Hyperdimensional Computing (HDC)** çekirdeğidir. Yoğun
kayan nokta matrisleri yerine `256 × int64` paketli vektörler, XOR bağlama,
çoğunluk süperpozisyonu, döngüsel permütasyon ve Hamming/`popcnt` rezonansı
kullanır.

Bu depo bir tasarım taslağı değil; aşağıdakileri içerir:

- taşma-semantikleri tanımlı deterministik SplitMix64 tabanlı vektör üretimi;
- 2 KiB paketli hiper-vektör cebiri ve geri çevrilebilir bağlama;
- taşma denetimli, atomik başarısız olan 32 KiB Hebbian bundling biriktiricisi;
- sabit kapasiteli, heap-parçalanmasız, gerçek paketli SoA çağrışımsal bellek;
- açıkça **exact** ve **yaklaşık LSH** cleanup API'lerinin ayrılması;
- role–filler çerçeve çözümleme, sıralı n-gram, analoji ve nedensel zincir;
- çoklu-faktör ayrıştırma için deterministic multi-restart **Resonator Network**
  ve bounded meet-in-the-middle exact verifier;
- iki-pass vocabulary, allocation-free ASCII ingest, streaming positional-context
  Hebbian öğrenme, checkpoint/resume telemetry ve SHA-256 journal'lı continual
  shard pretraining içeren HDC/VSA corpus pipeline;
- immutable context construction + source-ordered commit ile checkpoint-bit-tam
  deterministic OpenMP multi-core training; RAM/SSD/NUMA resource preflight;
- hardcoded cevap üretmeyen, operator TSV bilgi bankasindan grounded HDC
  retrieval yapan bounded chat/REPL yuzeyi ve abstention kapisi;
- yerel Fortran özellik/entegrasyon testleri, bağımsız Python referans modeli
  ve GitHub Actions ile denetimli derleme.

> **Önemli doğruluk notu:** HDC, Transformer/LMM'lerin genel amaçlı yerine
> geçeni değildir ve “her iş yükünde katbekat hızlı” olduğu iddia edilmez.
> Bit-seviyesi işlemler, özellikle sabit sembol sözlükleri ve bileşimsel
> anahtarlar için çok ucuzdur; doğruluk/kapasite, bundle kardinalitesi,
> sözlük büyüklüğü ve sorgu gürültüsüne bağlı istatistiksel bir konudur.
>
> **Corpus pretraining notu:** `vegai_pretrain`, Transformer/LLM eğitimi değil;
> deterministik positional + order-invariant hybrid context'lerden HDC semantic
> prototype öğrenimidir. Varsayılan `hierarchical`, uzun kayıtlarda bounded
> whole-record topical evidence ekler ve kısa kayıtlarda `hybrid` ile bit-tam
> aynıdır. `vegai_chat` da free-form cevap üreten bir LLM değildir; yalnız
> versioned TSV bilgi bankasindan grounded response retrieval yapar veya
> abstain eder. Unicode corpus'lar ön-tokenization
> ister; veri yönetişimi, checkpoint ve sınırlar [PRETRAINING.md](docs/PRETRAINING.md)'de, immutable
> shard'larla crash-safe continual çalışma ise
> [CONTINUOUS_PRETRAINING.md](docs/CONTINUOUS_PRETRAINING.md)'de açıkça tanımlıdır.
> Gerçek LMM/LLM ile mevcut HDC production kapsamı arasındaki kaynak-temelli gap
> analizi [LMM_SCOPE_AND_GAP_ANALYSIS.md](docs/LMM_SCOPE_AND_GAP_ANALYSIS.md)'dedir.

## Hızlı başlangıç

### GNU Fortran + Make

```bash
# Fortran 2023 modu destekleyen GNU Fortran ve OpenMP çalışma zamanı gerekir
make test
make run
make pretrain
make chat

# Sınır denetimli hata ayıklama derlemesi
make debug

# Bağımsız cebirsel/reference doğrulama
make check
```

Varsayılan Make derlemesi artık host-ISA bağımsız portable `-O3` profilidir;
`-march=native` varsayılan değildir. Bu, Intel Core Ultra 5 125H dahil uyumlu
CPU'larda aynı binary'nin illegal-instruction riski olmadan çalıştırılabilmesi
içindir. Aynı hostta opsiyonel tuning için ayrı native binary üretin:

```bash
make clean-all
make all                    # ./build/vegai_pretrain — portable profil
make native                 # ./build-native/vegai_pretrain — yalnız build hostu için

python3 tools/hardware_compatibility.py --format text --require-recognized-os
python3 tools/compare_pretraining_build_profiles.py \
  --portable-pretrain-bin ./build/vegai_pretrain \
  --native-pretrain-bin ./build-native/vegai_pretrain \
  --repetitions 3 --omp-threads 1
```

GPU/iGPU/NPU runtime'ı portable CPU/OpenMP yolu için gerekli değildir. GPU/NPU
adapter inventory'si tek başına compute iddiası değildir. Intel Core Ultra için
opt-in OpenVINO/oneAPI context provider vardır: immutable packed-HDC context
bundle işlemini gerçek worker device'ına dispatch eder, source-ordered
prototype commit'i CPU'da tutar ve checkpoint byte equality ile doğrulanır.
Bir `train` çağrısında worker resident kalır ve dynamic-batch graph cache'i
tekrar kullanılır. Açık CPU fallback policy'si ise OpenVINO CPU graph'ını
başlatmak yerine native fused/OpenMP context kernel'ine gider; bu, dürüstçe
CPU execution olarak raporlanır. `--openvino-require-accelerator` gerçek
GPU/NPU execution yoksa fail closed döner. RAM,
SSD, dual-channel/NUMA ve çok-core planı için
[RESOURCE_AWARE_TRAINING.md](docs/RESOURCE_AWARE_TRAINING.md)'ye; platform/
CPU/checkpoint sınırları için [HARDWARE_COMPATIBILITY.md](docs/HARDWARE_COMPATIBILITY.md)'ye bakın.

### CMake / CTest

```bash
cmake -S . -B build -DVEGAI_CPU_TARGET=portable -DVEGAI_ENABLE_OPENMP=ON -DVEGAI_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
OMP_NUM_THREADS=2 ./build/hds_engine

# Küçük deterministic corpus smoke run
./build/vegai_pretrain train examples/corpus/toy_hdc_corpus.txt toy.vgai \
  --max-vocab 64 --min-frequency 2 --epochs 2 --query king
./build/vegai_pretrain inspect toy.vgai --query dog

# Dengeli, Türkçe-ASCII sentetik fixture ile checkpoint/resume örneği
python3 tools/training_resource_plan.py \
  --corpus examples/corpus/mini_turkish_ascii_pretrain.txt --checkpoint mini.vgai \
  --max-vocab 128 --require-ready
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
./build/vegai_pretrain train examples/corpus/mini_turkish_ascii_pretrain.txt mini.vgai \
  --max-vocab 128 --min-frequency 8 --window 2 --context-mode hierarchical \
  --epochs 2 --strict --query okul

# Operator-supplied TSV'den grounded retrieval; cevap source code'dan degil
# knowledge base'den gelir, esik gecmezse answer uydurulmaz.
./build/vegai_chat ask mini.vgai examples/dialogue/mini_turkish_hdc_dialogue.tsv \
  --prompt "okul ders" --max-entries 64 --history-turns 0
python3 tools/verify_example_corpus.py --pretrain-bin ./build/vegai_pretrain
python3 tools/compare_pretraining_modes.py --pretrain-bin ./build/vegai_pretrain \
  --repetitions 3 --omp-threads 1

# Gerçek corpus'u eğitmeden önce local digest + ingest preflight
python3 tools/corpus_manifest.py CORPUS.txt --output corpus-manifest.json \
  --source 'dataset URI/version' --license 'license ID' --strict-contract

# Ordered immutable shard'larla crash-safe continual pretraining
python3 tools/continuous_shard_manifest.py data/shards/000001.txt data/shards/000002.txt \
  --output data/continual-ledger.json --source 'dataset version' --license 'license ID'
python3 tools/continuous_pretrain.py --pretrain-bin ./build/vegai_pretrain \
  --manifest data/continual-ledger.json --state-dir state/continual \
  --max-vocab 50000 --min-frequency 1 --context-mode hierarchical --epochs 1

# Controller çalıştırmadan completed journal/checkpoint/shard health audit'i
python3 tools/continuous_pretrain_status.py \
  --manifest data/continual-ledger.json --state-dir state/continual \
  --verify-shards --require-complete --require-unlocked --format text

# Production data foundry + record-level held-out leakage gate + auditable run
python3 tools/production_dataset.py --input governed-corpus.txt \
  --output-dir production-datasets/release-v1 \
  --source 'dataset registry/release-v1' --license 'approved-license-id' \
  --retrieved-at 'registry-event-2026-10-23' \
  --preprocessing 'versioned governed upstream pipeline' \
  --validation-basis-points 100
python3 tools/production_pretrain.py --pretrain-bin ./build/vegai_pretrain \
  --dataset production-datasets/release-v1/dataset.json \
  --run-dir production-runs/release-v1-hdc \
  --max-vocab 20000 --min-frequency 1 --context-mode hierarchical --epochs 1
```

GNU CMake hedefi `-std=f2023` bayrağını uygulama, test ve kütüphane hedeflerine
aktarır. Başka bir compiler kullanıyorsanız eşdeğer Fortran 2023 conformance
bayrağını doğrulayın.

Sınır ve çalışma zamanı kontrolleri için:

```bash
cmake -S . -B build-debug -DVEGAI_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_Fortran_FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'
cmake --build build-debug --parallel
ctest --test-dir build-debug --output-on-failure
```

### Fortran Package Manager

```bash
fpm test --flag '-std=f2023 -fopenmp' --link-flag '-fopenmp'
fpm run --flag '-O3 -std=f2023 -fopenmp' --link-flag '-fopenmp'
```

OpenMP isteğe bağlıdır: direktifleri desteklemeyen veya `-fopenmp` verilmeyen
bir derleyicide cleanup taraması doğru fakat sıralı olarak çalışır.

## Mimari özeti

| Katman | Dosya | Sorumluluk |
|---|---|---|
| Bit cebiri | `src/hyper_types.f90` | üretim, XOR binding, majority-3, permütasyon, Hamming, exact noise, Hebbian sayaç |
| Çağrışımsal bellek | `src/symbolic_memory.f90` | sabit kapasiteli SoA store, ID hash, LSH, exact/approximate cleanup |
| Akıl yürütme | `src/reasoning_core.f90` | role–filler frame, n-gram, kural bankası, analogik transfer |
| Resonator Network | `src/resonator_network.f90` | slot-spesifik codebook'larla çoklu-faktör çözümleme ve exact certification |
| Corpus ingest | `src/text_tokenizer.f90`, `src/vocabulary.f90` | bounded ASCII canonicalization, collision-safe frequency vocabulary |
| HDC pretraining | `src/hdc_pretraining.f90` | two-pass stream, hierarchical default, deterministic OpenMP / optional OpenVINO immutable-context preparation + source-ordered CPU commit, metrics, checkpoint/resume |
| Pretraining CLI | `app/pretrain.f90` | `train`, `resume`, `continue`, `inspect`, CPU/OpenVINO context-provider execution switch ve nearest semantic diagnostic |
| Grounded dialogue retrieval | `src/dialogue_retrieval.f90`, `app/chat.f90` | operator TSV response index'i, strict ASCII/OOV gate, exact Hamming retrieval, confidence/margin abstention ve bounded turn history; free-form LLM generation degil |
| Continual transaction + health audit | `tools/continuous_shard_manifest.py`, `tools/continuous_pretrain.py`, `tools/continuous_pretrain_status.py` | immutable SHA-256 ledger, strict ingest gate, candidate generation, atomic state journal, read-only checkpoint/prefix/payload audit |
| Production data/run control | `tools/production_dataset.py`, `tools/production_pretrain.py` | disk-backed exact record/token dedupe, deterministic held-out split, content-addressed shard package, exact leakage + vocabulary-capacity gate, immutable binary/data/policy provenance, real exactly-once run wrapper |
| Kaynak preflight | `tools/training_resource_plan.py` | bounded RAM/checkpoint storage estimate, CPU/NUMA/OpenMP recommendation, static accelerator truth contract |
| OpenVINO context worker | `tools/openvino_context_worker.py` | optional Intel uint32-lane immutable-context bridge; resident per-train service, dynamic-batch compiled-graph cache, execution mask / native-CPU bypass contract |
| Çalıştırılabilir doğrulama | `app/main.f90` | %25 exact gürültü, analoji, causal chain ve 3-faktör Resonator demo |
| Native testler | `test/test_all.f90` | cebirsel invariantlar, hata yolları, serial/OpenMP pretraining eşitliği ve uçtan uca test |
| Donanım/OS profili | `tools/hardware_compatibility.py` | privacy-conscious CPU/RAM/GPU/NPU inventory, portable CPU backend sözleşmesi |
| Bağımsız oracle | `tools/verify_reference.py` | standart SplitMix64 ve demo sözleşmelerinin Python modeli |

Bir hiper-vektörün payload'u tam olarak `256 × 8 = 2048` bayttır. Bu, 64-byte
cache-line varsayımında **32 cache-line payload birimine** eşittir. Fortran'ın
standart `allocate` işlemi 64-byte başlangıç hizalamasını garanti etmez; bu
nedenle gerçek fiziksel cache-line sayısının her platformda tam 32 olacağını
iddia etmiyoruz. Sıfır-kayıplı 64-byte hizalama gerekiyorsa derleyici/ABI'ye
özel hizalı allocator uygulanmalıdır.

Ayrıntılı tasarım ve invariantlar için [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
Resonator Network sözleşmesi için [docs/RESONATOR_NETWORK.md](docs/RESONATOR_NETWORK.md),
corpus veri hattı, yönetişim ve checkpoint sözleşmesi için
[docs/PRETRAINING.md](docs/PRETRAINING.md), immutable shard ledger ve
crash-safe exactly-once continual pretraining için
[docs/CONTINUOUS_PRETRAINING.md](docs/CONTINUOUS_PRETRAINING.md), data-driven grounded
chat/retrieval sözleşmesi için [docs/DIALOGUE_RETRIEVAL.md](docs/DIALOGUE_RETRIEVAL.md), production data foundry,
record-level leakage gate ve immutable run provenance için
[docs/PRODUCTION_PRETRAINING.md](docs/PRODUCTION_PRETRAINING.md), gerçek LMM/LLM
hedefine ilişkin kapsam/gap analizi için
[docs/LMM_SCOPE_AND_GAP_ANALYSIS.md](docs/LMM_SCOPE_AND_GAP_ANALYSIS.md), fixture üzerinde gerçek yürütme
telemetrisi için [docs/EXAMPLE_PRETRAINING_RUN.md](docs/EXAMPLE_PRETRAINING_RUN.md),
önce/sonra hız ve controlled semantic-cohesion sonuçları için
[docs/PRETRAINING_MODE_COMPARISON.md](docs/PRETRAINING_MODE_COMPARISON.md),
CPU/iGPU/OS portable build sözleşmesi için
[docs/HARDWARE_COMPATIBILITY.md](docs/HARDWARE_COMPATIBILITY.md), RAM/SSD,
çok-core, quality default ve GPU/NPU dispatch sınırları için
[docs/RESOURCE_AWARE_TRAINING.md](docs/RESOURCE_AWARE_TRAINING.md),
ölçülebilir doğrulama planı ve istatistiksel sınırlar için
[docs/VALIDATION.md](docs/VALIDATION.md) dosyalarına bakın.

## Temel işlem sözleşmeleri

İkili gösterimde `0 → -1`, `1 → +1` bipolar bileşenini temsil eder.

```fortran
use hyper_types

type(hv_t) :: role, filler, pair, restored
role = hv_from_seed(101_int64)
filler = hv_from_seed(202_int64)
pair = hv_bind(role, filler)       ! XOR / bipolar çarpım
restored = hv_bind(role, pair)     ! filler ile bit-bit aynıdır
```

- **Binding:** `A ⊗ B = A XOR B`; `A ⊗ (A ⊗ B) = B`.
- **Bundling:** `hv_bundle3` üçlü bitwise majority yapar. Genel weighted
  bundle için `hv_acc_t` + `hv_acc_add` + `hv_acc_finalize`; unit-weight,
  yüksek-throughput context bundle için bit-sliced `hv_bitacc_t` kullanılır.
  Çift kardinalitede tie-breaker seed'i zorunlu olarak deterministiktir.
- **Permutation:** pozitif kaydırma global bit konumu `p`'yi
  `p + k (mod D)` konumuna taşır. `Π⁻ᵏ(Πᵏ(A)) = A` native test ile sınanır.
- **Similarity:** `1 - 2·Hamming(A,B)/16384` aralığı `[-1,+1]` içindedir.
  `+1` eşitlik, `0` bağımsız rastgele vektör beklentisi, `-1` bit tümleyenidir.

`hv_inject_noise_exact` benzersiz pozisyon seçerek **tam** istenen bit
sayısını çevirir. Örneğin `%25`, 4096 flip demektir; eski “yaklaşık %25” mask
gibi eşik grubu değildir.

## Cleanup belleği: doğruluk modu farkı

`mem_cleanup` **exact nearest-neighbour** çağrısıdır. Aynı LSH kovasında
Hamming mesafesi sıfır olan bir aday görürse güvenli erken döner; aksi halde
tüm kayıtlı sembolleri tarar. En küçük mesafe eşitliklerinde en küçük slot
seçilir; OpenMP kullanılsa bile sonuç deterministiktir.

`mem_cleanup_lsh` ise yalnızca 12-bit proje kovasındaki en fazla 8 adayı
inceler. Bu API yüksek hızlı ama **yaklaşıktır**; kova taşması veya gürültü
nedeniyle gerçek en yakın komşuyu atlayabilir. İki API'nin ayrı tutulması LSH
heuristic'ini yanlışlıkla doğrulanmış retrieval olarak kullanmayı önler.

## Resonator Network ile çoklu-faktör çözümleme

`resonator_factorize`, hedef bir bağlı kompozisyonu slot-spesifik codebook'lara
karşı ayrıştırır:

```text
T = F₁ XOR F₂ XOR ... XOR Fₙ
Fᵢ <- Cleanupᵢ(T XOR product(Fⱼ, j ≠ i))
```

Her iterasyonda bir factor için diğer geçici tahminler unbind edilir ve yalnız
onun codebook'unda exact cleanup yapılır. Ağ synchronous veya asynchronous
update, deterministik restart ve en iyi residual kaydı sunar. Binary VSA'da
rastgele codebook'lar için bu iteratif faz **heuristiktir**; çözüm bulunduğu
varsayılmaz.

Bu nedenle varsayılan ayarda modül, her yarısı `max_exact_combinations` ile
sınırlandırılmış bir meet-in-the-middle verifier da çalıştırır. Verifier:

- tam bit eşitliği ile sonucu sertifikalar;
- 64-bit fingerprint çakışmalarını her hit'te tam `hv_equal` ile eler;
- en az iki eşdeğer tuple bulursa `RN_WARN_AMBIGUOUS` döndürür;
- sınır aşılırsa sonucu uydurmak yerine `RN_ERR_SEARCH_LIMIT` bildirir.

Demo, üç bağımsız üçlü candidate codebook'tan `HIPPOCAMPUS ⊗ HEAT_RISE ⊗ FLAGS`
tuple'ını geri getirir. API, konfigürasyon ve karmaşıklık ayrıntıları
[Resonator Network belgesinde](docs/RESONATOR_NETWORK.md) verilmiştir.

## Düzeltilen tasarım riskleri

İlk mimari fikrindeki performans hedefini korurken, üretim kodunda şu
hatalar/yanıltıcı varsayımlar giderildi:

1. **İmzasız taşma:** Fortran'da signed integer overflow taşınabilir değildir.
   SplitMix64 sabitleri, 2⁶⁴ toplama ve çarpma `add_mod64`/16-bit limb
   `mul_mod64` ile bit-tam olarak uygulanır.
2. **SoA ve `contiguous`:** `contiguous` allocatable component niteliği
   standart-geçerli değildir ve `type(hv_t)::vectors(:)` gerçek SoA değildir.
   Bellek `integer(int64)::vector_bits(HV_WORDS, capacity)` olarak paketlidir.
3. **Hash overflow/sentinel:** çarpımlı signed hash kaldırıldı; bit-mixing ve
   ayrı `hash_used` dizisi sayesinde `-1` dahil tüm `int32` ID'ler kayıtlıdır.
4. **Duplicate/stale LSH:** duplicate ID hata verir; Hebbian değişimden sonra
   LSH indeksi yeniden kurulur.
5. **Yanlış LSH erken çıkışı:** “yakın” LSH adayı artık exact sonuç diye
   dönmez; yalnızca mesafe sıfırsa güvenli erken çıkış vardır.
6. **Geçersiz holistic analogy:** iki zaten-bundle edilmiş frame'in XOR'u rol
   eşleşmesini genel olarak izole etmez. `encode_analogy_mapping`, hizalanmış
   `source_i ⊗ target_i` ilişkilerini bundle eder; doğrulanmış zero-shot
   transfer bunun üzerinden yapılır.
7. **Sayaç wraparound:** `int16` biriktiricinin tüm 16.384 counter'ı önce
   denetlenir. Taşmada durum değişmez ve hata kodu döner.

## Sınırlar ve güvenlik modeli

- Hiper-vektörler şifreleme değildir. Seed/vektör gizliliği veya saldırgana
  karşı bütünlük garantisi sağlamaz.
- Deterministik seed'ler reproducibility içindir; kullanıcı gizli verisini
  doğrudan seed olarak kullanmamalıdır.
- `hv_acc_t` sayacı her bileşende `[-32768, 32767]` sınırındadır. Çok uzun veya
  yüksek ağırlıklı bundle'lar parçalı/çok-seviyeli aggregation gerektirir.
- Kural/frame sonucu temizlenmeden tekrar bağlanırsa crosstalk zincir boyunca
  büyür. `infer_causal_step` bu nedenle her adımda cleanup yapıp kaydedilmiş
  vektörü geri alır.
- Resonator iterasyonları özellikle geniş rastgele codebook'larda yerel sabit
  noktalara düşebilir. `exact=.true.` veya zero residual sonucu doğrular; aksi
  durumda çözüm bulunduğu varsayılmamalıdır. Exact verifier sınırı aşıldığında
  problem boyutunu/slot codebook'larını küçültün ya da ayrı bir search stratejisi kullanın.
- CPU SIMD komutunun gerçekten AVX-512 `VPOPCNTDQ` veya ARM `CNT` olması
  derleyici, hedef CPU ve bayraklara bağlıdır. Bunu assembly/optimization
  report ile ölçün; kaynak kod tek başına komut sayısı garantilemez.

## Lisans

MIT — [LICENSE](LICENSE).
