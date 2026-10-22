# Vegai — doğrulanmış HDS-Fortran VSA/HDC motoru

[![Fortran verification](https://github.com/sswweett00/Vegai/actions/workflows/ci.yml/badge.svg)](https://github.com/sswweett00/Vegai/actions/workflows/ci.yml)

Vegai, **16.384 boyutlu ikili/bipolar hiper-vektörler** üzerinde çalışan, modern
Fortran 2018 ile yazılmış küçük ve denetlenebilir bir **Vector Symbolic
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
- yerel Fortran özellik/entegrasyon testleri, bağımsız Python referans modeli
  ve GitHub Actions ile denetimli derleme.

> **Önemli doğruluk notu:** HDC, Transformer/LMM'lerin genel amaçlı yerine
> geçeni değildir ve “her iş yükünde katbekat hızlı” olduğu iddia edilmez.
> Bit-seviyesi işlemler, özellikle sabit sembol sözlükleri ve bileşimsel
> anahtarlar için çok ucuzdur; doğruluk/kapasite, bundle kardinalitesi,
> sözlük büyüklüğü ve sorgu gürültüsüne bağlı istatistiksel bir konudur.

## Hızlı başlangıç

### GNU Fortran + Make

```bash
# GNU Fortran 12+ ve OpenMP çalışma zamanı gerekir
make test
make run

# Sınır denetimli hata ayıklama derlemesi
make debug

# Bağımsız cebirsel/reference doğrulama
make check
```

Varsayılan Make derlemesi `-O3 -march=native -fopenmp` kullanır. Taşınabilir
bir ikili için örneğin:

```bash
make clean
make engine OPT_FLAGS='-O3' OPENMP_FLAGS=''
```

### CMake / CTest

```bash
cmake -S . -B build -DVEGAI_ENABLE_OPENMP=ON -DVEGAI_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
OMP_NUM_THREADS=2 ./build/hds_engine
```

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
fpm test --flag '-std=f2018 -fopenmp' --link-flag '-fopenmp'
fpm run --flag '-O3 -fopenmp' --link-flag '-fopenmp'
```

OpenMP isteğe bağlıdır: direktifleri desteklemeyen veya `-fopenmp` verilmeyen
bir derleyicide cleanup taraması doğru fakat sıralı olarak çalışır.

## Mimari özeti

| Katman | Dosya | Sorumluluk |
|---|---|---|
| Bit cebiri | `src/hyper_types.f90` | üretim, XOR binding, majority-3, permütasyon, Hamming, exact noise, Hebbian sayaç |
| Çağrışımsal bellek | `src/symbolic_memory.f90` | sabit kapasiteli SoA store, ID hash, LSH, exact/approximate cleanup |
| Akıl yürütme | `src/reasoning_core.f90` | role–filler frame, n-gram, kural bankası, analogik transfer |
| Çalıştırılabilir doğrulama | `app/main.f90` | %25 exact bit gürültüsü, analoji, 4 adımlı nedensel zincir |
| Native testler | `test/test_all.f90` | cebirsel invariantlar, hata yolları, bellek ve uçtan uca test |
| Bağımsız oracle | `tools/verify_reference.py` | standart SplitMix64 ve demo sözleşmelerinin Python modeli |

Bir hiper-vektörün payload'u tam olarak `256 × 8 = 2048` bayttır. Bu, 64-byte
cache-line varsayımında **32 cache-line payload birimine** eşittir. Fortran'ın
standart `allocate` işlemi 64-byte başlangıç hizalamasını garanti etmez; bu
nedenle gerçek fiziksel cache-line sayısının her platformda tam 32 olacağını
iddia etmiyoruz. Sıfır-kayıplı 64-byte hizalama gerekiyorsa derleyici/ABI'ye
özel hizalı allocator uygulanmalıdır.

Ayrıntılı tasarım ve invariantlar için [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
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
- **Bundling:** `hv_bundle3` üçlü bitwise majority yapar. Daha genel bundle
  için `hv_acc_t` + `hv_acc_add` + `hv_acc_finalize` kullanılır. Çift
  kardinalitede tie-breaker seed'i zorunlu olarak deterministiktir.
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
- CPU SIMD komutunun gerçekten AVX-512 `VPOPCNTDQ` veya ARM `CNT` olması
  derleyici, hedef CPU ve bayraklara bağlıdır. Bunu assembly/optimization
  report ile ölçün; kaynak kod tek başına komut sayısı garantilemez.

## Lisans

MIT — [LICENSE](LICENSE).
