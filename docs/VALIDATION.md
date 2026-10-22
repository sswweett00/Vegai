# Doğrulama, istatistik ve kabul kriterleri

Vegai'nin doğrulaması üç birbirini tamamlayan seviyededir:

1. **native Fortran testleri** (`test/test_all.f90`): derlenen gerçek API'nin
   contract/property testleri;
2. **bağımsız Python referans modeli** (`tools/verify_reference.py`): bit
   cebiri, standart SplitMix64 oracle'ı ve demo senaryosunun farklı bir dilde
   yeniden hesaplanması;
3. **CI checked build** (`.github/workflows/ci.yml`): GNU Fortran,
   `-fcheck=all`, floating-point trap, OpenMP ve CTest.

Bu katmanlar aynı bug sınıfını farklı şekillerde yakalamaya yöneliktir. Python
modeli Fortran derleyicisinin yerine geçmez; native testler de referans modelin
yerine geçmez.

## Çalıştırma matrisi

| Komut | Kapsam | Beklenen sonuç |
|---|---|---|
| `make check` | bağımsız reference property + demo | `Python reference verification: PASS` |
| `make test` | native cebir, error path, memory, reasoning | `Checks: ..., failures: 0` |
| `make debug` | bounds/runtime/FPE denetimli native test | başarısızlık yok |
| `make run` | 512+ sembollü demo | üç assertion ve 4 causal adım PASS |
| `ctest --test-dir build` | CMake entegrasyonu | `vegai_fortran_properties` PASS |

CI bunların checked CMake/OpenMP varyantını ve ayrıca optimize Make hedefini
çalıştırır.

## Native test kapsamı

### `hyper_types`

- aynı seed → bit-bit aynı hiper-vektör;
- farklı seed'ler için Hamming mesafesinin geniş ama anlamlı near-orthogonal
  aralığı (`7000 < H < 9400`);
- XOR identity ve self-inverse unbinding;
- forward/inverse permutation ve dimension wrap;
- three-way accumulator sonucu = boolean majority-3;
- `int16` counter overflow'unda status ve **atomic state preservation**;
- tam 4096 bit noise, determinism ve invalid argument davranışı;
- Hebbian mask determinism ve değişen bit sayısının alt sınırı.

### `symbolic_memory`

- init/destroy lifecycle ve footprint;
- custom vector kayıtları;
- `-1` negatif ID lookup;
- duplicate, missing ID, empty cleanup ve capacity overflow hata yolları;
- full exact cleanup ve LSH-only cleanup metadata'sı;
- 3000-bit gürültü altında cleanup;
- Hebbian prototype değişiminden sonra LSH rebuild ve exact recall.

### `reasoning_core`

- 3-role frame encode/unbind;
- boyut uyumsuzluğu ve empty sequence hata yolları;
- n-gram sıra duyarlılığı;
- 3-pair aligned analogy;
- 4-rule causal bank üzerinden dört ardışık cleanup adımı;
- 50 distractor ile cleanup marjı.

## Olasılık modeli ve sayısal marjlar

Rastgele bağımsız binary hiper-vektörler için:

```text
H(A,B) ~ Binomial(D, 1/2), D = 16,384
E[H]   = 8,192
σ_H    = sqrt(D/4) = 64
```

Hoeffding/Chernoff sınırı:

```text
P(|H − D/2| >= t) <= 2 exp(-2t² / D)
```

Similarity için `|sim| >= ε`, `t = εD/2` olduğundan:

```text
P(|sim| >= ε) <= 2 exp(-ε²D/2)
```

Örneğin `ε=0.10` için üst sınır yaklaşık `5.3×10^-36`dır. Bu, ideal bağımsız
rastgele model altındaki teorik üst sınırdır; deterministik PRNG'nin tüm
seed/derleyici/işlemci kombinasyonlarına dair kriptografik kanıt değildir.

### Demo 1: 3-item frame + %25 exact noise

Üçlü majority bundle'da doğru bound pair'in her bitinin korunma olasılığı
`3/4`'tür. Sonra bağımsız olarak %25 flip uygularsak:

```text
P(correct after noise) = 0.75×0.75 + 0.25×0.25 = 0.625
Expected similarity    = 2×0.625 − 1 = 0.25
Expected distance      = 6,144
```

Rastgele sözlük vektörü 8.192±64 uzaklıktadır. Hedef ile rastgele merkezler
arasındaki yaklaşık 2.048-bit fark, ideal modelde 32σ düzeyindedir. Bu nedenle
512 sembollü demo sözlüğünde robust retrieval beklenir. Native ve Python
testleri belirlenmiş seed'lerde beklenen ID'yi kontrol eder.

### Demo 2: 3-aligned-pair analogy

`Mapping = majority(Source_i XOR Target_i)` üç item içerir. Sorgu source
vektörü ile unbind edildiğinde hedef pair bitinin beklenen doğruluğu `3/4`,
beklenen similarity'si `0.5` olur. Cleanup marjı frame testinden daha yüksektir.

### Demo 3: 4-rule causal bank

Bir hedef rule + üç crosstalk rule için, deterministic random tie-breaker
modelinde doğru bit olasılığı `11/16 = 0.6875`; beklenen similarity `0.375`'tir.
Her adımda cleanup yapılması bu gürültünün sonraki adıma taşınmasını engeller.

Bu rakamlar **kapasite garantisi değildir**. Bundle item sayısı arttıkça hedef
sinyali zayıflar; sözlük büyüdükçe rastgele en yakın komşu riski artar. Üretim
kullanımında kendi sözlük/bundle/gürültü dağılımınızla ROC, top-k accuracy ve
latency benchmarkı ölçün.

## Negatif testler ve fault containment

| Risk | Davranış |
|---|---|
| `max_symbols <= 0` | `MEM_ERR_INVALID_ARGUMENT` |
| allocation failure | `MEM_ERR_ALLOCATION`, kısmi alanlar serbest bırakılır |
| duplicate ID | `MEM_ERR_DUPLICATE_ID`, mevcut entry korunur |
| full memory | `MEM_ERR_FULL`, yeni entry eklenmez |
| missing ID | `MEM_ERR_NOT_FOUND`, output zero + `found=.false.` |
| empty cleanup | `MEM_ERR_EMPTY` |
| boş/uyumsuz reasoning dizileri | `REASON_ERR_INVALID_SHAPE` |
| `int16` accumulator overflow | `HV_ERR_ACCUMULATOR_OVERFLOW`, hiçbir count mutasyona uğramaz |
| invalid noise count/percent | `HV_ERR_INVALID_ARGUMENT`, output input'a eşit kalır |
| LSH bucket boş | `MEM_ERR_NO_LSH_CANDIDATE` yalnız approximate API'de |

## Reproducibility kaydı

Bir sonuç raporlanırken en az şu bilgiler saklanmalıdır:

- compiler ve version (`gfortran --version`);
- `FFLAGS`, `OMP_NUM_THREADS`, CPU/OS;
- symbol ID/seed seti;
- bundle tie seedleri;
- rule/frame cardinalitesi;
- cleanup modunun `exact_search` değeri;
- Hamming distance, similarity, sözlük büyüklüğü;
- çalışma zamanı için warm-up ve örnek sayısı.

Bu metadata olmadan yalnız “mikrosaniye” değeri veya “%X accuracy” değeri
tekrar üretilebilir performans/doğruluk kanıtı değildir.

## Sürüm kabul checklist'i

Bir değişiklik merge edilmeden önce:

- [ ] `python3 tools/verify_reference.py`
- [ ] `make debug`
- [ ] `make run` (tüm executable assertion'ları PASS)
- [ ] `cmake` + `ctest` checked build
- [ ] yeni public API için normal ve hata yolu testi
- [ ] kaynakta signed-overflow'a dayanan yeni integer aritmetiği yok
- [ ] exact ve approximate cleanup sözleşmeleri karıştırılmadı
- [ ] benchmark iddiası hedef platformda ölçüm çıktısıyla desteklendi
