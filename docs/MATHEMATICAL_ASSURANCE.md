# Vegai matematiksel güvence ve performans sınırları

Bu belge, Vegai'nin bit-seviyesi HDC/VSA işlemlerinde hangi matematiksel
özelliklerin kod ve test tarafından denetlendiğini açıklar. Bir öğrenilmiş
checkpoint'in dilsel doğruluğunu, gerçek-dünya bilgisini veya herhangi bir
modelin evrensel üstünlüğünü kanıtlamaz. Bu iddialar için ayrı, versioned
held-out evaluation gerekir.

## 1. Kapsam ve kanıt türleri

| Alan | Kod sözleşmesi | Kanıt türü | Kapsam dışı |
|---|---|---|---|
| Packed HDC cebiri | `D = 16384`, XOR binding, cyclic permutation, Hamming | native property testleri + bağımsız Python reference | learned semantics veya task accuracy |
| Exact-rate mask | seçilen dimension sayısı tamdır; yüksek yoğunlukta complement sampling | cardinality/determinism regression | SplitMix'in kriptografik rastgeleliği |
| Cleanup | global minimum `(distance, slot)` | exhaustive scan ve tie regression | LSH-only API'nin nearest-neighbour garantisi |
| Resonator | retained XOR composite residual denklemi | rebind oracle + bounded exact MITM certificate | heuristic phase'in tek başına başarı kanıtı |
| Walsh field | integer transform/inverse/Parseval | exact integer regression | field'in arbitrary input için lossless/semantic olması |
| Release/optimizer istatistiği | bounded Hoeffding/Wilson hesapları | strict parser + synthetic negative/edge tests | independence, calibration veya held-out suite kalitesi |

## 2. HDC cebiri ve sayısal temsil

Bir packed hypervector `256 × int64 = 16,384` bittir. `hv_bind(A,B)` kelime
bazında XOR'dur; dolayısıyla self-inverse unbinding eşitliği

```text
A XOR (A XOR B) = B
```

bit-tam geçerlidir. `hv_permute` tek bir `D`-bit halkayı kullanır; word-local
rotation değildir. Property testleri negatif shift ve `D + k` normalization
invariantlarını kontrol eder.

Hamming distance için

```text
H(A,B) = sum(popcnt(A_word XOR B_word))
sim(A,B) = 1 - 2 H(A,B) / 16384
```

kullanılır. `hv_similarity_from_distance` bütün Fortran report yüzeylerinde
canonical dönüşümdür. Geçerli `H` aralığı `0..16384` olduğunda sonuç
`1/8192` adımlarındadır ve power-of-two denominator nedeniyle `real32` içinde
tam temsil edilir. Bu helper'a aralık dışı değer verilmesi API önkoşul ihlalidir;
normal production path değeri yalnız `hv_hamming`den alır.

Accumulator update'leri, `int16` counter sınırını önce tüm dimensionlarda
kontrol eder; ardından commit eder. Böylece counter overflow bir hata üretir ve
kısmi accumulator mutation bırakmaz. Bit-sliced accumulator ile legacy signed
counter sonucu property testlerinde karşılaştırılır.

## 3. Exact subset sampling ve Hebbian update

`hv_exact_selection_mask(k, seed, ...)` için yeni lineage aşağıdaki yöntemi
kullanır:

1. `m = min(k, D-k)` seçilir.
2. Floyd algorithm `m` farklı pozisyon örnekler.
3. `k > D/2` ise örneklenen dışlanan kümenin complement'i alınır.
4. Her bounded draw, `2^31` domaininde rejection ile modulo bias olmadan
   normalize edilir.

Böylece mask cardinality tam `k` olur; çalışma, özellikle `%50` üzerindeki
oranlarda, eski coupon-collector retry davranışına dayanmaz. Determinism,
seed'e bağlıdır. Bu PRNG kullanımının uniformity'si ve determinism'i bir
kripto-security guarantee değildir.

`hv_hebbian_update_percent` selected dimensions üzerinde prototype bitini
context bitiyle değiştirir. Seçilen sayı tam olsa da context ile prototype aynı
bit taşıyorsa observable Hamming change daha düşük olabilir; testler bu iki
kavramı ayırır.

### Checkpoint lineage

Mask sequence'i öğrenme trajectory'sini etkiler. Bu nedenle v7 checkpoint
version'u Floyd/complement sampler'ını bağlar. v4-v6 yüklenince historical
bounded-retry selection path'i seçilir ve bu lineage tekrar v5/v6 ile yazılır.
Böylece legacy checkpoint resume'i sessizce yeni sampler'a geçmez. Bu format
seçimi wire-layout migration değildir; future update semantics için explicit
compatibility boundary'dir. Ayrıntı: [PRETRAINING.md](PRETRAINING.md).

## 4. Exact cleanup ve resonator maliyeti

`mem_cleanup` LSH'nin candidate sonuçlarını global exact result olarak kabul
etmez. LSH yalnız distance-zero short path'tir; genel durumda bütün kayıtlar
incelenir. Parallel reduction şu lexicographic key'i minimize eder:

```text
rank(distance, slot) = distance * (count + 1) + slot
```

Burada `slot` 1-based olduğundan rank injective'dir. `int64` üst sınırı,
izinli `int32` capacity ve `D=16384` için güvenlidir. Bu, eski parallel distance
pass + serial tied-slot rescan'ini tek pass ile değiştirir; `(min distance,
lowest slot)` sonucu değişmez.

Resonator factor update için

```text
P = XOR_i F_i
R_i = T XOR P XOR F_i
```

identity'si kullanılır. Synchronous tur `P`yi bütün slot residual'ları için
paylaşır; asynchronous tur bir slot değiştiğinde `P XOR old XOR new` ile
günceller. Böylece recomposition maliyeti bir turda `O(N² × D/64)` yerine
`O(N × D/64)` olur. Cleanup scan maliyeti `sum_i O(m_i × D/64)` olarak kalır ve
büyük codebook'larda baskın olabilir. Returned factor tuple her zaman target'a
bit-tam rebind edilerek kontrol edilir; `enable_exact_verifier` açıksa bounded
MITM ayrıca certificate üretir.

## 5. Float sınırları ve fail-closed policy

Dialogue similarity/margin config'i `ieee_is_finite` ile doğrulanır. NaN veya
infinity threshold'u comparison bypass ederek response ground etmeye yol
vermez; index initialization fail-closed reddeder.

Python EPO ve LMM release governor, Hoeffding union-bound çarpanını `log`domaininde hesaplar. Bu, matematiksel olarak

```text
log((2 * intervals) / delta)
log((2 * arms * metrics * n * (n + 1)) / delta)
```

eşitlikleriyle aynıdır fakat legal positive subnormal `delta` değerinde ara
çarpımın `inf` olmasını önler. Parser yine yalnız finite, positive, bounded
policy değerlerini kabul eder. İstatistiksel bounds, declared observations'ın
independent/bounded olduğu varsayımına dayanır; bu varsayımlar telemetry veya
model quality tarafından otomatik kanıtlanmaz.

## 6. Çalıştırılabilir doğrulama

Native Fortran toolchain bulunan bir hostta:

```bash
make check
# veya
cmake -S . -B build -DVEGAI_ENABLE_OPENMP=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Bu suite exact cardinality, 99/100% complement, similarity endpoints,
nonzero-distance lowest-slot tie, sampler-lineage checkpoint resume,
synchronous/asynchronous resonator path, tokenless Parseval ve dialogue NaN
negative case'lerini kapsar.

Python numerical/control-plane regressions:

```bash
python3 tools/verify_evidential_optimizer.py
python3 tools/verify_lmm_release_governor.py
```

Bu verifiers subnormal confidence-delta edge case'ini de test eder. Perf claim
yapmadan önce hedef donanımda benchmark çalıştırılmalıdır; kaynak analizindeki
`O(...)` maliyetleri wall-clock hızlanma garantisi değildir.
