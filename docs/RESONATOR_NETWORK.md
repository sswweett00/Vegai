# Resonator Network: çoklu-faktör VSA çözümleme

`src/resonator_network.f90`, paketli binary/bipolar Vegai hipervektörlerinde
çoklu-faktör ayrıştırma için Fortran 2023 hedefli bir **discrete Resonator
Network** uygular. Modül, yalnızca bir composite'a yakın olan bir öğeyi cleanup
etmekten farklı olarak, her factor slot için ayrı candidate vocabulary ile

```text
T = F₁ XOR F₂ XOR ... XOR Fₙ
```

denklemindeki bilinmeyen `F₁..Fₙ` tuple'ını arar.

Bu doküman iki şeyi açıkça ayırır:

1. **Resonator phase:** bounded, multi-restart iteratif candidate refinement;
2. **Exact verifier:** küçük/orta codebook ürünlerinde meet-in-the-middle ile
   bit-tam certificate.

Bu ayrım, heuristic bir sabit noktanın yanlışlıkla kanıtlanmış factorization
gibi sunulmasını engeller.

## 1. Girdi modeli

Her factor için bir `hds_memory_t` codebook gerekir. Codebook'lar slot
spesifiktir; aynı ID farklı slotlarda kullanılabilir, fakat bir codebook içinde
`mem_register_symbol` duplicate ID kabul etmez.

```fortran
use hyper_types
use symbolic_memory
use resonator_network

type(hds_memory_t) :: books(3)
type(resonator_config_t) :: config
type(resonator_result_t) :: answer
type(hv_t) :: a, b, c, target, scratch
integer(int32) :: status

call mem_init(books(1), 4_int32, status)
call mem_init(books(2), 4_int32, status)
call mem_init(books(3), 4_int32, status)

call mem_register_symbol(books(1), 101_int32, 'A', a, status)
call mem_register_symbol(books(2), 202_int32, 'B', b, status)
call mem_register_symbol(books(3), 303_int32, 'C', c, status)

target = hv_bind(hv_bind(a, b), c)
config%max_iterations = 32_int32
config%restarts = 8_int32
config%max_exact_combinations = 65536_int32
config%enable_exact_verifier = .true.
call resonator_factorize(target, books, config, answer, status)

! status == RN_OK means one certified solution.
! status == RN_WARN_AMBIGUOUS means a valid, but non-unique, solution.
```

`target` vektörünün factor'ları gerçekten ilgili slot codebook'larında olmalıdır.
Aynı HVS birden fazla candidate'a atanabilir; bu durumda problem doğası gereği
ambiguity içerir.

## 2. Discrete Resonator update

Bir restart state'i her slot için bir codebook index'i tutar. Iterasyon `t` için:

```text
P(t)   = F₁(t) XOR F₂(t) XOR ... XOR Fₙ(t)
R_i(t) = T XOR P(t) XOR F_i(t) = T XOR (XOR over j != i of F_j(t))
F_i(t+1) = argmin_{c in Codebook_i} Hamming(c, R_i(t))
```

İkinci eşitlik XOR'un self-inverse olmasından gelir. Uygulama `P(t)`yi bir kez
tutar: synchronous turda yalnız state değişirse yeniden compose eder; asynchronous
(Gauss–Seidel) turda ise `P ← P XOR old_F_i XOR new_F_i` ile günceller. Bu
hesaplama eski residual denklemiyle semantik olarak aynıdır, fakat her factor
update'inde diğer bütün factor'ları tekrar bind etmez.

`mem_cleanup` full exact nearest-neighbour olduğundan her slot update'i kendi
codebook'u için deterministiktir. Eşit Hamming mesafesinde en düşük kayıt slotu
seçilir. İki schedule vardır:

| `synchronous_updates` | Davranış |
|---|---|
| `.true.` (varsayılan) | Her factor update'i önceki tam state'ten hesaplanır. Klasik resonator katmanı gibi paralel düşünülmelidir. |
| `.false.` | Slotlar sırayla güncellenir; sonraki slot daha önceki yeni tahminleri görür. Gauss–Seidel benzeri davranış verir. |

Başlangıçlar `initialization_seed` ve restart/factor indeksinden, signed
overflow olmadan, `hv_from_seed` aracılığıyla türetilir. Bu nedenle aynı input,
config ve codebook düzeni bit-tam tekrar üretilebilir.

### Neden tek başına garanti değildir?

Rastgele XOR codebook'larda yanlış factor tuple'ları da çoğunlukla birbirine
ortogonaldir. Bu, coordinate cleanup'in yerel sabit nokta veya cycle'a düşmesine
neden olabilir. Özellikle factor sayısı ve candidate sayısı arttığında yalnızca
`network_converged=.true.` veya `network_best_distance==0` sonucu kesin bir
başarının işaretidir. Pozitif similarity ya da düşük görünen residual tek
başına semantic kanıt değildir.

Modül her restart'ta en düşük residual tuple'ı saklar; çözüm bulunamadığında bu
candidate diagnostic amaçlı sonuçta kalır, ancak `exact=.false.` olur.

## 3. Bounded exact verifier

Varsayılan `enable_exact_verifier=.true.` ayarında Resonator fazı sonrasında
certificate çalışır. Factor dizisi iki gruba ayrılır:

```text
L = F₁ XOR ... XOR F_k
R = F_(k+1) XOR ... XOR Fₙ
T = L XOR R
```

Sağ yarının bütün sınır-içi composite'leri şöyle indexlenir:

```text
fingerprint(R) -> [right factor slot tuple(s)]
```

Her sol composite için istenen sağ değer `T XOR L` hesaplanır. Fingerprint hit
olursa modül **mutlaka** ilgili sağ tuple'ı yeniden bind eder ve 16.384-bit
`hv_equal` kontrolü yapar. Böylece 64-bit fingerprint collision sadece ek bir
karşılaştırma maliyeti yaratır; yanlış çözüm üretemez.

`max_exact_combinations`, **her yarı** için izin verilen maksimum kombinasyon
sayısıdır. Örnekler:

| Factor cardinaliteleri | Split | Sol / sağ combination | `max_exact_combinations` için minimum |
|---|---:|---:|---:|
| `4 × 4 × 4` | `1 + 2` | `4 / 16` | `16` |
| `8 × 8 × 8 × 8` | `2 + 2` | `64 / 64` | `64` |
| `16 × 16 × 16 × 16 × 16` | `2 + 3` | `256 / 4096` | `4096` |

Limit aşılırsa verifier allocation veya gizli approximate sonuç üretmez:
`RN_ERR_SEARCH_LIMIT` döner. Resonator fazı zaten exact bir tuple bulmuşsa
sonuç yine valid `exact=.true.` kalabilir, ama `search_limit_reached=.true.`
ile non-exhaustive verifier bilgisi korunur.

Ambiguity tespiti maliyeti sınırlamak için solution sayısını `0`, `1` veya
`2` ("iki ya da daha fazla") olarak saklar. `RN_WARN_AMBIGUOUS` negatif hata
olmadığından kullanılabilir bir sonucu ifade eder; çağıran uygulama semantik
tie-break politikasını seçmelidir.

## 4. Sonuç sözleşmesi

`resonator_result_t` alanları:

| Alan | Anlam |
|---|---|
| `factors`, `factor_ids`, `factor_labels`, `factor_slots` | Seçilen tuple'ın her factorü |
| `network_best_distance` | Sadece iterative network'ün en iyi Hamming residual'ı |
| `network_converged` | Network doğrudan residual 0'a ulaştı mı? |
| `hamming_distance`, `similarity`, `exact` | Nihai sonuç; verifier başarılıysa certificate sonrası değer |
| `used_exact_verifier` | Bounded MITM verifier exact tuple buldu mu? |
| `exact_verifier_attempted`, `search_limit_reached` | Verification coverage bilgisi |
| `ambiguous`, `exact_solution_count` | Non-uniqueness bilgisi (`2` = iki veya daha fazla) |
| `iterations`, `restarts_attempted` | Resonator fazının çalışma hesabı |

Her çağrı önceki `result` allocatable alanlarını güvenli biçimde temizler.
İşiniz bittiğinde açıkça `resonator_result_destroy(result)` çağırabilirsiniz;
bu call idempotent'tir.

## 5. Durum kodları

| Kod | Anlam |
|---|---|
| `RN_OK` | Bir exact, unique (verifier kapsamındaysa) factorization |
| `RN_WARN_AMBIGUOUS` | En az iki exact tuple; ilk deterministik tuple döndü |
| `RN_ERR_INVALID_ARGUMENT` | Factor sayısı `<2`, geçersiz iteration/restart/limit |
| `RN_ERR_CODEBOOK` | Uninitialized, boş veya kapasitesiyle tutarsız codebook |
| `RN_ERR_MEMORY` | Resonator cleanup çağrısı başarısız oldu |
| `RN_ERR_UNRESOLVED` | Network ve çalıştırılan verifier çözüm bulamadı |
| `RN_ERR_SEARCH_LIMIT` | Network çözmedi, verifier half-combination sınırına takıldı |
| `RN_ERR_ALLOCATION` | Bounded verifier workspace allocation başarısız oldu |

Negatif olmayan `RN_OK` ve `RN_WARN_AMBIGUOUS` kullanılabilir sonuçlardır.

## 6. Maliyet ve bellek bütçesi

`m_i` factor `i`nin candidate sayısı, `N` factor sayısı, `D=16384` olsun.

- Tutulan composite ile synchronous veya asynchronous bir resonator turu
  yaklaşık `O(N × D/64 + Σ_i m_i × D/64)` bit işlemi/cleanup çalışmasıdır.
  Eski her-factor yeniden-compose yolu `O(N² × D/64)` ek bind maliyeti
  taşıyordu. Exact `mem_cleanup` scan'i büyük codebook'larda hâlâ ikinci terimi
  baskın yapar; bu değişiklik certificate veya cleanup semantiğini değiştirmez.
- En çok `restarts × max_iterations` tur çalışır.
- Verifier sağ yarı için `R` tuple tutar: yaklaşık
  `O(R × (8 + 4 + 4×N_right))` byte index workspace; full 2 KiB product
  saklamaz.
- Her fingerprint candidate'ı tam recomposition ve `hv_equal` ile kontrol
  edildiğinden low-memory index doğruluktan ödün vermez.

Üretimde config limitlerini memory budget'a göre belirleyin. Çok büyük
codebook ürünlerinde verifier'ı kapatmak mümkün olsa da, bu durumda result'ın
heuristic olduğunu uygulama seviyesinde açıkça kaydetmek gerekir.

## 7. Test edilen senaryolar

Native `test/test_all.f90` aşağıdakileri kapsar:

1. üç adet dört-candidate codebook'tan exact `A₂ XOR B₃ XOR C₄` çözümü;
2. returned factor'ların tekrar bind edilip target'a bit-bit eşitliği;
3. invalid exact-limit ve uninitialized codebook hata yolları;
4. aynı vectorün bir slotta iki farklı ID ile kaydedildiği ambiguity;
5. result workspace lifecycle;
6. retained-composite residual identity'nin hem synchronous hem async yolu;
   returned factor'lar target'a yeniden bind edilerek bağımsız bit oracle ile
   kontrol edilir.

`app/main.f90` da üç farklı semantic codebook üzerinden 3-factor demo
çalıştırır. CI, Fortran 2023 strict mode, runtime checks, OpenMP ve executable
assertion'ları ile bunu doğrular.
