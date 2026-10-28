# Aşırı Kapasiteli Seyrek Uzman Katmanı (Fortran 2023)

Bu belge `src/sparse_expert_layer.f90` modülünün tasarımını, karmaşıklık kanıtını,
API sözleşmesini ve ölçüm yöntemini tanımlar. Ölçülen sayılar
`app/sparse_layer_bench.f90` tarafından üretilir ve CI'da her push'ta yeniden
doğrulanır; bu dosyada elle yazılmış hiçbir süre yoktur.

## 1. Çözülen problem

Yoğun bir katmanın parametre sayısı `W` ise, adım maliyeti de `O(W)`'dir:
`O(W)` zaman, `O(W)` bellek trafiği, `O(W)` FLOP. Kapasite ile adım maliyeti aynı
sayıdır. Modül bu iki niceliği birbirinden ayırır:

```
W_total          = expert_capacity * (input_width * output_width + output_width)
active params    = k * (input_width * output_width + output_width)      # k = active_experts
active FLOPs     = O(k)                                                 # kapasiteden bağımsız
```

`expert_capacity` 947.000 ve uzman şekli `32x32` olduğunda `W_total = 1.000.032.000`
olur; tek bir adımda dokunulan parametre sayısı ise `8 * 1056 = 8448` olarak kalır.

## 2. Kapasitenin milyarlara çıkabilmesinin üç nedeni

**2.1 Yönlendirme sıralama değil, yerellik-duyarlı hash'tir.**
Girdi, dondurulmuş rastgele hiperdüzlem ailesiyle `router_bits` bitlik bir imzaya
dönüştürülür (`O(router_bits * input_width)`, sabit). İmzanın düşük `bucket_bits`
biti kova; kova da karıştırıcıdan geçirilip `[0, expert_capacity)` aralığına
eşlenir. Bu bir **doğrudan adres hesabıdır**: `expert_capacity` üzerinde hiçbir
aday listesi kurulmaz, sıralanmaz, okunmaz. `k` adet sonda (probe) birbirinden
bağımsız hiperdüzlem kümeleri kullandığı için `k` farklı uzmana gider.

Bir uzmanı `W_total` aday arasından *seçmenin* maliyeti kaçınılmaz olarak
`O(W_total)`'dir; hash, bu ödemeyi gereksiz kılan mekanizmadır. Benzer girdiler
aynı düşük bitleri yüksek olasılıkla paylaştığı için eşleme yerellik duyarlıdır.

**2.2 Uzmanlar dokunulana kadar sanaldır.**
`W_total` boyutunda hiçbir dizi yoktur. Bir uzmanın ilk ağırlıkları
`splitmix64(seed, expert_id)` akışından **saf fonksiyon** olarak üretilir; hiç
dokunulmayan uzman sıfır bayt, sıfır zaman maliyetindedir. Dokunulan uzmanlar
append-only sayfa deposunda tutulur ve açık adreslemeli (linear probing, yük
faktörü ≤ 1/2) bir indeksle `O(1)` beklenen sürede bulunur. İndeks boyutu
kapasiteye değil **sayfa sayısına** göredir.

**2.3 Gradyan ve optimize edici durumu da seyrektir.**
Her sayfa kendi `weight_gradient`, `bias_gradient`, `velocity` ve sütun başına
`energy` dizisini yalnızca gerektiğinde tahsis eder. Bu yüzden optimize edici
belleği de adım başına `O(k)`'dir.

## 3. Adım başına `O(k)` kanıtı

| Aşama | Dokunulan veri | Karmaşıklık |
|---|---|---|
| İmza | `router_bits * input_width` hiperdüzlem bileşeni | `O(bits*in)` — sabit |
| Uzman kimliği | `splitmix64` (birkaç ALU işlemi) + `modulo` | `O(1)` |
| Tekilleştirme | en fazla `k` kimlik üzerinde doğrusal arama | `O(k²)`, `k ≤ 64` |
| Sayfa araması | açık adresleme, yük ≤ 1/2 | beklenen `O(1)` |
| Sayfa oluşturma | yalnızca ilk dokunuşta; büyüme geometrik | amortize `O(1)` |
| İleri | `k` × `(in*out)` MAC | `O(k·in·out)` |
| Geri | `k` × (`dW` + `dx` + `db`) | `O(k·in·out)` |
| Güncelleme | yalnızca **kirli liste** (en fazla `k` sayfa) | `O(k·in·out)` |

Kirli liste (`dirty_slots`) tasarımın kritik parçasıdır: naif bir uygulama
optimize edici geçişinde tüm sayfa deposunu tarar, bu da uzun süren bir eğitimde
soğuk sayfalar biriktikçe adım süresini **yavaşça artırır** ve `O(k)` iddiasını
sessizce bozar. Bu modülde geri yayılım dokunduğu sayfaları listeye yazar,
güncelleme yalnızca bu listeyi dolaşır ve listeyi temizler. `resident_bytes`
hesabı da kapalı formdadır (sayfa sayısı × sabit sayfa boyutu), böylece her ekleme
`O(pages)` iş yapmaz.

## 4. Neden öğrenilebilir bir router yok

Tüm uzmanlar üzerinde softmax yapan öğrenilebilir bir router tanımı gereği
`O(W_total)`'dir ve bu modülün ortadan kaldırdığı darboğazı geri getirir. Modül bu
yüzden **dondurulmuş** (eğitimsiz) LSH yönlendirmesi kullanır. `O(k)` özelliğini
koruyan öğrenilebilir bir genişleme mümkündür — *hash-then-score*: kova seçimi
hash ile `O(1)` yapılır, skor yalnızca seçilen `k` uzman için öğrenilir — ancak bu
kodda iddia edilmez, belgelenmiş bir uzantı olarak bırakılır.

`iso_c_binding`, BLAS, LAPACK, cuBLAS ve OpenMP yönergesi kullanılmaz. Sıcak
çekirdekler `do concurrent` ile yazılmıştır; OpenMP açıkken derleyici bunları
iş parçacıklarına eşleyebilir, kapalıyken yasal birer seri döngüdür.

`simple` (Fortran 2023) anahtar sözcüğü **bilinçli olarak kullanılmamıştır**:
güncel gfortran/flang sürümleri bu anahtar sözcüğü reddeder ve "derlenmeye hazır"
bir teslimi derlenemez hale getirir. Aynı yan-etkisizlik sözleşmesini taşıyan
`pure` kullanılır.

## 5. API sözleşmesi

```fortran
use sparse_expert_layer
type(sparse_layer_config_t) :: config      ! in, out, expert_capacity, active_experts,
                                           ! router_bits, bucket_bits, seed, NOVA sabitleri
type(sparse_layer_t)        :: layer
type(sparse_forward_t)      :: cache
integer(int32)              :: status

call sparse_layer_init(layer, config, status)              ! router + boş sayfa deposu
call sparse_layer_forward(layer, input, output, cache, status)   ! O(k) yönlendirme + ileri
call sparse_layer_backward(layer, cache, grad_output, grad_input, status)  ! O(k) geri
call sparse_layer_apply_update(layer, status)              ! yalnızca kirli sayfalar
call sparse_layer_destroy(layer)
```

Sayaçlar (`layer%stats`): `pages_allocated`, `resident_bytes`, `active_parameters`,
`active_flops`, `cumulative_*`, `updates_applied`, `router_evaluations`.
`sparse_layer_parameter_count` kapasiteyi (`O(W_total)` bir *sayı*, asla bir
tahsis değil), `sparse_layer_active_parameter_count` adım başına dokunulan
parametre sayısını verir.

Durum kodları kapalı-başarısız çalışır: `SE_OK`, `SE_ERR_INVALID_ARGUMENT`,
`SE_ERR_ALLOCATION`, `SE_ERR_NUMERICAL`, `SE_ERR_ORDER`.

**Yeniden üretilebilirlik:** bir uzmanın ilk ağırlıkları `(seed, expert_id)`
fonksiyonudur. Dolayısıyla kontrol noktası parametreleri değil, `seed` ile
dokunulmuş kimlikleri saklamak yeterlidir; aynı tohum iki bağımsız katmanda
bit-bit aynı sonucu verir (sözleşme testi bunu doğrular).

## 6. Benchmark: yöntem ve ölçütler

`app/sparse_layer_bench.f90` üç noktayı ölçer:

| Nokta | Uzman | k | Kapasite | `W_total` |
|---|---|---|---|---|
| A | 16×16 | 4 | 4 | 1.088 |
| B | 32×32 | 8 | 977 | 1.031.712 |
| C | 32×32 | 8 | 947.000 | 1.000.032.000 |

B ve C aynı uzman şeklini ve aynı `k`'yi paylaşır: aralarındaki tek fark
kapasitedir. Her nokta 32 adım ısınma + 256 ölçülen adım koşar; adım
= ileri + geri + güncelleme, `system_clock` ile ölçülür. İki mod vardır:

* **steady** — her adımda aynı girdi; ısınmadan sonra tüm sayfalar yerleşik, sayı
  saf aritmetiktir.
* **streaming** — her adımda yeni deterministik girdi; sayı soğuk uzmanların
  oluşturulma maliyetini de içerir. 1e9 kapasiteli, yalnızca küçük bir dilimi
  yerleşik olan bir katmanın gerçek rejimi budur.

Program üç iddiayı **eşitlik olarak** sınar ve sağlanmazsa sıfırdan farklı çıkış
koduyla biter (dolayısıyla CTest'te bir testtir):

1. B ve C'nin `active_parameters` ve `active_flops` sayaçları birebir eşittir;
2. `pages_allocated ≤ (ısınma + ölçüm + 1) * k` ve `pages_allocated < capacity`;
3. C'nin steady gecikmesi, B'nin 1,5 katı ile 1/1,5 katı arasındadır.

Makine-okunur tek satır (`SPARSE_LAYER_BENCH_JSON: {...}`) CI annotation'ına ve
iş özetine yazılır.

### 6.1 Ölçülen sonuç (CI, `e91826b`, koşu 40669623211)

Ubuntu x86-64, AMD EPYC 7763, **Debug** yapı (`-fcheck=all -ffpe-trap=...`, yani
mutlak süreler iyimser değil kötümserdir; önemli olan oranlardır):

| Nokta | `W_total` | aktif param | aktif FLOPs | sayfa | yerleşik | steady µs/adım | streaming µs/adım |
|---|---|---|---|---|---|---|---|
| A 16×16, k=4, cap 4 | 1.088 | 1.088 | 17.536 | 4 | 0,04 MiB | 42,40 | 42,70 |
| B 32×32, k=8, cap 977 | 1.031.712 | 8.448 | 123.392 | 790 | 19,2 MiB | 289,03 | 364,31 |
| C 32×32, k=8, cap 947.000 | 1.000.032.000 | 8.448 | 123.392 | 1.595 | 38,6 MiB | **290,48** | 456,27 |

Kapasite B → C arasında **970×** büyürken steady adım süresi **1,005×** değişti;
aktif parametre ve aktif FLOP sayaçları son basamağa kadar aynı kaldı; yerleşik
bellek 19,2 MiB → 38,6 MiB oldu, yani kapasiteyle değil **dokunulan sayfa
sayısıyla** ölçeklendi (`1.000.032.000 × 8 B = 7,45 GiB` yerine 38,6 MiB).

Tekrar (koşu 40670901969, aynı kod): B 296,34 µs, C 288,54 µs → oran **0,974×**;
streaming 369,5 → 468,8 µs (1,27×). İki koşulda da steady oranı 1,00 ± 0,03
bandında, yani ölçüm gürültüsü kadar.

Streaming modu 364 → 456 µs (1,25×) çıktı ve bu beklenen davranıştır: streaming
süresi kapasiteyle değil **adım başına yeni uzman oranıyla** ölçeklenir. Aynı
sayıda adımda C, B'den daha fazla soğuk uzmanla karşılaştı (1.595'e karşı 790
sayfa), çünkü 947.000 uzmanlı uzayda aynı girdi akışı daha az tekrar üretir.
Soğuk uzman üretimi uzman başına bir kez ödenen `O(in*out)` işidir; sabit
yönlendirme dağılımına oturmuş bir katmanda bu terim sıfıra gider (steady modu
tam olarak bunu ölçer).

Çalıştırma:

```bash
make sparse-bench      # build/vegai_sparse_layer_bench
make sparse-tests      # sözleşme testleri
# veya
cmake -S . -B build && cmake --build build --parallel && ctest --test-dir build -R sparse
```

## 7. Sözleşme testleri

`test/sparse_layer_tests.f90` (CTest adı `vegai_sparse_layer_contract`):

1. geçersiz yapılandırma reddi (`k > 64`, `k > capacity`, `bucket_bits > router_bits`);
2. aktivasyon başına `k` **farklı** uzman ve aynı girdi için deterministik
   yönlendirme; kapıların toplamı 1;
3. hiç yönlendirilmeyen uzman için sayfa yok; `destroy` sayaçları sıfırlar;
4. güncelleme geçişi sayfa deposunu değil **yalnızca aktif sayfaları** dolaşır
   (`updates_applied` farkı = `active_count`);
5. gradyan yerelliği: yönlendirilmeyen uzmanın ağırlıkları bit-bit değişmez;
6. tekrar determinizmi: aynı tohum → aynı parametreler, farklı tohum → farklı;
7. kapasite değişmezliği: 64 ve 200.000 uzmanlı katmanlarda aktif parametre/FLOP
   sayaçları eşit, sayfa sayıları eşit, bellek kapasiteyle büyümüyor;
8. katman öğreniyor: 200 adımda oyuncak hedefe göre hata azalıyor;
9. sıra ihlali (`forward` olmadan `backward`) reddediliyor.

## 8. Dürüst sınırlar

* **Soğuk başlangıç maliyeti.** Yeni bir uzmanın ağırlıkları hash'ten üretilir;
  bu `O(in*out)`'dur ve uzman başına **bir kez** ödenir. Streaming modu bu
  maliyeti ölçüye dahil eder, steady modu etmez. Kalıcı olarak yeni uzmanlara
  gidilen bir iş yükünde amortize edilemeyen kısım budur.
* **Yönlendirme kalitesi.** Dondurulmuş LSH, öğrenilmiş yönlendirmenin uzman
  uzmanlaşmasını garanti etmez; `bucket_bits` bu dengeyi kurcalayan tek düğmedir.
  Bu modül bir *kapasite/maliyet* mekanizmasıdır, kalite garantisi değildir.
* **Tekilleştirme.** Bir aktivasyonda iki sonda aynı uzmana düşerse kimlik bir
  sonraki kullanılmayan uzmana kaydırılır; `k` farklı uzman garantisi buradan
  gelir, bedeli `O(k²)` aramadır (`k ≤ 64`).
* **Bellek üst sınırı yoktur.** Sayfa deposu append-only'dir; uzun süre çalışan
  bir katmanda dokunulan uzman sayısı büyür. Kapasiteyle sınırlıdır, adımla
  değil; gerçek bir sunucuda bu noktada tahliye (eviction) politikası gerekir ve
  bu modülde yoktur.
* Bu katman mevcut eğitim hattına **bağlanmamıştır**: teslim edilen, tek başına
  derlenen ve ölçülen bir katman + benchmark'tır, uçtan uca model değildir.
