# Sabit Bütçeli Bağlam Motoru (Fixed-Budget Context Engine)

Bu belge `src/fixed_budget_context.f90`, `src/fixed_budget_backward.f90`,
`src/fixed_budget_optimizer.f90`, `src/fixed_budget_training.f90`,
`app/context_engine.f90` ve `test/fixed_budget_context_tests.f90` tarafından
uygulanan motoru tanımlar. Sayısal iddiaların tamamı ya CI'da çalıştırılan
testlerden ya da CI'da ölçülen benchmark çıktısından gelir; ölçülmemiş hiçbir
sonuç burada başarı olarak sunulmaz.

## 1. Sözleşme

`--window` **anlamsal** bir büyüklüktür: yalnızca çok ölçekli özyinelemeli
belleğin bozunma sabitlerini seçer. Hiçbir dizi boyutunda, hiçbir ayırma
ifadesinde, hiçbir belirteç döngü sınırında ve hiçbir dikkat tarama uzunluğunda
görünmez. Fiziksel boyutların tamamı ayrı düğmelerdir:

| Düğme | Rol | Varsayılan | Sınır |
| --- | --- | --- | --- |
| `--window` | mantıksal ufuk (int64) | 32 | ≥ 1, taşmada reddedilir |
| `--local-window` | halka tampon (fiziksel) | 32 | 1..4096 |
| `--memory-scales` | sabit bozunma ölçeği | 4 | 1..16 |
| `--memory-slots` | sabit bellek yuvası | 64 | 1..1024 |
| `--train-chunk-tokens` | güncelleme başına tahmin | 256 | 1..8192 |

`--window` değeri boş, işaretli (`+`/`-`), sayı olmayan, sıfır, negatif veya
int64 taşması içeren biçimlerde **reddedilir**; sessiz sarma yoktur. Aritmetik
olarak temsil edilemeyen ufuklar (`> 4e15`) kırpılır ve bu durum hem planda hem
kontrol noktası başlığında `horizon_clamped` alanıyla **bildirilir**.

## 2. Üç bağlam kanalı

1. **Yerel (fiziksel) kanal.** `local_window` uzunluğunda halka tampon üzerinde
   nedensel dikkat. Kopyalama yok (halka indisi `modulo(cache_head - ordinal,
   window) + 1`), yeniden ayırma yok, gelecek sızıntısı yok: test bunu son
   belirteci değiştirip önceki konumların logitlerinin bit düzeyinde aynı
   kaldığını doğrulayarak kanıtlar.
2. **Çok ölçekli özyinelemeli bellek.** Ölçek başına
   `M_t^(j) = λ_j M_(t-1)^(j) + (1-λ_j) φ(k_t) v_tᵀ` ve
   `z_t^(j) = λ_j z_(t-1)^(j) + (1-λ_j) φ(k_t)`, okuma
   `φ(q)·M^(j) / (φ(q)·z^(j) + ε)`. `λ_j = exp(-1/τ_j)`, `τ_j` mantıksal ufuk
   ile fiziksel pencere arasında logaritmik olarak eşit aralıklı, **başlatmada
   bir kez** hesaplanır; belirteç başına `exp` yoktur. Durum boyutu yalnızca
   `heads × scales × (dh × dh + dh)`'dir.
3. **Yuva (slot) kanalı.** `memory_slots` adet kapasite; her belirteç O(1)
   içinde tüm yuvalara `g_m = σ((k·K_m)/√d / T)` geçidiyle yazar, önem
   birikimini `s_m ← ρ s_m + (1-ρ) g_m` ile günceller, sonra softmax ile okur.
   Tam geçmiş taraması yoktur.

Kanallar üç öğrenilebilir lojistik üzerinden softmax ile karıştırılır; bu
karışım da eğitilir.

**Dürüst ufuk raporu.** "Etkin ufuk" bir bozunma erişimidir, kayıpsız depolama
değildir. Özyinelemeli bellek üstel bir ortalamadır: uzak belirteçler unutulur,
yuvalar çarpışır ve yanlış hatırlama üretebilir. 1e12 ufuk "1e12 belirteç
saklanıyor" demek değildir; "en yavaş ölçeğin bozunma sabiti 1e12 belirteçlik
bir zaman sabitine karşılık gelir" demektir.

## 3. Değişmez maliyet (yapısal sayaçlar)

`fbc_plan` ve `context_engine plan` şu sayıları üretir ve test bunları dört
farklı ufukta (32, 8192, 1e6, 1e12) **birebir eşit** olarak doğrular:

- parametre sayısı,
- akış durumu baytı,
- yerel önbellek / özyinelemeli durum / yuva durumu eleman sayısı,
- belirteç başına ileri ve geri işlem sayısı,
- belirteç başına yerel skor işlemi sayısı.

CI'da ölçülen yapı (genişlik 32, 2 kafa, ffn 64, sözlük 32, yerel 32, 4 ölçek,
64 yuva): **15 497 parametre**, **67 072 bayt akış durumu**, belirteç başına
**43 008** ileri işlem — her ufukta aynı.

## 4. Eğitim: gerçek gradyanlar

`fbc_step_backward` tek bir adımın uygulanmış hesabının **tam türevidir**
(derinlik-1 kesikli BPTT). Kapsam:

- bağlı gömme ve çıkış yanlılığı,
- dikkat q/k/v projeksiyonları, dikkat çıkış projeksiyonu,
- FFN kapı/değer/çıkış projeksiyonları,
- her iki RMSNorm ölçeği,
- çok ölçekli okuma ağırlıkları (`scale_logit`), kanal karışımı,
- dört yuva projeksiyonu ve yuva önem ağırlığı,
- mesafe yanlılığı.

İki **aynı adım içi yazma** tam olarak türevlenir, çünkü okuma aynı adımda
onları kullanır: yuva yazması (anahtar, değer, geçit, önem) ve özyinelemeli
güncelleme (`φ(k)`, `v`). Önceki belirteçlerin yazmaları dondurulmuştur; bu
bilinçli ve belgelenmiş bir yaklaşımdır, "tam BPTT" olarak sunulmaz.

Doğrulama: sonlu fark sözleşme testi 11 parametreyi (okuma ağırlığı, kanal
lojistiği, iki yuva yazma projeksiyonu, yuva sorgu projeksiyonu, mesafe
yanlılığı, FFN kapısı, sorgu, anahtar, değer, bağlı gömme) merkezi farkla
karşılaştırır ve `1e-4` bağıl toleransla geçer.

## 5. Optimize edici

NOVA-1 biçimli, sütun başına tek hız + tek enerji:

```
e_c   ← ρ e_c + (1-ρ) mean(g_c²)
a     ← <g_c, v_c>                       (anlaşma sondası)
γ     ← momentum  (a < 0 ise momentum × restart_damp)
v_c   ← γ v_c + (1-γ) g_c                (momentum EMA)
p_c   ← (1 - lr·wd) p_c - lr · v_c / (√e_c + ε)
```

Momentum bir **EMA**'dır, koşan toplam değildir: toplam, enerji
normalizasyonundan önce yer değiştirmeyi `1/(1-γ) = 10×` büyütür ve öğrenme
oranının söylediğinin on katı adım atardı (bu, sözleşme testinde eğitimin
ıraksamasına yol açtı ve düzeltildi). İkinci moment yoktur, eleman başına
transandantal yoktur.

## 6. Kontrol noktası

`VGFB0001` sürüm 1. İçerik: sözlük/genişlik/kafa/ffn/yerel pencere/ölçek/yuva,
**mantıksal ufuk**, etkin ufuk ve kırpma bayrağı, RMS epsilon, yuva bozunması ve
sıcaklığı, güncelleme ve belirteç sayaçları, optimize edici hiper-parametreleri,
tüm parametre yükü ve tüm optimize edici durumu.

Eski `VGNR0001` (neural_pretrain v5) kontrol noktaları **dönüştürülmez**:
`FBT_ERR_LEGACY` (−1503) ile reddedilir ve eski ikili dosyayla açılması
gerektiği söylenir. Test bunu hem başlık okuma hem yükleme yolunda doğrular.
Kaydet→yükle sonrası logitler bit düzeyinde aynıdır.

## 7. Testler ve komutlar

```
make context-tests                     # sözleşme paketi
make context-engine                    # CLI
cmake -S . -B build -DVEGAI_BUILD_TESTS=ON && cmake --build build
ctest --test-dir build -R vegai_context
```

Sözleşme grupları: seçenek ayrıştırma; ufuk değişmezliği (yapısal sayaçlar);
gelecek sızıntısı yok; parça/akış eşdeğerliği; belirlenimcilik; uzun akışta
sonluluk ve sınırlı durum; sonlu fark gradyan kontrolü; eğitimde kayıp düşüşü;
her bellek bileşeninin gradyan alması ve değişmesi; 93 belirteç uzaklıktaki
cevabın sabit boyutlu durum üzerinden sorgu logitlerine ulaşması (ve bellek
kanalları kapalıyken ulaşamaması); kontrol noktası gidiş-dönüşü ve eski
biçimin reddi.

Kullanım:

```
./build/vegai_context_engine plan --window 1000000000000
./build/vegai_context_engine sweep --tokens 256 --train-chunk-tokens 32
./build/vegai_context_engine train --tokens 4096 --checkpoint m.vgfb
./build/vegai_context_engine inspect --checkpoint m.vgfb
./build/vegai_context_engine evaluate --tokens 2048 --checkpoint m.vgfb
./build/vegai_context_engine bench --window 1000000000000
```

## 8. Ölçülen sonuçlar (CI, `VEGAI` "Fortran verification")

Çalıştırma: `vegai_context_engine sweep`, genişlik 32 / 2 kafa / ffn 64 /
sözlük 32 / yerel 32 / 4 ölçek / 64 yuva, 256 belirteç, Debug derleme
(`-fcheck=all -ffpe-trap=invalid,zero,overflow`), koşu 40681155794.
Fiziksel yapı dört satırda da aynı; yalnızca `--window` değişiyor.

| window | parametre | durum baytı | ileri işlem/tok | p50 µs | p95 µs | çıkarım tok/s | eğitim tok/s | eğitim NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 15 497 | 67 072 | 43 008 | 49.211 | 59.701 | 19 908 | 6 615 | 2.6914 |
| 8 192 | 15 497 | 67 072 | 43 008 | 49.613 | 59.551 | 19 850 | 6 753 | 2.7077 |
| 1e6 | 15 497 | 67 072 | 43 008 | 49.652 | 59.641 | 19 852 | 6 713 | 2.7083 |
| 1e12 | 15 497 | 67 072 | 43 008 | 49.612 | 59.060 | 19 948 | 6 810 | 2.6848 |

`window` 32 → 1e12 (3.1e10 kat) için p50 oranı **1.008×**, çıkarım hızı oranı
**1.002×**: ölçüm gürültüsü içinde. Mutlak süreler Debug derlemesine aittir;
Release sayısı burada iddia edilmez. Aynı dört satır CI iş özetinde ve
`CONTEXT_BENCH_JSON` satırlarında da yayımlanır.

## 9. Sınırlar (iddia edilmeyenler)

- Özyinelemeli bellek kayıpsızdır denmez: üstel unutma, yuva çarpışması ve
  yanlış hatırlama vardır ve ölçülebilir.
- Etkin ufuk bozunma tabanlıdır; 4e15 üstü aritmetik olarak kırpılır ve
  bildirilir.
- Geri yayılım derinlik-1 kesiklidir; önceki belirteçlerin anahtar/değer
  gradyanları yoktur.
- Yeni modüllerde OpenMP yönergesi yoktur; mevcut isteğe bağlı OpenMP yolu bu
  motor için kullanılmadı ve bu yüzden burada hızlanma iddiası da yoktur.
- Benchmark süreleri Debug yapısında ölçülür; mutlak değil göreli kanıt olarak
  okunmalıdır.
