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
| `--window` | mantıksal ufuk (int64) | **8192** | ≥ 1, taşmada reddedilir |
| `--local-window` | halka tampon (fiziksel) | 32 | 1..4096 |
| `--memory-scales` | sabit bozunma ölçeği | 4 | 1..16 |
| `--memory-slots` | sabit bellek yuvası | 64 | 1..1024 |
| `--train-chunk-tokens` | güncelleme başına tahmin | 256 | 1..8192 |

`--window` varsayılanı **8192**'dir: 32 belirteçlik fiziksel pencerenin 256
katı. Bu, dört varsayılan ölçeğin gerçekten farklı olduğu en kısa ufuktur
(`τ = 32 / 203 / 1290 / 8192`). Ufuk fiziksel pencereye eşit seçilirse
(`--window 32 --local-window 32`) dört ölçek de `λ = exp(-1/32)` değerine çöker;
bu durum sözleşme testinde ayrıca doğrulanır.

`--memory-scales` varsayılanı **4**'te kalır: 8 ölçeğin kaliteyi artırdığı
**ölçülmedi** (bkz. §8.2). Ölçüm, ölçek sayısını artırmanın bu rejimlerde
görünür bir kazanç üretmediğini gösterdi; kanıt olmadan varsayılanı büyütmek
yalnızca durum baytını büyütürdü.

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
karışım da eğitilir. Karışım **girdiye bağımlıdır**: her kanal lojistiği
`channel_logit(c) + channel_gate_proj(c, :) · x` biçimindedir ve ölçek
karışımı için aynısı `scale_gate_proj` ile yapılır; burada `x`, adımın
normalleştirilmiş girdisidir (q/k/v, yuva projeksiyonları ve FFN'in okuduğu
vektör). Böylece model "bu adımda yerel pencereye mi, yoksa özyinelemeli
belleğe mi bakmalıyım" kararını verebilir. Her iki projeksiyon **tam olarak
sıfırdan** başlar; sıfırda davranış eski statik lojistiklerle birebir aynıdır,
yani bu bir yeniden yorumlama değil eklemedir. Maliyet belirteç başına
`(3 + ölçek) × genişlik` çarpım-toplamıdır: ek ayırma yok, kendi `exp`'i yok,
ufka bağlı hiçbir sınır yok. Eğim, bir softmax'ın girdiye göre afin
lojistiklerinde iki yolludur ve ikisi de uygulanır: geçit satırları
`dL/dlojistik · x` alır, paylaşılan girdi ise `Σ dL/dlojistik · W` alır.

**Geçidin ikinci yarısı: bağlam koşullu yönlendirici.** Yalnızca girdiye bakan
geçit "bu *belirteç* yuvalara benzeyen bir belirteç" diyebilir; "yerel pencere
az önce bir tekrar buldu, o yüzden oradan oku" diyemez, çünkü kanalların ne
döndürdüğünü görmez. `channel_router` (`3 × 3·width`) bunu ekler:

```
lojistik_eff(c) = channel_logit(c)
                + channel_gate_proj(c, :) · x
                + Σ_{s,k} channel_router(c, (k−1)·3 + s) · context(k, s)
```

Burada `context(k, s)`, `s` kanalının (yerel, özyinelemeli, yuva) bu belirteç
için döndürdüğü vektörün `k`'ıncı bileşenidir; sütun düzeni
`(bileşen − 1)·3 + kanal`'dır. Satır = çıkış kanalı. Yönlendirici **tam olarak
sıfırdan** başlar, yani başlangıçta davranış girdiye bağımlı geçitle birebir
aynıdır (test bunu bit düzeyinde doğrular: sıfır yönlendirici ile etkin
lojistikler `channel_logit + channel_gate_proj · x`'e eşittir). Maliyet
`9 × genişlik` parametre ve belirteç başına `2 × 9 × genişlik` işlemdir; ek
ayırma yok, kendi `exp`'i yok, ufka bağlı hiçbir sınır yok (sınırlar `width` ve
sabit `3`'tür). Gradyan iki yönlüdür ve ikisi de uygulanır: yönlendirici
sütunları `dL/dlojistik · context` alır, üç bağlam vektörü ise
`dot(dL/dlojistik, yönlendirici sütunu)` kadar **ek** gradyan alır ve bu ek,
kanal geri yayılımının girdisine (yuva okuması, özyinelemeli bellek kafaları,
yerel dikkat) eklenir.

**Yerel kanalın ikinci okuması: indüksiyon (kaydırılmış okuma).** Tek bir
dikkat bloğu, `q_t`'yi yalnızca o anki belirteçten kurar. Bu yüzden "eşleşen
konumun *değerini* oku" (yerinde okuma) yapabilir, ama "eşleşen konumdan
**bir sonra** gelen değeri oku" yapamaz. Oysa `Tekrar et: X. Sonuç: X` ya da
`Say: abc. Sonra: bcd` tipi görevlerin istediği tam olarak budur: `t`
adımındaki bayt daha önce `j` konumunda geçtiyse, cevap `j+1`'deki bayttır.
Klasik indüksiyon kafası bunu iki bloğun bileşimiyle yapar; burada aynı iş
tek taramada yapılır, çünkü halka tamponu `j+1`'in değerini zaten tutar:

```
read_shifted(h) = Σ_{d=1..count} w_{d+1}(h) · value(d - 1)
local_context(:, h) = read_place(h) + induction_mix(h) · read_shifted(h)
```

`w` yerinde okumanın **aynı** softmax ağırlıklarıdır (yeni `exp` yok, yeni
tampon yok), `value(-1)` bu adımın kendi `v`'sidir, `value(d-1)` ise
`d-1` uzaklıktaki önbellek girdisidir. `induction_mix` kafa başına **bir
skalerdir** ve sıfırdan başlar: sıfırda kanal tam olarak yerinde okumadır, yani
bu da eklemedir, yeniden yorumlama değil. Maliyet: kafa başına bir parametre
(4 kafada +4), belirteç başına `count × dh × head` çarpım-toplama — sınır
`local_window`, yani fiziksel sabit; ufuk yok. Gradyan iki yönlüdür ve
paylaşılan ağırlıklar nedeniyle *aynı* `d(w)` girdilerine eklenir:
`d(induction_mix) = <dlocal_context, read_shifted>`, `d(w_{d+1}) += <dind, value(d-1)>`,
`d(v) += w_2 · dind`. Okunan değerler önceki adımların yazdığı donmuş
önbellektir, dolayısıyla derinlik-1 kesikliliği bozulmaz.

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
**43 008** ileri işlem — her ufukta aynı. (Bu sayılar girdiye bağımlı geçitten
önceki koşunun kaydıdır; geçit aynı şekilde parametre sayısını
`(3 + ölçek) × genişlik` = 224 artırır ve belirteç başına ileri işleme
`2 × 224` ekler. Durum baytı değişmez, çünkü geçit öğrenilen bir parametredir,
akan durum değil.)

## 4. Eğitim: gerçek gradyanlar

`fbc_step_backward` tek bir adımın uygulanmış hesabının **tam türevidir**
(derinlik-1 kesikli BPTT). Kapsam:

- bağlı gömme ve çıkış yanlılığı,
- dikkat q/k/v projeksiyonları, dikkat çıkış projeksiyonu,
- FFN kapı/değer/çıkış projeksiyonları,
- her iki RMSNorm ölçeği,
- çok ölçekli okuma ağırlıkları (`scale_logit`) ve girdiye bağımlı ölçek
  geçidi (`scale_gate_proj`),
- kanal karışımı (`channel_logit`), girdiye bağımlı kanal geçidi
  (`channel_gate_proj`) ve bağlam koşullu kanal yönlendiricisi
  (`channel_router`) — hem yönlendirici satırları hem de üç bağlam vektörünün
  yönlendirici üzerinden aldığı ek gradyan,
- yerel kanalın kaydırılmış (indüksiyon) okuması: kafa başına
  `induction_mix` **ve** paylaşılan dikkat ağırlıklarının bu okumadan gelen
  payı (q, k, mesafe yanlılığı ve bu adımın `v`'si),
- dört yuva projeksiyonu ve yuva önem ağırlığı,
- mesafe yanlılığı.

İki **aynı adım içi yazma** tam olarak türevlenir, çünkü okuma aynı adımda
onları kullanır: yuva yazması (anahtar, değer, geçit, önem) ve özyinelemeli
güncelleme (`φ(k)`, `v`). Önceki belirteçlerin yazmaları dondurulmuştur; bu
bilinçli ve belgelenmiş bir yaklaşımdır, "tam BPTT" olarak sunulmaz.

Doğrulama: sonlu fark sözleşme testi 18 tek parametreyi (okuma ağırlığı, kanal
lojistiği, iki yuva yazma projeksiyonu, yuva sorgu projeksiyonu, mesafe
yanlılığı, FFN kapısı, sorgu, anahtar, değer, bağlı gömme, **öğrenilen bellek
bozunumu ve yazma gücü**, **girdiye bağımlı kanal ve ölçek geçitleri**,
**bağlam koşullu yönlendiricinin iki sütunu**, **indüksiyon ağırlığı**)
merkezi farkla karşılaştırır ve
`1e-4` bağıl toleransla geçer. Tek parametreli sondaların yanında bağlı gömme
matrisinin **tamamı** taranır: `width × sözlük` = 128 elemanın her biri
merkezi farkla karşılaştırılır (`1e-5` tolerans). Bu tarama, aşağıda §4.2'de
anlatılan hatayı yakalayan kontroldür. Yönlendirici sıfırdan başladığı için sonda
*önce* üç elemanına sıfırdan farklı değer verilir (0.30 / −0.20 / 0.25); aksi
halde kayıp yönlendiriciye göre tam doğrusal olurdu ve yalnızca kanal
türevleriyle de eşleşirdi — yani testin asıl konusu olan *bağlam* türevini
sınamazdı. Ayrıca geçidin ve yönlendiricinin sıfır başlatmada sıfırdan farklı
eğim aldığı, eğitimde gerçekten **hareket ettiği**, sıfır yönlendiricinin
girdiye bağımlı karışımı birebir yeniden ürettiği, sıfırdan farklı
yönlendiricinin karışımı **bağlama göre** değiştirdiği ve kontrol noktasından
birebir geri yüklendiği ayrı kontrollerdir (§7, 12. test). İndüksiyon ağırlığı
da sıfırken sınanmaz: `induction_mix` sonda öncesi 0.40 / −0.25 yapılır, çünkü
sıfır ağırlıkta paylaşılan dikkat ağırlıklarına eklenen kaydırılmış terim
kaybolur ve `key`/`query`/`value`/`distance_bias` sondaları onu sınamazdı.

### 4.1 Öğrenilen bellek dinamiği

Ölçek başına iki parametre vardır:

```
decay_logit_j           lambda_j = clamp(sigmoid(decay_logit_j), 1e-6, 1-1e-12)
write_strength_logit_j  g_j      = (1 - lambda_j) * 2 * sigmoid(write_strength_logit_j)
```

Yazma `R <- lambda_j R + g_j phi(k) v^T`. İki türev de aynı adımın içinde
alınır (derinlik-1 BPTT):

- `dL/dlambda_j = sum(dR . R_before)`, ve `R_before` yazma sonrası durumdan geri
  çözülür: `R_before = (R_after - g_j phi(k) v^T) / lambda_j` — `1/lambda_j`
  çarpanı unutulursa gradyan tam bir bozunum faktörü kadar eksik kalır;
- iki parametre **bağlıdır**: `g_j` `lambda_j`'ye bağlı olduğu için
  `d/d(decay_logit_j)` içinde `-2 sigmoid(w_j) * dL/dg_j` terimi de bulunur.
  Bu terim olmadan analitik gradyan, önbelleği tazeleyen sonlu farkla
  uyuşmaz — sözleşme testi tam olarak bunu yakaladı.

Önbellek (`model%scale_decay`, `model%write_strength`) yalnızca
`fbc_refresh_dynamics` ile tazelenir: model kurulumunda, her optimizer
güncellemesinden sonra ve kontrol noktası yüklendikten sonra. **Belirteç
döngüsünde transandantal yok**, ufuk hiçbir boyutta kullanılmaz.
`fbc_learned_horizon` öğrenilen erişimi (`max_j -1/log lambda_j`) bildirir;
başlangıçta bu değer yapılandırılan ufka eşittir, çünkü `decay_logit` eski
merdivenden, `write_strength_logit` ise sıfırdan başlar (`g = 1 - lambda`,
yani eski ileri geçişle **bit düzeyinde aynı**).

### 4.2 Düzeltilen hata: bağlı gömme gradyanı yalnızca tek sütuna yazılıyordu

`logits(j) = h · E(:, j) + b(j)` olduğu için bağlı gömme matrisinin türevi
`h ⊗ dlogits` **tam dış çarpımıdır**: her `j` sütunu `dlogits(j)` kadar
gradyan alır. Gönderilen sürüm bu dış çarpımı yalnızca **okunan belirtecin**
sütununa yazıyordu (`E(:, token) += h · dlogits(token)`, üstüne girdi
terimi). Sonuç: model yanlış baytı **bastırmayı** öğrenemiyor, yalnızca o anda
okuduğu baytın vektörünü kaydırabiliyordu. `dlogits(j) = p_j − 1[j = y]`
olduğundan kayıp, hedef dışındaki sütunlarda tam olarak `p_j` kadardı — yani
her adımda sözlüğün neredeyse tamamı gradyansız kalıyordu ve çıkış kafası
"yanlış baytı düşür" sinyalini hiç almıyordu.

Gönderilen şekilde (genişlik 64, bayt sözlüğü 256) yazılması gereken eleman
sayısı `64 · 256 = 16 384`, yazılan ise 64'tü: çıkış kafası gradyanının
%0,39'unu alıyordu.

Bunun o güne kadarki sözleşme paketiyle yakalanmamış olması öğretici: mevcut sonda okunan
belirtecin sütununu sınıyordu (`E(3, 9)`, belirteç 9) ve **tam o sütunda**
eksik terim zaten yazıldığı için hatalı sürüm ile doğru sürüm aynı sayıyı
veriyordu. Tek sütunluk bir sonda bu hatayı yapısal olarak göremez; bu yüzden
doğrulama artık matrisin tamamını (128 eleman) tarar ve tek sütunluk sonda da
korunur (§7, 12. grup).

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

`VGFB0001` sihirli dizesi, **sürüm 5**. İçerik: sözlük/genişlik/kafa/ffn/yerel
pencere/ölçek/yuva,
**mantıksal ufuk**, etkin ufuk ve kırpma bayrağı, RMS epsilon, yuva bozunması ve
sıcaklığı, güncelleme ve belirteç sayaçları, optimize edici hiper-parametreleri,
tüm parametre yükü ve tüm optimize edici durumu.

Sürüm 2, öğrenilen dinamiği (ölçek başına `decay_logit` ve
`write_strength_logit`) ve onların optimizer momentumunu ekler. Sürüm 3
girdiye bağımlı iki geçit projeksiyonunu (`channel_gate_proj`,
`scale_gate_proj`) ekler (yük 21 → 23 dizi). Sürüm 4 bağlam koşullu kanal
yönlendiricisini (yük 23 → 25 dizi), sürüm 5 ise kafa başına indüksiyon
ağırlığını ve onun momentumunu ekler (yük 25 → 27 dizi). Sürüm 1, 2, 3 ve 4
dosyaları okunabilir ama **dönüştürülmez**: `FBT_ERR_VERSION` (−1506) ile
reddedilir, çünkü o dosyalarda yeni parametrelerin değeri yoktur ve
uydurulacak bir değer (örneğin sıfır geçit ya da sıfır yönlendirici) modeli
sessizce değiştirirdi. Sürüm 2, 3 ve 4 reddi ayrı testlerdir: başlık
yazılır, hem yükleyicinin hem başlık okuyucusunun `FBT_ERR_VERSION`
döndürdüğü ve başlığın gerçek sürüm numarasını koruduğu doğrulanır. En yeni
eski sürüm (4) özellikle önemlidir: diskte bulunması en olası biçim odur ve
sıfır indüksiyon ağırlığıyla tamamlanması öğrenilmiş bir yeteneği sessizce
kapatırdı. Yükleme sonunda
önbellek yeniden kurulur.

Eski `VGNR0001` (neural_pretrain v5) kontrol noktaları **dönüştürülmez**:
`FBT_ERR_LEGACY` (−1503) ile reddedilir ve eski ikili dosyayla açılması
gerektiği söylenir. Test bunu hem başlık okuma hem yükleme yolunda doğrular.
Kaydet→yükle sonrası logitler bit düzeyinde aynıdır.

## 7. Testler ve komutlar

```
make context-tests                     # sözleşme paketi
make context-engine                    # CLI
# CI: .github/workflows/context-quality.yml  (27 koşu, ölçek merdiveni)
cmake -S . -B build -DVEGAI_BUILD_TESTS=ON && cmake --build build
ctest --test-dir build -R vegai_context
```

Sözleşme grupları: seçenek ayrıştırma; ufuk değişmezliği (yapısal sayaçlar);
gelecek sızıntısı yok; parça/akış eşdeğerliği; belirlenimcilik; uzun akışta
sonluluk ve sınırlı durum; sonlu fark gradyan kontrolü; eğitimde kayıp düşüşü;
her bellek bileşeninin gradyan alması ve değişmesi; 93 belirteç uzaklıktaki
cevabın sabit boyutlu durum üzerinden sorgu logitlerine ulaşması (ve bellek
kanalları kapalıyken ulaşamaması); kontrol noktası gidiş-dönüşü ve eski
biçimin (sürüm 2 ve sürüm 3) reddi; bağlam koşullu yönlendirici (sıfır
yönlendirici girdiye bağımlı karışımı birebir yeniden üretir, sıfırdan farklı
yönlendirici karışımı **bağlama göre** değiştirir, sıfır satırlı kanal
dokunulmadan kalır); indüksiyon okuması (`5 7 5 7 5` akışında eşleşen
konumların **bir sonrasındaki** değeri okur, yerinde okuma sıfır kalır);
planın saydığı şeyin gerçekten var olanla eşleşmesi (parametre elemanı,
durum baytı, ve optimize edicinin bir adımda dokunduğu eleman sayısı —
parametre sayısına **tam** eşit); sıfırlanmış durumun taze durumla
**bit-birebir** aynı logitleri vermesi; dolu halka tamponu (her yuvanın kendi
mesafesindeki belirtecin anahtarını ve değerini tutması, ve kaydırılmış
okumanın elle hesaplanan toplamla eşleşmesi).

12. gruptaki gradyan kontrolü ayrıca bağlı gömme matrisinin **tamamını**
tarar (bkz. §4.2): tek sütunluk bir sonda, sondayı okunan belirtecin sütununa
koyduğu sürece bu sınıftaki hatayı göremez.

Buna ek olarak, elle seçilmiş sondaların yapısal körlüğünü kapatmak için **tek
bir yönlü türev** kontrolü vardır: tüm 25 tensörün her elemanı aynı anda
`±δ` kadar (damalı işaretlerle) kaydırılır, kayıp iki yanda ölçülür ve merkezi
fark, aynı gezintinin gradyan üzerinde topladığı `Σ s·g` ile karşılaştırılır.
Geri yayılımın hiç yazmadığı bir tensör analitik toplama hiçbir şey katmaz ama
sonlu fark onu görür; bu yüzden tek sayı 25 tensörün tamamını kapsar. Bozunum
ve yazma gücü önbellekleri her kaydırmadan sonra tazelenir (`fbc_refresh_dynamics`),
çünkü adım lojitleri değil önbelleği okur.

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
(`-fcheck=all -ffpe-trap=invalid,zero,overflow`), koşu 40758732752.
Fiziksel yapı dört satırda da aynı; yalnızca `--window` değişiyor.

| window | parametre | durum baytı | ileri işlem/tok | p50 µs | p95 µs | çıkarım tok/s | eğitim tok/s | eğitim NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 15 497 | 67 072 | 43 008 | 52.418 | 70.041 | 18 548 | 6 721 | 2.6914 |
| 8 192 | 15 497 | 67 072 | 43 008 | 52.228 | 62.497 | 18 911 | 6 719 | 2.7077 |
| 1e6 | 15 497 | 67 072 | 43 008 | 52.388 | 62.407 | 18 888 | 6 693 | 2.7083 |
| 1e12 | 15 497 | 67 072 | 43 008 | 52.168 | 61.605 | 18 986 | 6 738 | 2.6848 |

`window` 32 → 1e12 (3.1e10 kat) için p50 oranı **0.995×**, çıkarım hızı oranı
**1.024×**: ölçüm gürültüsü içinde. Mutlak süreler Debug derlemesine aittir;
Release sayısı burada iddia edilmez.

### 8.2 Ölçek merdiveni: 8 ölçek kaliteyi artırıyor mu?

`Context quality ladder` iş akışı (koşu 40758732610): 3 tohum, aynı bütçe
(32 768 eğitim + 6 144 held-out belirteç), aynı optimize edici; yalnızca ölçek
sayısı değişiyor. Ölçüm **ayrı** akıştan alınır, yani held-out.

| görev | ölçek | parametre | durum baytı | held-out NLL | sd | 4'e göre | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| markov@256 | 2 | 15 495 | 58 368 | 0.00441 | 0.00038 | −0.00001 | — |
| markov@256 | 4 | 15 497 | 67 072 | 0.00442 | 0.00036 | temel | — |
| markov@256 | 8 | 15 501 | 84 480 | 0.00443 | 0.00036 | +0.00001 | — |
| needle@256 | 2 | 15 495 | 58 368 | 0.04996 | 0.00564 | −0.00009 | 0.9565 |
| needle@256 | 4 | 15 497 | 67 072 | 0.05005 | 0.00568 | temel | 0.9565 |
| needle@256 | 8 | 15 501 | 84 480 | 0.05011 | 0.00565 | +0.00006 | 0.9565 |
| needle@1024 | 2 | 15 495 | 58 368 | 0.02688 | 0.00256 | −0.00002 | 0.8000 |
| needle@1024 | 4 | 15 497 | 67 072 | 0.02690 | 0.00258 | temel | 0.8000 |
| needle@1024 | 8 | 15 501 | 84 480 | 0.02692 | 0.00258 | +0.00002 | 0.8000 |

**Karar: 8 ölçek benimsenmedi.** Farklar tohumlar arası standart sapmanın
~1/50'si mertebesinde, yani ölçüm gürültüsü; recall dört ondalıkta birebir aynı.
8 ölçek durum baytını %26 büyütüyor (67 072 → 84 480) ve karşılığında ölçülebilir
bir kazanç üretmiyor. 1024 belirteç uzaklık, 32 belirteçlik halka tamponun
erişemediği bir mesafedir; yani zor rejim de ölçüldü ve orada da fark yok.

Bu sonucun sınırı: görevler 32 bin belirteçte doyuma ulaşıyor (recall 0.9565 /
0.80) ve 64 yuvalı yuva kanalı bu mesafeleri zaten taşıyor. Daha uzun eğitim,
daha büyük model veya merdivenin orta bandının kritik olduğu bir görev
ölçülmedi; bu yüzden "8 ölçek asla yardım etmez" denmiyor, "bu bütçede ve bu
görevlerde yardım etmedi" deniyor.

## 9. Sınırlar (iddia edilmeyenler)

- Özyinelemeli bellek kayıpsızdır denmez: üstel unutma, yuva çarpışması ve
  yanlış hatırlama vardır ve ölçülebilir.
- Etkin ufuk bozunma tabanlıdır; 4e15 üstü aritmetik olarak kırpılır ve
  bildirilir.
- Geri yayılım derinlik-1 kesiklidir; önceki belirteçlerin anahtar/değer
  gradyanları yoktur.
- Yeni modüllerde OpenMP yönergesi yoktur; mevcut isteğe bağlı OpenMP yolu bu
  motor için kullanılmadı ve bu yüzden burada hızlanma iddiası da yoktur.
- Ölçek merdiveni 32 bin belirteçlik bütçede ve iki görevde ölçüldü. "Daha çok
  ölçek daha iyi" iddiası bu bütçenin dışında **kanıtlanmadı**; varsayılan 4
  ölçek, ölçülmüş bir kazancın yokluğuna dayanır, varlığına değil.
- Benchmark süreleri Debug yapısında ölçülür; mutlak değil göreli kanıt olarak
  okunmalıdır.
- Yönlendirici **tek adım derinliğinde** çalışır: geri yayılım derinlik-1
  kesikli olduğu için "yönlendirme → bağlam → yönlendirme" geri beslemesi bir
  adımdan uzağa taşınmaz. Bu bir eksiklik olarak yazılıyor, gizlenmiyor.
- İndüksiyon okuması eşleşen konumun **bir** sonrasını okur; daha uzun
  gecikmeli ("iki sonra", "üç sonra") kopyalama bu mekanizmanın dışındadır ve
  ayrıca ölçülmedi.
- Yönlendirmenin **ölçülmüş kalite kazancı** ablasyonsuzdur: aynı şekilde
  yönlendiricisiz bir koşu yapılmadı, dolayısıyla "yönlendirici şu farkı
  yarattı" denmez (§8.8, `PROMPT_RESPONSE_PRETRAINING.md`).
- §4.2'deki bağlı gömme hatası **düzeltilmeden önce** üretilen tüm eğitim
  sayıları (kopyalama 0/200, sonraki-bayt 0/100 dahil) çıkış kafası yalnızca
  tek sütunla eğitilmiş modellere aittir; bu sayılar düzeltme sonrası
  koşularla **karşılaştırılamaz** ve "model şu kadar öğrendi" kanıtı olarak
  kullanılmaz. Düzeltme sonrası ölçüm ayrıca raporlanır.

## 10. 600 MB prompt/response ön-eğitimi

Ayrı bir boru hattı, aynı motor: bayt düzeyinde tokenizasyon, dosyadan akış,
oturum zinciri ve prompt → yanıt ölçümü.
Ayrıntı: [`PROMPT_RESPONSE_PRETRAINING.md`](PROMPT_RESPONSE_PRETRAINING.md).

- korpus: `tools/build_prompt_response_corpus.py` (deterministik, manifest +
  SHA-256, 600 000 000 bayt varsayılan),
- eğitim/ölçüm: `app/prompt_engine.f90` (`train`, `ask`, `probe`),
- kanıt: `.github/workflows/prompt-response-600mb.yml`,
- gerçek koşu isteği: `ci/prompt-600mb.json` (dispatch yetkisi olmayan
  ortamlar için push tabanlı istek).
