# Yuva simetrisi: eşit bütçeli kalite ve maliyet incelemesi (2026-11)

## Bulgular ve kapsam

V5/V6'da bütün yuvalar sıfırdan başlar ve aynı belirteç için aynı yazma
formülünden geçer. Bu nedenle anahtar, değer ve önem dizileri bir akış boyunca
özdeş kalır: fiziksel yuva sayısı, bağımsız adreslenebilir bellek kapasitesini
kanıtlamaz. Yeni regresyon, farklı belirteçlerle 512 adımda bu eşitliği iki
eski sürüm için doğrular. Eski bir testin “dört belirteç yuvaları farklılaştırır”
yorumu yanlıştı; sorgu/önem için `> 0` kontrolü yuvarlama artığını gradyan
zannedebiliyordu. Bu test düzeltildi.

Deneydeki V7 adayı, birden fazla yuva olduğunda anahtarın ilk bileşenine
sıfırlamada kalıcı, belirlenimci ve farklı bir kimlik yerleştirir. Diğer anahtar
bileşenleri, değer ve önem güncellenmeye devam eder; sabit bileşen hem yazma
geçidine hem okuma skoruna katılır. Tek yuvada V6 yazma davranışı korunur.
Bir key bileşeninin içerik için kullanılamaması bir **kapasite ödünleşimidir**;
çeşitlenen durum, kayıpsız saklama veya her görevin daha iyi çözülmesi demek
değildir. Ek öğrenilebilir parametre, ek durum dizisi, checkpoint kopyası veya
ek optimizer güncellemesi yoktur. Sıfırlamada `slots` kadar kimlik atanır; tam
geçmiş taranmaz. Geri yayılım hâlâ **tek adımlı kesikli** yaklaşımdır.

Regresyonlar 1/2/3/8/64 yuva, 1.500 değişken belirteç, reset, tek-yuvada V6
çıkış/ilgili gradyan eşitliği, sıfır olmayan sorgu/önem gradyanları, sabit key
bileşeni dahil merkezi sonlu farklar ve dolu halka üzerinde bütün-parametre
yönlü farkı sınar. V5/V6 checkpoint'ler kendi davranış ve sürümünü korur; V7,
farklı ileri-hesap semantiğini gizlememek için ayrı sürümdür. Bir V7 dosyası,
V7'yi tanımayan eski ikili tarafından açılmaz.

## Veri ve yöntem

Bu küçük CPU byte-model deneyi, diyalog kalitesi, güvenlik, üretime hazır olma
veya bütün endüstri modellerinden üstünlük kanıtı değildir. Veri depoya
konmadı:

- Eğitim ve geliştirme: [UD Turkish BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN),
  `f828b02872123a8dfb336584eab0767c4e55c5f9`, **CC BY-SA 4.0**; resmî
  train'in ilk 200/1.000, dev ve test'in ilk 80 `# text = ` satırı. Kaynak
  README'sindeki Marşan ve ark. ile Türk ve ark. atıfları ve lisans geçerlidir.
  Test-80 daha önceki V5/V6 deneylerinde kullanıldığı için yeni, hiç
  görülmemiş bir test olarak sunulmaz.
- Ek ayrı kaynak: [UD Turkish PUD](https://github.com/UniversalDependencies/UD_Turkish-PUD),
  `63a134a087c8475ea832713b960baba8ff2f9401`, **CC BY-SA 3.0**; test
  dosyasının ilk 80 `# text = ` satırı. Bu bölüm haber çevirileridir ve PUD
  yazar/katkı ve lisans koşulları kaynak README'dedir. Korpus metnini burada
  dağıtmıyoruz.

Her satır sonuna LF eklenmiştir. Kimlikler:

| Parça | Satır / byte | SHA-256 |
| --- | ---: | --- |
| BOUN train-200 | 200 / 17.056 | `0889ecce1aabec92284cb4cb3994aaa1e862609487a2adc526fa02f71622ac74` |
| BOUN train-1000 | 1.000 / 83.834 | `5195714be1073baef5098ee87a6b68584579171510b537a77a721a40e3b8d48c` |
| BOUN dev-80 | 80 / 6.484 | `708a8a99c1f88b0f5c9313333c3f874f9fdd4ec527b2a1063998fd61a81fb9ab` |
| BOUN test-80 | 80 / 5.706 | `fc9e7cddd57d6636f2baa0c6dfdadb61fbf9aa7091c3eae4b01ac3ad65ebd0f3` |
| PUD test-80 | 80 / 10.260 | `28a17570abc1031a2275f002b047d26c3a18c6f444589b9e6ec90850a42adc9c` |

Bu satırlar, sabitlenmiş `.conllu` girdileri kullanıcı tarafından sağlandığında
Fortran `vegai_ud_text_extract` ile Python'suz yeniden hazırlanabilir. Linux
x86-64/GNU Fortran 12.2 ile çıkarılan beş dosyanın SHA-256 değerleri tablodaki
beş değerin tamamıyla eşleşti; UTF-8, LF/CRLF ve boşlukları karıştıran 160 ek
rastgele girdi geliştirme sırasında kullanılan bağımsız Python byte-oracle ile
aynı sonucu verdi (çıkarıcı ve yerleşik sözleşme testi Python gerektirmez).
Başka işletim sistemi ve derleyiciler bu yerel sonuçla kanıtlanmaz. Kullanım, byte kuralları ve
sınırlamalar [UD metin hazırlığı](../UD_TEXT_EXTRACTION.md) belgesindedir;
kaynak lisans ve atıf koşulları ayrıca geçerlidir.

BOUN train-200/1.000 ile PUD test-80 arasında native exact-record aracına
göre sıfır ortak **tam satır** vardır
([ham çıktı](context-slot-runs/pud-overlap.txt)). Bu, kısmi veya anlamsal
örtüşme ve kaynakların tümüyle bağımsızlığı için kanıt değildir. İlk 80 satır
korpusların tamamını temsil etmez.

İki kol aynı ağırlık tohumu, bayt/token eşlemesi, veri sırası, LR=0,006,
chunk=512, fiziksel yapı ve eğitim bütçesiyle çalıştırılır. Tohumlar
21/42/73/101/137; width=32, heads=2, FFN=64, local=32, scales=4, slots=64,
horizon=8192. Her iki kolda **23.411 parametre**, **67.072 state dizi baytı**;
train-200'de 16.896 prediction/33 update, train-1.000'de 83.456/163 vardır.
Kısa dosya kuyruğu eşit prefix için kırpılır; bu, tam dosya CLI epoch'u
değildir. Her held-out'tan önce ayrı reset yapılır; BOUN test'te 5.705, PUD'da
10.259 öğretmen zorlamalı byte prediction değerlendirilir. Ölçüm NLL'dir
(nat/byte; düşük daha iyi); farklı V7 ileri-hesap davranışının etkisini içerir.

Geliştirmede sabit anahtar kimliği ve sabit slot-geçit yanlılığı adayları
karşılaştırıldı. Test kümelerine geçmeden, iki bütçede 5/5 daha düşük dev NLL
veren anahtar kimliği seçildi; geçit yanlılığı uzun bütçede 3/5 iyileşti. Bu
seçim eldeki küçük dev'e dayalıdır. Test sonuçları aşağıda olduğu gibi,
PUD'daki kötüleşen tohum dahil raporlanır.

Süreler `fbc_train_stream` duvar saatidir; veri okuma, hazırlık, değerlendirme
ve checkpoint I/O hariçtir. Kol sırası dönüşümlüdür. Linux x86-64 VM, 2 vCPU
Intel Xeon @ 2.60 GHz, kaynaklardan geçici derlenmiş GNU Fortran 12.2 ve
`-O3 -std=f2018` kullanıldı. Bootstrap derleyici ve onun `-O1` çalışma zamanı
bağımsız üretim toolchain sertifikası değildir. VM yükü ve beş örnek zamanlama
belirsizliği oluşturur; sonuçlar başka makinelere aktarılmaz.

## Test sonuçları: ilk anchor uygulaması

Her hücrede V7−V6 medyan NLL farkı ve iyileşen tohum sayısı vardır. Süre oranı
V7/V6'dır; 1'in üstü yavaşlamadır.

| Train | BOUN test NLL farkı | PUD test NLL farkı | Medyan süre oranı |
| --- | --- | --- | ---: |
| 200 | −0,00384613 (5/5) | −0,00369765 (4/5) | 1,0122 |
| 1.000 | −0,00669737 (5/5) | −0,00746190 (5/5) | 1,0237 |

PUD kısa bütçe seed 137'de NLL **+0,00179997** kötüleşti. İlk uygulamanın
medyan süreleri de bu iki ölçümde sırasıyla yaklaşık **%1,2 ve %2,4** arttı;
bu nedenle sonuç “hız hiç düşmedi” diye sunulamaz. Varsayılan V6 olarak
bırakılmış, V7 yalnızca açık seçimle sunulmuştur. Eski ve yeni dalların makine
kodu/koşulları farklılaşabileceğinden bunun **her makinede** V6 hızının mutlak
olarak değişmediğini de kanıtlamadığı unutulmamalıdır.

Döngü dallanmasını yuva döngüsünün dışına taşıyan ayrı bir aday, beş tohumda
16.896 prediction sonunda anchor ile byte-byte aynı checkpoint üretti. Fakat
BOUN dev üzerinde 200/1.000 eğitim satırında medyan oranları **1,0116/1,0419**
olduğundan depoya alınmadı; ham sonuçlar
[`fast-dev-200.txt`](context-slot-runs/fast-dev-200.txt),
[`fast-dev-1000.txt`](context-slot-runs/fast-dev-1000.txt) ve
[`anchor-fast-checkpoint-parity.txt`](context-slot-runs/anchor-fast-checkpoint-parity.txt)'dedir.
Başka bir bağlı-gömme geri yayılım birleştirmesi deneyinde de beş checkpoint
anchor ile aynıydı; kısa dev ölçümünün medyanı **1,0383** çıktı ve uzun ölçüm
ortamın sıfırlanmasıyla tamamlanmadı. Bu eksik hız kanıtıyla aday depoya
alınmadı. Tekil koşuların hızları daha fazla değişir.

### Sonraki güvenlik kontrollerinin maliyeti

Akış-sayacı/checkpoint ve daha sonra model-şekli/sıra kontrolleri eklenirken
aynı GCC 12.2, veri, V6 LR=0,006, 16.896 prediction, 33 update ve beş tohumla
önce/sonra checkpoint'leri byte-byte karşılaştırıldı: **5/5 aynı**. Eski V5
checkpoint'ten 1.024 prediction/32 update devamı da beş tohumda byte-byte
aynıdır. Bu sonuç bu derleyici/bayrak ve girdiler içindir; farklı derleyiciler
arasında bit düzeyinde eşitlik vaat etmez.

Dönüşümlü, dış duvar saati ölçümüne süreç başlatma, corpus okuma, eğitim ve
checkpoint yazma dahildir. Son V6 şekil kontrollerinde ortanca yeni/eski oranı
**0,9979**; tekil oranlar **0,9159–1,1077** idi
([ham kayıt](context-slot-runs/layout-r6-200.txt),
[V5 devamı](context-slot-runs/layout-v5-resume-parity.txt)). Önceki yalnız
sayaç/checkpoint-korumalı aşamanın V6/V7 kısa koşuları da
[burada](context-slot-runs/guard-r6-200.txt) ve
[burada](context-slot-runs/guard-r7-200.txt) saklanır; bunlar son şekil
kontrollerinin V7 öncesi/sonrası sonucu değildir. O eski referans ikilisi ortam
sıfırlanmasında kaybolduğu için şekil guard'larının V7 öncesi/sonrası süresi
ölçülemedi.

**Yeni checkpoint güvenliği için aynı-host tekrar:** Şekil kontrolleri her iki
kolda mevcutken, [bu yamanın](context-slot-runs/checkpoint-safety.patch)
öncesi/sonrası aynı GCC 12.2 bootstrap ve aynı BOUN bütçesiyle derlenip
karşılaştırıldı. V6 ve V7'de **5/5'er**, V5 devamında **5/5** önce/sonra
checkpoint byte-byte aynı çıktı. Medyan dış süre yeni/eski V6 **1,002408**
(aralık **0,950520–1,032081**), V7 **0,995825** (aralık
**0,934279–1,018906**). Süreler süreç, veri okuma, eğitim ve kayıt yazımını
kapsar; V5 yalnız parite için ölçüldü. Ham kayıtlar
[V6](context-slot-runs/checkpoint-safety-r6-200.txt),
[V7](context-slot-runs/checkpoint-safety-r7-200.txt) ve
[V5](context-slot-runs/checkpoint-safety-v5-resume.txt)'tedir. Kaynak, derleyici,
çalışma zamanı ve ikili hash'leri kayıtlardadır. Zamanlama sürücüsü geçici
Python idi; runtime ve eğitim Python gerektirmez.

Önemli sınır: yeni sandbox'ta yeniden derlenen **her iki** kol, önceki
sandbox'tan kalan checkpoint SHA'larına V5/V6/V7 için **0/5** eşleşti. Eski
checkpoint/binary dosyaları artık mevcut olmadığından farkın kesin nedeni
belirlenemedi; derleyici sürüm adı aynı olsa da bootstrap çalışma zamanı farklı
kurulmuştur. Bu yüzden yukarıdaki **5/5** yalnız yeni sandbox'taki eşleştirilmiş
öncesi/sonrası sonucu olarak okunmalı, sandbox'lar arası bit-parite iddiası
olarak kullanılmamalıdır.

Son kaynakla daha önce kullanılmış aynı BOUN/PUD held-out'ları tekrar
skorlandığında kısa bütçede medyan NLL farkları yine **−0,00384613 / −0,00369765**,
V7/V6 medyan eğitim süresi **1,002833** oldu
([ham sonuç](context-slot-runs/final-anchor-test-two-200.txt)); beş tohum,
her iki kolda ve iki kümede 20 NLL'nin önceki kayıttan en büyük mutlak farkı
**8,89×10⁻¹⁶ nat/byte**'tır. Bu, yeni bağımsız bir
test kümesi değildir. Beş örnek ve VM değişkenliği, yavaşlamanın tüm koşullarda
olmadığını veya farklı bütçe/donanımlarda aynı sonucu kanıtlamaz; V7 opt-in
kalmaktadır.

GCC'de hata denetimli `allocate(stat=...)` eklendiğinde kayıp çekirdeğinin
uzmanlaşması değişebildiği ve küçük yuvarlama farkları oluştuğu görüldü.
Eğitime özel açık-biçimli görünüm aynı bit sonucunu geri getirdi; genel loss
API'si strided dizileri desteklemeyi sürdürür. Bu yol yeni bir tensor
kopyalamaz veya tam dizi taraması eklemez; test edilen GCC derlemesinde çağrı
geçici dizi uyarısı üretmedi. Diğer derleyicilerin makine kodu ve hızı ayrıca
ölçülmelidir.

Ham veri, doğrulama ve kaynak/binary kimlikleri:
[`context-slot-runs/`](context-slot-runs/). NLL karşılaştırmalarını yeniden
çalıştırmak için CMake ile `vegai_slot_quality_ab` hedefi derlenip aşağıdaki
örnek çalıştırılır; araç iki held-out'u **aynı iki eğitimden** sonra ayrı ayrı
skorlar:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVEGAI_BUILD_TESTS=ON
cmake --build build --target vegai_slot_quality_ab
build/vegai_slot_quality_ab train-200.txt boun-test-80.txt pud-test-80.txt 512 6000 prompt
```

Dosyalar caller tarafından lisanslarına uygun hazırlanmalıdır. Araç aynı dosya
veya byte-byte aynı girdileri reddeder, ama kısmi veri sızıntısını tespit
etmez. Exact-record kontrolü eğitim maliyetinin dışında ayrı çalıştırılabilir.
NUL byte V6/V7'de ortak biçimde token 256'ya eşlenir. Süre ve NLL'ler, burada
ölçülmeyen veri, model şekli, eğitim bütçesi, derleyici veya donanım için
performans garantisi vermez.
