# Byte-prompt eğitimi: v5/v6 ayrı küme deneyi (2026-11)

## Kapsam ve veri

Bu, küçük bir CPU byte modeli üzerindeki sınırlı bir deneydir; üretim kalitesi
veya genel dil-modeli başarısı garantisi değildir. Corpus metni repoya eklenmez.
Kaynak: [UD Turkish BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN),
`f828b02872123a8dfb336584eab0767c4e55c5f9` revizyonu, CC BY-SA 4.0. Yazarlar,
lisans ve yeniden dağıtım koşulları kaynak README ve LICENSE dosyasındadır.
Bkz. Marşan ve ark., *Enhancements to the BOUN Treebank Reflecting the
Agglutinative Nature of Turkish*, arXiv:2207.11782; Türk ve ark., *Resources
for Turkish Dependency Parsing*, DOI: 10.1007/s10579-021-09558-0.

Resmî train/dev/test dosyalarındaki ilk 200 veya 1.000 train, ilk 80 dev ve ilk
80 test `# text = ` satırı, her birinin sonuna LF eklenerek kullanıldı. Train ile
dev/test arasında aynı tam satır yok; bu sonradan native `vegai_corpus_overlap`
ile de doğrulandı (`optimized-final-overlap.txt`). Daha kısmi örtüşme ve kaynak
bağımsızlığı ayrıca denetlenmedi. Sıradan seçilmiş bu az sayıda cümle corpusun
tamamını temsil etmez. Kontrol toplamları:

| Parça | Satır / byte | SHA-256 |
| --- | ---: | --- |
| train-200 | 200 / 17.056 | `0889ecce1aabec92284cb4cb3994aaa1e862609487a2adc526fa02f71622ac74` |
| train-1000 | 1.000 / 83.834 | `5195714be1073baef5098ee87a6b68584579171510b537a77a721a40e3b8d48c` |
| dev-80 | 80 / 6.484 | `708a8a99c1f88b0f5c9313333c3f874f9fdd4ec527b2a1063998fd61a81fb9ab` |
| test-80 | 80 / 5.706 | `fc9e7cddd57d6636f2baa0c6dfdadb61fbf9aa7091c3eae4b01ac3ad65ebd0f3` |

Örneğin kaynak `.conllu` dosyasından `LC_ALL=C awk -v n=200 '/^# text = / {sub(/^# text = /, ""); print; if (++k == n) exit}' tr_boun-ud-train.conllu > train.txt` ile train-200 byte-byte yeniden üretildi. Dev/test için ilgili dosya ve `n=80`, train-1000 için `n=1000` kullanılır. Kaynağı indirirken ve metni dağıtırken upstream lisans koşulları geçerlidir.

## Yöntem ve ön seçim

Test kümesine bakılmadan, dev üzerinde 0,003/0,004/0,005/0,006 adaylarından
**0,006** seçildi; uzun dev'de 0,003/0,005/0,006 da karşılaştırıldı. Kısa ve
uzun dev'de 0,006 beş seed'in hepsinde eski v5/0,003'ten daha düşük NLL verdi.
Test öncesi karar eşiği: her iki bütçede en az 4/5 seed iyileşmesi ve ilave
token, güncelleme, parametre veya recurrent-state allocation olmaması. Testte
sadece seçilen oran ve salt gradient etkisini göstermek için eşit 0,003 kontrolü
çalıştırıldı; test sonucuna göre başka oran seçilmedi.

`vegai_context_quality_ab` komutu aynı tohumdan, aynı tokenları ve ön eki
kullanarak iki kolu dönüşümlü sırayla çalıştırır. Gerçek prompt eşlemesinde
1..255 byte'ları aynı token ID'leridir; v5 NUL'u kodlayamadığından bu A/B
aracı NUL içeren dosyayı reddeder. Önceki keşif çalışmalarındaki tüm byte'ları
bir kaydıran token eşlemesi gerçek prompt CLI'yi temsil etmediğinden burada
seçim veya test sonucu olarak kullanılmamıştır.

- Seed'ler: 21, 42, 73, 101, 137; width=32, heads=2, ffn=64, local=32,
  scales=4, slots=64, istenen horizon=8192; 23.411 parametre ve 67.072 byte
  recurrent-state dizisi; chunk=512.
- 200 cümle: her kolda 16.896 train prediction, 33 update. 1.000 cümle:
  83.456 prediction, 163 update. Her iki durumda eğitim ön eki 512'nin katına
  kırpılır; normal CLI dosya sonundaki kısa kuyruğu da işler. Dolayısıyla bu
  bir tam-corpus CLI epoch ölçümü değil, eşit prefix bütçesidir (CLI ile aynı
  ön ek için sırasıyla `--bytes 16897` ve `--bytes 83457`).
- Held-out state sıfırlanır; öğretmen zorlamalı tek byte NLL'si (nat/byte,
  düşük daha iyi) dev için 6.483, test için 5.705 prediction üzerinden ölçülür.
  Aynı state test içindeki satırlar arasında korunur. Bu skor diyalog kalitesi,
  güvenlik veya downstream görev başarısı değildir.
- V5 kolu bu checkout'un korunan v5 eğitim kuralını kullanır; eski binary değil.
  Gerçek eski binary ile v5 checkpoint devamı ayrıca doğrulanmıştır. V6 ile
  seçilen 0,006'nın karşılaştırması **gradient ve LR değişimini birlikte**
  ölçer; eşit LR kontrolü gradient etkisini ayrıştırır.
- Süre, sadece `fbc_train_stream` çağrılarının duvar saatidir; corpus okuma,
  setup, held-out değerlendirme ve checkpoint I/O dahil değildir. Dönüşümlü
  sıra ve oranlar sistem yükünü tamamen ortadan kaldırmaz. Linux x86-64 VM,
  2 vCPU Intel Xeon @ 2.60 GHz, GNU Fortran 12.2.0, `-O3`, F2018, portable
  CPU profili kullanıldı; bu geçici bootstrap derleyicisi bağımsız üretim
  toolchain sertifikası değildir. Context çekirdeği bu ölçümde seri çalışır.

Dev medyan farkları (v6−v5 NLL, negatif iyileşme):

| Train | v6 LR | Medyan NLL farkı | Medyan eğitim süre oranı v6/v5 |
| --- | ---: | ---: | ---: |
| 200 | 0,003 | −0,000389 | 0,9956 |
| 200 | 0,004 | −0,100619 | 0,9977 |
| 200 | 0,005 | −0,157715 | 0,9984 |
| 200 | 0,006 | −0,197074 | 1,0050 |
| 1.000 | 0,003 | −0,000439 | 0,9900 |
| 1.000 | 0,005 | −0,043916 | 1,0043 |
| 1.000 | 0,006 | −0,056336 | 0,9924 |

## Ayrı test sonuçları

V5 LR=0,003; seçilen V6 LR=0,006. Sayılar nat/byte ve eğitim saniyesidir.
Aynı güncellemeler, seed ve parametrelerle beş seed'in tamamında NLL düştü.

| Train | Seed | v5 NLL | v6 NLL | Fark | v5 s | v6 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 21 | 2,903380 | 2,689312 | −0,214069 | 0,9064 | 0,8975 |
| 200 | 42 | 2,860125 | 2,674679 | −0,185446 | 0,8882 | 0,8980 |
| 200 | 73 | 2,786236 | 2,632920 | −0,153316 | 0,9144 | 0,8789 |
| 200 | 101 | 2,846781 | 2,640390 | −0,206391 | 0,8834 | 0,8680 |
| 200 | 137 | 2,910910 | 2,682910 | −0,227999 | 0,8765 | 0,8692 |
| 1.000 | 21 | 2,493402 | 2,431037 | −0,062365 | 4,3649 | 4,3333 |
| 1.000 | 42 | 2,483189 | 2,427631 | −0,055558 | 4,2667 | 4,4173 |
| 1.000 | 73 | 2,476705 | 2,413885 | −0,062820 | 4,2809 | 4,3518 |
| 1.000 | 101 | 2,484340 | 2,440798 | −0,043541 | 4,3644 | 4,3187 |
| 1.000 | 137 | 2,468944 | 2,408903 | −0,060040 | 4,3618 | 4,3057 |

Test medyan NLL farkı sırasıyla **−0,206391** ve **−0,060040**; medyan
süre oranları 0,9901 ve 0,9928. Bazı tekil seed'lerde v6 daha yavaştı; bu
ölçümler her donanımda çalışma süresinin artmayacağını kanıtlamaz. Eşit LR
0,003 kontrolünün test medyan farkı −0,000430 ve −0,000655'tir; uzun bütçede
yalnız 3/5 seed iyileşmiştir. Dolayısıyla büyük kazanımı salt doğru gradient'e
atfetmek doğru değildir. Ek bir Make smoke koşusunda (küçük şekil, ASCII train
ve UTF-8 fixture held-out, 6.528/642 prediction, eşit LR=0,003) medyan fark
**+0,072824** oldu; farklı dağılımlı küçük fixture'larda bu değişiklik tek başına
kalite garantisi vermiyor. Bu fixture'lar BOUN gibi ayrı-kaynak değerlendirmesi
değildir. Tüm BOUN seed ve tam hassasiyetli çıktıları
[`context-prompt-byte-runs/`](context-prompt-byte-runs/) içindedir.

### Uçtan uca maliyet ve son optimizasyon

Kayıp testinden sonra v6 backward'daki beş outer-product geçici matrisi yerine
mevcut gradyana sütun bazında biriktirme uygulandı; v5 eski ifadeyi korur.
Önceki ve sonraki v6 binary'lerle yukarıdaki iki prefix, beş seed ve LR 0,006
çalıştırıldığında checkpoint'ler **10/10 byte-identical** çıktı (yalnız bu GNU
ve makinede). Dolayısıyla yukarıdaki held-out değerleri bu koşullarda değişmedi;
farklı derleyicilerde aynı bit düzeni vaat edilmez. Kayıp raporundaki kaynak
hash'i bu optimizasyondan *önceki* dosyaya aittir.

Ayrı CLI duvar süresi ölçümünde eski gerçek binary LR=0,003, optimizasyondan
önceki ve sonraki v6 binary'leri LR=0,006 ile dönüşümlü çalıştırıldı; okuma,
kurulum ve checkpoint yazımı dahildir. Yeni/eski medyan süre oranı 200 ve 1.000
bütçede sırasıyla 0,9909 ve 0,9744; yeni/optimizasyon öncesi 0,9890 ve
0,9480'dir. 1.000 bütçede bir seed'de yeni/eski 1,0110 oldu. Önceki ayrı,
optimizasyon öncesi eşit-LR CLI ölçümünde medyan yeni/eski 1,0243'tü; bu
olumsuz sonuç da ham dosyada korunmuştur. Zamanlama değişkendir, hiçbir
platformda maliyet artmaması garantisi çıkarılamaz; token/update/parametre ve
state bütçesi ise aynıdır, ek tam-array taraması veya checkpoint kopyası yoktur.

Daha sonra gerçek eski v5 ve optimize v6 **ikisi de LR=0,003** ile, aynı
corpus, seed, parametre, token ve update bütçelerinde, sıra dönüşümlü olarak
tekrar ölçüldü. 200 ve 1.000 cümle ön ekleri için beş eşleştirilmiş süre
oranının medyanı sırasıyla **0,9452** ve **0,9480** idi; uzun bütçede bir
seed'de oran **1,0018** oldu. Bu işlem başlatmayı, corpus okumayı ve checkpoint
yazımını içeren duvar saatidir; paylaşılan VM gürültüsü ve sınırlı örneklem,
kostun her koşuda azaldığını veya artmayacağını kanıtlamaz. İki kolda da CMake
JSON denetimi 16.896/83.456 prediction, 33/163 update, 23.411 parametre ve
`numeric_ok` durumunu doğruladı. Sadece süre ölçüldü; yeni bir held-out deney
çalıştırılmadı. Eşit-LR binary hash'leri ham dosyadadır; daha sonra CLI/probe
çıktı ve sınır güvenliği değişiklikleri yapıldığı için bu zamanlama bitişteki
CLI binary'sinin doğrudan ölçümü değildir.

Tam satırlar `cli-equal-rate-timing.txt`, `cli-outer-validation.txt` ve
`cli-optimized-equal-rate.txt` dosyalarındadır.

### Optimize kaynakla doğrudan held-out tekrarı (2026-11-02)

Optimize V6 ve korunan V5 kuralı, **aynı BOUN test satırları**, iki bütçe ve beş
seed ile güncel benchmark binary'sinde tekrar çalıştırıldı. Bu yeni, bağımsız
bir test kümesi değildir; aynı deneyin optimizasyon sonrası tekrarıdır. V5 yine
eski binary değil, aynı binary'deki korunan koldur. NLL değerleri önceki
optimizasyon-öncesi tablodan en fazla `1,34e-15` nat/byte farklıdır; bu yalnız
buradaki derleyici ve veri için gözlemdir, farklı platformlarda bit eşitliği
vaat etmez. Eğitim için 16.896/83.456 prediction, 33/163 update, 23.411
parametre ve 67.072 state byte korunmuştur.

| Train | V6 LR (V5 0,003) | Medyan V6−V5 NLL | Medyan eğitim süresi oranı V6/V5 |
| --- | ---: | ---: | ---: |
| 200 | 0,003 | −0,0004302605 | 0,9633 |
| 200 | 0,006 | −0,2063913277 | 0,9938 |
| 1.000 | 0,003 | −0,0006549543 | 0,9615 |
| 1.000 | 0,006 | −0,0600403825 | 0,9745 |

LR=0,006 ve kısa bütçede seed 101 için süre oranı **1,2189**, uzun bütçede
iki seed için 1'in üzerindeydi. Bu farklar ve yüksek koşu değişkenliği nedeniyle
maliyet artmayacağı iddia edilemez. Bu tablo yalnız eğitim çağrısının süresidir;
başlatma ve checkpoint I/O içermez ve eski gerçek binary ile CLI karşılaştırması
değildir. Ham seed sonuçları ve güncel araç/kaynak/derleyici kimlikleri
`optimized-test-*.txt` ve `optimized-final-metadata.txt` dosyalarındadır.

Bu sınırlı test eşiği karşılandığından yalnız **yeni**, varsayılan mimari,
chunk=512, reset kapalı prompt eğitimlerinde varsayılan LR 0,006 oldu. Diğer
şekil/chunk/reset konfigürasyonları 0,003'te kalır, açık `--lr` her zaman
geçerlidir, v5/v6 checkpoint devamında checkpoint'in LR ve diğer optimizer
ayarları korunur. Büyük/uzun eğitimler, diğer diller, cihazlar ve gerçek
kullanıcı görevleri için oran yeniden doğrulanmalıdır.

Tekrarlamak için `cmake --build <build> --target vegai_context_quality_ab`,
sonra örneğin `vegai_context_quality_ab train.txt test.txt 512 6000 prompt`;
`make context-quality-ab` eşdeğer yerel hedefi sağlar. Araç eğitim ve held-out
olarak birebir aynı byte'ları reddeder; farklı isimle kopya sızıntısını da
önler, fakat başka veri sızıntılarını tespit etmez. İsteğe bağlı ayrı
[`vegai_corpus_overlap`](../CORPUS_OVERLAP.md) tam satır çakışmalarını da
araştırır; eğitimden ayrı bir dosya taramasıdır ve yakın/semantik sızıntıları
tespit etmez.

Sonraki kaynak sürümünde benchmark, geçersiz UTF-8, sondaki boşluklu veya
fazla uzun corpus yollarını reddeder; başarısız dosya boyutu sorgusunu ayrıca
ele alır. Bu girdi doğrulaması ölçümlerin ardından eklenmiştir ve aşağıdaki
ölçüm kaynağı hash'leri güncel dosyaların hash'i değildir.

Ölçüm kaynaklarının SHA-256 değerleri (derleyici/binary kimliğinin yerine geçmez):

```
4a8ece4e2595519164f38fea6b8da83fc7613c5647797f246c37ffdf12aa37d2  tools/benchmark_context_quality.f90
396daac4b9e66d64888af13287a219d50f0a93120f8c5966053c6cc7ec3248d3  src/fixed_budget_context.f90
d8a9f52d73e2a1cd91e8b5d48ff4dfc2ac79d414c7dd1f268479e2526382b418  src/fixed_budget_backward.f90
c403d1bbb29408c1229d3b2e6fc7e6dfe32bc0606e432cee3478158ceb2f1867  src/fixed_budget_optimizer.f90
983ecabe2c66aacf7cccec7c08921e29b1c74809e13e0c932d5d23b16eee0fba  src/fixed_budget_training.f90
```
