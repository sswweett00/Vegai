# Kısa Türkçe byte-model eğitimi — 2026-11-05

Bu kayıt doğal dil anlamanın veya üretim hazır olmanın sertifikası değildir.
Eşit bütçeyle eğitilmiş **23.411 parametreli V6 byte modeli** için küçük,
ölçülebilir bir NLL karşılaştırmasını ve gerçek eğitim throughput'unu verir.
Metrikler yalnız burada belirtilen veri, seed, derleyici ve CPU için geçerlidir.

## Kapsam ve ön seçim

Önce BOUN'un `train` ön ekleri üzerinde beş eşli seed (21, 42, 73, 101, 137),
ikisi ayrı BOUN `dev` diliminde kullanıldı. Başlangıç modeli, veri sırası,
512-byte chunk, reset kapalı, hedef byte sayısı, update sayısı ve optimizer'ın
öğrenme oranı dışındaki ayarları aynıydı. Her çifti eşit sayıda eğiten
`vegai_natural_quality_ab` kolların çalışma sırasını seed'e göre tersler.
Kısa 33-update dev taraması LR 0,008/0,010/0,012/0,016 ile yapıldı; altı
stream/cold/prefix × iki dev metriğinin tümünde her aday 5/5 seed iyileşti.
Uzun 163-update dev'de LR 0,012 için iki stream metriği 5/5, diğer dört metrik
4/5 seed iyileşti; seed 73'te cold/prefix kötüleşti. LR 0,016 ile ikinci uzun
dev koşusu başlatılmış olsa da ortam sıfırlaması sonucu ham çıktısı korunmadı;
bunun sonucunu tahmin etmiyoruz. 0,012 uzun dev ham satırları süreç çıktısından
[buraya](turkish-natural-runs/prior-dev-1000-12000.txt) aktarılmıştır; derlemeyle
aynı anda çalıştığı için oradaki süreler performans kanıtı değildir. Kısa sweep'in
ham dosyaları sıfırlamada kayboldu; yukarıdaki özet, o koşuların kaydedilmiş
sonuçlarıdır. Korunan uzun dev karşılaştırmasında aday−temel medyanları
(nat/byte) şöyleydi:

| Metrik | BOUN dev A | BOUN dev B | İyileşen seed |
| --- | ---: | ---: | ---: |
| Stream | −0,036389 | −0,031042 | 5/5, 5/5 |
| Cold | −0,042591 | −0,032580 | 4/5, 4/5 |
| Prefix | −0,045876 | −0,045216 | 4/5, 4/5 |

Bağımsız held-out NLL sonuçlarına bakılmadan, daha düşük oran ve kısa dev'in
bir kısmında cold/prefix'in 0,016'ya göre daha iyi olması nedeniyle **0,012**
seçildi. Bu seçim genel varsayılan yapılmadı: `turkish-short` opt-in profili
sadece yeni V6, varsayılan şekil, 512 chunk, reset kapalı ve en çok 163 update
koşularını kabul eder. Seçilen oran checkpoint'te saklanır; kullanıcı resume
ile kapsam dışına çıkabilir. Böyle bir devamın kalitesi ölçülmüş sayılmaz.

## Kaynak veriler ve tekrarlanabilirlik

- [UD Turkish BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN),
  commit `f828b02872123a8dfb336584eab0767c4e55c5f9`, **CC BY-SA 4.0**.
- [UD Turkish Kenet](https://github.com/UniversalDependencies/UD_Turkish-Kenet),
  commit `2d99116a34d98dca3cc965544d387d2795b9829a`, **CC BY-SA 4.0**;
  dilbilgisi örnekleri, BOUN ile aynı dağılım değildir.

Kaynak dosyalar ve türetilen cümleler repoya eklenmez. Native Fortran
`vegai_ud_text_extract`, CoNLL-U içindeki `# text = ` kayıtlarını LF ile ayrılmış
UTF-8 olarak çıkarır. Aynı çıktı aşağıdaki işlemle hazırlanabilir; dev/test
ayıklama yalnız **tam kayıt** düzeyindedir, anlamca benzer satırlar kalabilir:

```sh
cmake --build build --target vegai_ud_text_extract vegai_corpus_overlap \
  vegai_natural_quality_ab vegai_natural_eval vegai_prompt_engine
# BOUN ve Kenet'i yukarıdaki commitlerde BOUN/ ve KENET/ dizinlerine açın.
mkdir -p slices
build/vegai_ud_text_extract BOUN/tr_boun-ud-train.conllu slices/train-1000.txt 1000
head -n 200 slices/train-1000.txt > slices/train-200.txt
build/vegai_ud_text_extract BOUN/tr_boun-ud-dev.conllu slices/dev160.txt 160
head -n 80 slices/dev160.txt > slices/dev-a-80.txt
tail -n 80 slices/dev160.txt | LC_ALL=C grep -Fvx -f slices/train-1000.txt > slices/dev-b-79.txt
build/vegai_ud_text_extract BOUN/tr_boun-ud-test.conllu slices/test240.txt 240
# BOUN test'in ilk 80 satırı önceki çalışmalarda kullanılmıştır; 81–240 ayrıldı.
tail -n 160 slices/test240.txt | LC_ALL=C grep -Fvx -f slices/train-1000.txt > slices/test-boun-159.txt
build/vegai_ud_text_extract KENET/tr_kenet-ud-test.conllu slices/test-kenet-160.txt 160
for f in slices/dev-a-80.txt slices/dev-b-79.txt slices/test-boun-159.txt slices/test-kenet-160.txt; do
  build/vegai_corpus_overlap slices/train-1000.txt "$f"
done
sha256sum slices/{train-200,train-1000,dev-a-80,dev-b-79,test-boun-159,test-kenet-160}.txt
```

| Dilim | Satır | SHA-256 |
| --- | ---: | --- |
| BOUN train ön ek | 200 | `0889ecce1aabec92284cb4cb3994aaa1e862609487a2adc526fa02f71622ac74` |
| BOUN train ön ek | 1000 | `5195714be1073baef5098ee87a6b68584579171510b537a77a721a40e3b8d48c` |
| BOUN dev A | 80 | `708a8a99c1f88b0f5c9313333c3f874f9fdd4ec527b2a1063998fd61a81fb9ab` |
| BOUN dev B | 79 | `9c7ad3135524d9d16564d8b29f1c2642784805f6a0d30d665b4caf8062d5f53e` |
| BOUN yeni test | 159 | `b96524c2c52897e37dc592372099ee69252714c075b83fa16c05ace4aebb1022` |
| Kenet test | 160 | `e9375fa275456d7a797fee24b691815a0de08ba42c275e4866cc60c2f0f0867d` |

Dev B ve BOUN test'ten birer satır exact-train eşleştiği için çıkarıldı. Native
satır-örtüşme denetimi dört held-out/dev diliminde 1000-train ile sıfır tam
kayıt eşleşmesi bildirdi; bu semantik veya kaynaklar arası sızıntıyı dışlamaz.
BOUN kaynak train/dev/test SHA-256 sırasıyla `ae54aef82e69cb7ccc54703a07e2ebcde66c2e4453a5debfac1eb78dfed60e9f`,
`b3531bb26337b3ef258796e6948d2c3469e71704613b9373cd87f341409e2d8d`,
`c7f234f6a7c1b7025f8cab9025239286836e971b653fb1dd3f266a84ac74dbeb`;
Kenet test SHA-256 `74eeefb83d2914b749aa85096ad038db1bd371dc75586d58afc2f78e441b6e74`.

## Ölçüm tanımları

`stream` NLL tüm dosyayı tek akış gibi, satır sonları dahil puanlar. `cold`
her LF kaydını boş durumdan başlatır, ilk byte ve LF'yi puanlamaz. `prefix`
her kaydın en fazla ilk 32 byte sol bağlamıyla yapılan cold tahminlerini
puanlar. Üçü de doğal log negatif olabilirlik, **nat/byte**: daha düşük daha
iyi. Kısa BOUN eğitiminde 16.896 hedef byte ve 33 update, uzun eğitimde 83.456
hedef byte ve 163 update kullanıldı. Aday−temel farkı negatife döndüğünde aday
NLL'si düşmüştür. Bağımsız held-out sonuçları ve her seed'in ham satırları bu
tablosunda ve ham dosyalarda verilmiştir; geliştirme seçimi bu sonuçlara göre
sonradan değiştirilmedi.

## Bağımsız held-out NLL sonuçları

Her hücre medyan aday−temel NLL (nat/byte) ve iyileşen seed sayısıdır.
A = BOUN test 81–240, exact-çakışma temizlendikten sonra 159 kayıt;
B = kaynak-disjoint Kenet test 160 kayıt. Beş seed için verinin sırası ve
bütçe eşlendi. [Ham seed sonuçları](turkish-natural-runs/) saklandı.

| Eğitim | Metrik | BOUN test A | Kenet test B |
| --- | --- | ---: | ---: |
| 33 update | Stream | −0,086998 (5/5) | −0,076922 (5/5) |
| 33 update | Cold | −0,080928 (5/5) | −0,054147 (5/5) |
| 33 update | Prefix | −0,068691 (5/5) | −0,053322 (5/5) |
| 163 update | Stream | −0,028548 (5/5) | −0,005371 (3/5) |
| 163 update | Cold | −0,036385 (4/5) | −0,016912 (3/5) |
| 163 update | Prefix | −0,048689 (4/5) | −0,029666 (3/5) |

Olumsuz sonuçlar gizlenmez: 163 update'te seed 73 BOUN cold/prefix'i sırasıyla
**+0,013494/+0,053721**, Kenet stream/cold/prefix'i **+0,003407/+0,030348/+0,062272**
kötüleştirdi; seed 101 Kenet'te **+0,022426/+0,033599/+0,030804** kötüleştirdi.
Sonuç her seed veya her doğal dil kaynağı için iyileşme değildir. BOUN/Kenet
cold paydaları 14.757/11.412, prefix paydaları 4.899/5.058 hedef byte'tır.
Dev ve held-out koşuları derlemelerle eşzamanlı yürüdüğü için oradaki
süre oranları hız karşılaştırması olarak kullanılmaz. Gerçek CLI'nin 21 ve 73
seed'li dört checkpoint'i iki held-out'ta ayrı `vegai_natural_eval` ile de
puanlandı: cold/prefix NLL'leri bellek-içi A/B ölçümüyle en fazla
`1,29e-14` ayrıldı (kayıt bazında toplamaların sırası farklıdır); sekiz ham
JSON ve [karşılaştırma](turkish-natural-runs/cli-eval-parity.txt) saklandı.

## Gerçek eğitim throughput'u

Gerçek BOUN train-1000 metni ve `vegai_prompt_engine train` ile **her oturumda
83.456 byte-hedef token, 163 update**, varsayılan V6 şekli ve chunk 512
kullanıldı. Aynı beş seed'de temel 0,006 ile seçilen profil sırayla ters
çalıştırıldı; ölçüm sırasında diğer proje derleme/test süreçleri çalışmıyordu.
Host Linux x86-64, sanal Intel Xeon 2,60 GHz ve 2 görünür mantıksal CPU;
serial portable Release (`-O3`, OpenMP kapalı), tek eğitim thread'i. Derleyici
kaynaklardan önyüklenen GNU Fortran 12.2'dir; scanner için iki downstream
kaynağından eşleşen dosya kullanıldı, bu resmi/üretim toolchain doğrulaması
değildir. Ortam, kaynak/ikili hash'leri ve ham raporlar
[buradadır](turkish-natural-runs/).

| Kapsam | Temel LR 0,006 | `turkish-short` LR 0,012 |
| --- | ---: | ---: |
| Medyan eğitim döngüsü süresi | 5,4985 sn | 5,3102 sn |
| Medyan eğitim throughput'u | **15.178 byte-token/sn** | **15.716 byte-token/sn** |
| Min–maks eğitim throughput'u | 15.003–16.621 | 15.173–16.520 |
| Medyan uçtan uca process süresi | 5,502 sn | 5,314 sn |

Eşli profil/temel eğitim süresi oranı medyanı **0,98998**, aralığı
**0,95464–1,00610**. Ölçüm sonuçları bu örnekler ve host içindir; evrensel hız
veya sıfır yavaşlama garantisi değildir. `session_seconds_precise`, eğitim
loop'u ile corpus okuma/kapatmayı içerir; başlatma ve checkpoint yazmayı
**içermez**. Ayrı shell duvar saati uçtan uca işlemi, dolayısıyla checkpoint
çağrısını da içerir; OS cache'i ve dayanıklı disk flush'ı garanti edilmez.
Token, bu modelde bir BPE/kelime tokenı değil, tahmin edilen **bir byte**'tır.

Tekrarlama örneği (aynı komut her seed için iki kolda çalıştırıldı):

```sh
build/vegai_prompt_engine train --corpus slices/train-1000.txt --checkpoint /tmp/short.vgfb \
  --max-updates 163 --seed 73 --progress-every 0 --training-profile turkish-short
# JSON session_tokens, session_seconds_precise, session_tokens_per_second_precise
# alanlarını okuyun; kullanıcının kendi hostunda yeniden ölçmesi gerekir.
```

## Geçerli UTF-8 serbest üretimi

Geliştirme verisinden 79 kısa prompt, 33 update'li LR 0,006 model, seed 21,
sıcaklık 0,8 ve 32 byte çıktı bütçesiyle: ham üretimde **52/79 geçerli**,
**27 geçersiz**, 3 boş yanıt; opt-in `--text-output utf8` ile **79/79 geçerli**,
0 geçersiz ve 0 boş yanıt üretildi. İki modda teacher-forced next-byte
98/548, exact match 0/79 ve NLL önceki koşuda 2,9592416819325065 nat/byte,
yeniden derlenmiş koşuda 2,9592416819325069 idi. Maskelenmiş sampling'in RNG
yolu farklılaşabilir; bu kodlama geçerliliği deneyi metnin anlamlı veya yanıtın
doğru olduğunu göstermez. Prompt TSV SHA-256
`485ad60e92dd4cf033934deff7ebd963841836f8c48783e3d67eafdece2884f9`; önceki
checkpoint SHA-256 `7bdfb79ba5c6cb4b7521b10ab20d7da1e824ccc92f364d6bd4376769f096fc40`,
tekrardaki `af64960729040eba5178a3095758728f226f3a422721027c89002e2810f501e2`.
Checkpoint baytları farklıdır, nedeni doğrulanmadı; bit-parite iddia edilmez.
Ölçüm özetleri [ham çıktılarda](turkish-natural-runs/); metin kayıtları
lisansları nedeniyle repoya dahil edilmedi.

## Sınırlar

Gerçek kullanıcı ve görev değerlendirmesi, daha büyük/çeşitli dil verisi,
geniş bütçeler, uzun bağlam ve dağılım değişimi bu sonuçlarla kanıtlanmaz.
UTF-8 geçerliliği ve byte-NLL iyi diyalog, doğruluk, güvenlik ya da rakiplerden
üstünlük ölçüsü değildir. Eğitim iç döngüsüne ek tam-array taraması, optimizer
güncellemesi veya checkpoint kopyası eklenmedi. Checkpoint üzerine yazma
kesintide eski dosyayı kaybettirebilir; bağımsız yedek/snapshot gerekir.
Windows/macOS, farklı derleyiciler, GPU/NPU ve production toolchain doğrulanmadı.
