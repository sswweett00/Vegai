# 600 MB prompt/response ön-eğitimi ve ölçümü

Bu belge, sabit bütçeli bağlam motorunu **600 MB'lık bir prompt/response
korpusu** üzerinde eğiten ve ardından **prompt → yanıt** davranışını ölçen boru
hattını anlatır. Amaç dil bilgisi iddiası değil; mimarinin veriyle ne yaptığını
ölçülebilir kılmaktır.

İlgili sözleşme: [`FIXED_BUDGET_CONTEXT_ENGINE.md`](FIXED_BUDGET_CONTEXT_ENGINE.md).
Oradaki ufuk bağımsızlığı, üç kanal, gradyan ve kontrol noktası kuralları burada
da geçerlidir.

## 1. Neden sentetik korpus

Depoda 600 MB lisanslı Türkçe metin **yok** ve bu ortamın Wikipedia dökümüne
erişimi de yok. Bu yüzden korpus uydurulmuş bir döküm değil, **tohumdan tam
olarak yeniden üretilebilen** yapılandırılmış bir veri kümesidir:

- `tools/build_prompt_response_corpus.py` korpusu üretir,
- her koşuda manifest yazar: **istenen bayt sayısı, gerçek bayt sayısı, SHA-256,
  belge sayısı, tüm olgu bağları (subject → answer) ve probe değerleri**,
- `verify` adımı dosyayı yeniden okuyup boyutu ve SHA-256'yı manifestle
  karşılaştırır; uyuşmazsa koşu durur.

Üretilen metin Türkçe biçimli sahte sözcüklerden oluşur (açık hecelerden
birleştirilir), yani **dünya bilgisi taşımaz**. Taşıdığı şey ölçülebilir üç
görevdir:

| Görev | Cümle kalıbı | Neyi ölçer |
| --- | --- | --- |
| `fact` | `Soru: X nedir? Cevap: Y.` | Akış boyunca taşınan bağ: X'i görüp Y'yi hatırlamak |
| `copy` | `Tekrar et: W. Sonuç: W.` | Bağlam içi kopyalama (ezber değil) |
| `next` | `Say: abcdef. Sonra: bcdefg.` | Yerel bir kural: bir harf kaydırma |

Her belgede 6 olgu cümlesi, dolgu metni, bir kopyalama ve bir kaydırma görevi
vardır; olgular belge başına deterministik bir akışla karıştırılır, böylece aynı
olgu düzenli aralıklarla değil, düzensiz konumlarda geçer. Bu, "hatırlama"yı
sabit konumlu bir arama olmaktan çıkarır.

Varsayılanlar: **600 000 000 bayt**, tohum **20261028**, **2048** olgu bağı.

```
python3 tools/build_prompt_response_corpus.py generate corpora/prompt_600mb.txt \
  --bytes 600000000 --facts 2048 --probe-out corpora/probe.tsv --overwrite
python3 tools/build_prompt_response_corpus.py verify corpora/prompt_600mb.txt
```

## 2. Tokenizasyon: bayt = token

Motorun sözlük varsayılanı 256'dır, korpus ise UTF-8 metindir. Bu yüzden
**token kimliği = bayt değeri**: tokenizer yok, sözlük dosyası yok, eğitim ile
değerlendirme arasında sözlük kayması **olamaz**. Manifestte yazan bayt sayısı,
modelin gördüğü token sayısıdır.

Türkçe harfler UTF-8'de iki bayttır; `app/prompt_engine.f90` bunu
`transfer`/`iand` ile ham bayt olarak ele alır (`byte_of`, `byte_character`),
`iachar`/`achar`'ın 7 bit varsayımına güvenmez.

## 3. Eğitim: dosyadan akış

`vegai_prompt_engine train` korpusu **belleğe almaz**:

1. `chunk + 1` bayt, dosya içinde açık bir konumdan okunur,
2. bu blok tek bir optimizasyon adımı üretir (`fbc_train_stream`),
3. konum `chunk` bayt ilerler; bloğun son baytı bir sonraki bloğun ilk baytı
   olarak yeniden okunur, böylece **tahmin kayması blok sınırında bozulmaz**,
4. yinelemeli durum (bellek kanalları, yuvalar, yerel halka) bloklar arasında
   **sıfırlanmaz**; bellek dosya boyunca taşınır.

Son blok kısaltılabilir: bir aralığın son baytı hiçbir zaman tahmin hedefi
olamayacağı için aralık `bayt - 1` uzunluğundadır ve bu dürüstçe raporlanır.

Koşuyu durduran her sınır açıktır ve raporlanır:

| Seçenek | Anlamı |
| --- | --- |
| `--bytes N` | Eğitimin **bitiş** baytı (0 = dosya sonu) |
| `--start-byte N` | Bu oturumun başlayacağı bayt (zincirleme için) |
| `--max-updates N` | Güncelleme sayısı sınırı |
| `--max-seconds S` | Duvar saati sınırı (oturum uzunluğu) |
| `--resume FILE` | Kontrol noktasından devam (model + optimizer + sayaçlar) |
| `--progress-every N` | İlerleme satırı sıklığı |

Her oturum `CORPUS_TRAIN_JSON { ... }` bloğu basar; blok **geçerli JSON**'dur
(bu, bir hata sınıfını ortadan kaldırdı: Fortran `L1` ile `T`/`F`, `F0.x` ile
`.000` yazıyor ve virgülleri koymuyordu).

## 4. Kanıt iş akışı

`.github/workflows/prompt-response-600mb.yml`:

1. gfortran kurar, motoru, sözleşme testlerini ve korpus CLI'sini derler,
2. **sözleşme testlerini çalıştırır** (gradyan sondaları, ufuk değişmezliği,
   kontrol noktası gidiş-dönüşü),
3. korpusu üretir ve doğrular (boyut + SHA-256),
4. eğitimi **kontrol noktalı oturumlar** hâlinde yürütür; her oturum
   `--max-seconds` ile sınırlıdır ve bir sonraki oturum kaldığı bayttan devam eder,
5. `tools/summarise_prompt_response_run.py` ile oturum raporlarını doğrular ve
   toplar (eksik alan veya `T`/`F` görürse **hata** verir),
6. `filter-probe` ile yalnızca **eğitilen bayt aralığında gerçekten geçen**
   olguları seçer; görülmemiş olgular ölçümden çıkarılır ve kaç tanesinin
   çıkarıldığı raporlanır,
7. `probe` ve `ask` ile prompt → yanıt ölçümünü yapar,
8. tüm sayıları `tools/report_prompt_response_evidence.py` ile tek blok hâlinde
   yazar ve **annotation** olarak da yayınlar (artifact deposu okunamasa bile
   sonuç görünür kalır).

Workflow `workflow_dispatch` ile parametre alır. Bu ortamda dispatch yetkisi
yoktur (HTTP 403), bu yüzden **gerçek koşu `ci/prompt-600mb.json` dosyasıyla
istenir**: push geldiğinde workflow bu dosyayı okur; dosya yoksa beş dakikalık
bir duman testi koşar. Aynı dosya `resume_run_id` / `resume_start_byte` alanları
ile **önceki bir koşunun kontrol noktasından devam** edebilir; devam ederken
korpus SHA-256'sı eşleşmezse koşu reddedilir. Dosya `enabled: true` demedikçe koşu başlatılmaz:
varsayılan `false`'tur, çünkü gerçek bir geçiş saatler sürer ve kazara
tetiklenmemelidir.

## 5. Ölçüm: ne sayılıyor

Her probe satırı iki kez puanlanır:

1. **greedy yanıt** (açgözlü çözümleme) → `exact` eşleşmesi; yanıt `'.'` veya
   satır sonunda kesilir, etiketle birebir karşılaştırılır;
2. **teacher-forced** geçiş → beklenen yanıtın baytları tek tek beslenir ve
   - kaç beklenen baytı model **ilk sırada** sıraladı (`correct_bytes`,
     `byte_accuracy`),
   - beklenen yanıtın ortalama negatif log-olabilirliği (`mean_nll`).

İkinci ölçüm, tam eşleşme sıfır olduğunda bile modelin ne kadar yaklaştığını
gösterir; 256 sınıflı bir bayt modelinde şans düzeyi `1/256 = 0.0039`'dur ve
karşılaştırma bu tabana göre yapılmalıdır.

## 6. Yeniden üretim

```bash
make prompt-engine context-tests          # derleme + sözleşme testleri
python3 tools/verify_prompt_response_corpus.py
python3 tools/build_prompt_response_corpus.py generate corpus.txt --bytes 2000000 \
  --facts 256 --probe-out probe.tsv --overwrite
./build/vegai_prompt_engine train --corpus corpus.txt --checkpoint model.vgfb \
  --train-chunk-tokens 128 --width 16 --ffn 32 --local-window 16 --memory-slots 32
python3 tools/build_prompt_response_corpus.py filter-probe corpus.txt \
  --probe probe.tsv --out probe-trained.tsv --trained-bytes 2000000
./build/vegai_prompt_engine probe --checkpoint model.vgfb --probe probe-trained.tsv
./build/vegai_prompt_engine ask --checkpoint model.vgfb --prompt "Soru: ... nedir? Cevap: "
python3 tools/report_prompt_response_evidence.py .
```

## 7. Ölçülen sonuçlar

Aşağıdaki sayılar workflow'un kendi raporlarından alınmıştır; hiçbiri elle
hesaplanmamıştır. Her koşunun kimliği (run id, commit, SHA-256) kayıtlıdır.

### 7.1 İlk tam geçiş — 600 MB, `lr = 3e-3` (run 40772938467, commit `3a5b026`)

| Ölçüm | Değer |
| --- | --- |
| Korpus | 600 000 000 bayt, 1 028 247 belge, 2048 olgu bağı |
| Korpus SHA-256 | `d3adca9ba0348605da2f0b4968a0f9e996420c961dedd1162633e687985f6b7d` |
| Tüketilen | **599 999 999 token = korpusun %100'ü**, 7 oturum |
| Süre / hız | 9 566 s (2 sa 39 dk), **62 722 token/s** |
| Eğitim NLL | oturum 1: 1.2178 → oturum 7: **1.4185** (yükseldi) |
| Öğrenilen erişim | 3 935.9 → **984.6 token** (model unutmayı hızlandırdı) |
| Parametre | 8 001 |
| Probe `fact` | tam eşleşme 0/200; sonraki-bayt **%3.3** (2 148 bayt), NLL 4.21 |
| Probe `copy` | tam eşleşme 0/200; sonraki-bayt **%3.3** (2 171 bayt), NLL 4.15 |
| Probe `next` | tam eşleşme 0/100; sonraki-bayt **%4.2** (600 bayt), NLL 9.15 |
| Genel | tam eşleşme **0.000**, sonraki-bayt **0.034** (500 satır) |

Yorum (ölçümün söylediği, fazlası değil):

- Korpusun **tamamı** gerçekten tüketildi; bu bir iddia değil, `end_byte =
  599 999 999` ve `whole_file` raporundan okunuyor.
- Kayıp **arttı**. Yani bu koşu "daha iyi bir model" üretmedi: sabit `lr = 3e-3`
  ile 586 bin güncelleme sonunda model kararsızlaştı (NaN yok, `numeric_ok`
  doğru; bu bir ıraksama değil, salınım).
- Öğrenilen bellek erişimi 3 936 token'dan 985 token'a indi: optimizasyon
  bellekten çok yerel bağlama yaslandı. Bu, yeni dinamiğin **gerçekten
  öğrenildiğinin** kanıtı; mimarinin ufku koruduğunun değil.
- Tam eşleşme sıfır, sonraki-bayt doğruluğu %3.3–4.2. Şans düzeyi
  1/256 = **%0.39**; yani model şansın ~8–10 katı, ama bir yanıtı uçtan uca
  doğru yazmaktan çok uzak.
- `next` ailesinde NLL 9.15: beklenen 6 harflik kaydırma dizisi model için en
  zor görev, çünkü 26^6 olasılık arasından tek doğru diziyi istiyor.

### 7.2 İkinci tam geçiş — 600 MB, `lr = 3e-4` (run 40795532458, commit `51e6839`)

Aynı korpus (aynı SHA-256), aynı model şekli, tek fark öğrenme oranı.

| Ölçüm | Değer |
| --- | --- |
| Tüketilen | **599 999 999 token = korpusun %100'ü**, 8 oturum |
| Süre / hız | 11 906 s (3 sa 18 dk), **50 395 token/s** |
| Eğitim NLL (oturum sırasıyla) | 1.4086, 1.3007, 1.2809, 1.2564, 1.2275, 1.2158, 1.2089, **1.2034** |
| Öğrenilen erişim | 8 050.5 → **7 940.0 token** |
| Probe | tam eşleşme **0/500**; sonraki-bayt: fact %3.6, copy %3.2, next %2.7, genel **%3.3** |
| Teacher-forced NLL | fact 7.23, copy 7.34, next 10.07 |
| Parametre | 8 001 |

Bu koşunun checkpoint'i ayrı bir değerlendirme koşusunda (run 40816527156)
artifact'tan geri yüklenip yeniden probe edildi: korpus SHA-256'sı eşleşti ve
sayılar aynı çıktı. Yani **devam/değerlendirme yolu** da çalışıyor.

### 7.3 İki geçişin karşılaştırması

| | `lr = 3e-3` | `lr = 3e-4` |
| --- | --- | --- |
| Tüketilen | %100 (600 MB) | %100 (600 MB) |
| Eğitim NLL | 1.2178 → **1.4185** (yükseldi) | 1.4086 → **1.2034** (düştü) |
| Öğrenilen erişim | 3 935.9 → 984.6 | 8 050.5 → **7 940.0** |
| Hız | 62 722 token/s | 50 395 token/s |
| Probe (sonraki-bayt) | %3.3–4.2 | %2.7–3.6 |
| Probe (tam eşleşme) | 0/500 | 0/500 |

Bulgular (ölçümün desteklediği kadar):

1. **`lr = 3e-3` kararsızdı.** Kayıp 586 bin güncelleme boyunca düştüğü yerde
   yükseldi ve öğrenilen bellek erişimi 984 token'a çöktü. Bu, verinin değil
   adım büyüklüğünün sonucu: aynı veri, aynı mimari, `lr = 3e-4` ile monoton
   biçimde iyileşiyor.
2. **Öğrenilen dinamik gerçekten öğreniliyor.** `decay_logit` yapılandırılan
   8192'den 7940'a iniyor: model kendi erişimini ayarlıyor, ama ufku
   *koruyor*. `3e-3`'te aynı parametre kaçıp kısa belleğe çöküyor — yani
   parametre etkisiz değil, tersine fazla etkili.
3. **Prompt → yanıt hâlâ çalışmıyor.** 8 001 parametreli bayt modeli yanıt
   konumunda tek bir baytı tekrarlıyor:

   ```
   Soru: zırdüş nedir? Cevap:  ->  kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
                                    (beklenen: zozraşvö)
   Soru: hontız nedir? Cevap:  ->  kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
                                    (beklenen: ratfısstör)
   ```

   Tam eşleşme 0/500; sonraki-bayt doğruluğu %3.3, şans düzeyi %0.39. Yani
   model şansın ~8 katı, ama üretim çöküyor. Bu iki geçişin sınırları §8'de
   ölçülerek ayrıştırıldı; sıradaki deney orada.

### 7.4 Duman testi (her push, 2 MB)

500 satırlık probe ile uçtan uca boru hattı: korpus üretimi + doğrulama, 1
oturum eğitim, filtre, probe, ask, rapor. Amaç sayı üretmek değil, boru hattının
her adımının çalıştığını göstermek.

### 7.5 Yeniden üretim

Her koşu `prompt-response-600mb-evidence` artifact'ini üretir (korpus manifesti,
oturum raporları, `probe.log`, `ask.log`, `training-summary.json`,
`evidence.log`). Sayılar ayrıca **annotation** olarak yayınlanır, çünkü bu
ortamda artifact deposu okunamıyor; sonuç kaybolmasın diye.

## 8. Kalite geçişi: iki geçiş neden başarısızdı, ne değişti

§7'deki iki tam geçiş 600 MB'ı tüketti ve kaybı düşürdü, ama tek bir prompt'a
bile doğru yanıt üretmedi. Bu bölüm o sonucu ölçerek ayırıyor: eksik olan veri
miktarı değil, **kapasite ve güncelleme sayısı**.

### 8.1 Teşhis

Üç ayrı ölçüm, üç ayrı sınırı gösteriyor:

1. **Pencere soru iskeletinden kısaydı.** Korpusun soru biçimi
   `Soru: <özne> nedir? Cevap: `; altı baytlık bir özne için bu **27 bayt**.
   §7'deki iki gerçek geçiş `--local-window 16` ile koştu, yani model yanıtın
   ilk baytını üretirken özne *çoktan* yerel dikkat penceresinin dışındaydı.
   Yerel kanal bağlamı taşıyamıyordu; geriye yalnızca iki yinelemeli kanal
   kalıyordu ve 8 001 parametre onlara yetmiyordu.
2. **Model 8 001 parametreydi** (genişlik 16, ffn 32). 2 048 olgu bağını
   tutmak için bu, ölçümle de görülen büyüklükte bir eksik.
3. **Güncelleme sayısı azdı.** Nova-1 kuralı RMS ile normalize edildiği için
   her güncelleme her parametreyi yaklaşık `lr` kadar oynatır; ilerleme
   *token* sayısından çok *güncelleme* sayısıyla sınırlıdır.

### 8.2 Kalibrasyon geçişi (run 40857056713, commit `df8207a`)

Uzun bir geçiş harcamadan önce şekli ölçen kısa koşu: 20 MB korpus, 64 olgu,
genişlik 64 / ffn 192 (86 449 parametre), fiziksel pencere 32, 512 token'lık
güncelleme, `lr = 3e-4`, 20 dakika.

| Ölçüm | Değer |
| --- | --- |
| Korpus | 20 000 000 bayt, 34 275 belge, 64 olgu, `sha256 ed1f6dfd…fbe4` |
| Tüketilen | 9 582 080 token, 2 oturum |
| Hız | 7 985 token/s |
| Parametre | 86 449 |
| Eğitim NLL | 0.9222 → **0.6369** |
| Öğrenilen erişim | 8 197.2 → 8 200.8 |
| Probe tam eşleşme | fact 0/64, copy 0/200, next 0/100 |
| Probe sonraki-bayt | fact 0.048, copy 0.062, next 0.020, genel 0.052 |

Kayıp, "çerçeve + dolgu öğrenildi, içerik rastgele" tabanının altına indi:
yani model sözcüklerin *biçimini* öğrendi (hece yapısı, UTF-8 devam baytları),
ama **hangi** sözcük olduğunu taşımadı. 20 dakikada ~19 bin güncelleme vardı;
bu, RMS ile normalize edilmiş adım bütçesinde dağılımdan yeni çıkılan nokta.

### 8.3 Kalite geçişi

Değişenler — hepsi ölçülebilir, hiçbiri gizli değil:

| | §7 geçişleri | Kalite geçişi |
| --- | --- | --- |
| Parametre | 8 001 | **86 449** |
| Genişlik / ffn | 16 / 32 | **64 / 192** |
| Yerel pencere | 16 (soru iskeletinden kısa) | **48** |
| Olgu bağı | 2 048 | **32** |
| Güncelleme başına token | 1 024 | **256** |
| Öğrenme oranı | 3e-4 | 3e-4 |

Borç kalan yan: korpus 600 MB olarak aynı üreticiden geliyor, ama koşu süre
bütçesiyle sınırlı; tüketilen bayt miktarı raporda açıkça yazıyor ve
"tamamlandı" denmiyor.

### 8.4 Kalite geçişinin sonucu (run 40860680347, commit `d5ba77a`)

| Ölçüm | Değer |
| --- | --- |
| Tüketilen | 118 847 232 token, 10 oturum, 8 428 token/s |
| Parametre | 86 465 |
| Eğitim NLL (oturum ortalamaları) | 0.4096 → **0.1557** |
| Probe (o koşuda) | fact tam 0/32, sonraki-bayt 0.034, nll **14.316** |

Kayıp 0.1557, yani **perplexity 1.17**. Girdisine hiç bakmayan bir model bu
kaybı tutturamaz; buna karşılık probe'un ölçtüğü yanıt-baytı NLL'si 14.3. İki
sayı aynı korpusta, aynı ağırlıklarla ölçüldü; aralarındaki fark modelin
kendisinden çok **ölçüm yolundan** geliyordu.

### 8.5 İki yanlış hipotez ve gerçek hata

Sırayla iki hipotez kuruldu ve ikisi de ölçümle sınandı:

1. **"Soğuk durum" hipotezi.** Eğitim her baytı metin tüketmiş bir durumdan
   tahmin eder; `probe` ve `ask` ise durumu sıfırlayıp soruyu akışın en başına
   koyar. Bellek kanalları orada boş, eğitimde doludur ve kanal karışımı
   girdiden bağımsız bir dışbükey birleşimdir. Bunun için `--warmup-bytes`
   eklendi: sorudan önce N bayt korpus akıtılır.
   **Ölçüm hipotezi çürüttü:** 512 baytlık ısınma NLL'yi 14.316 → **13.866**
   düşürdü, sonraki-bayt doğruluğu 0.023'te kaldı (run 40896847438).

2. **Gerçek hata: soru tamponunun dolgusu.** `respond` ve `score_continuation`
   soruyu `len(prompt)` ile yürüyordu; oysa `prompt` sabit uzunlukta bir tampona
   bağlı (ask için 1024, probe satırı için 8192 bayt). Yani model her sorudan
   sonra **yüzlerce dolgu boşluğu** okuyup devam ediyordu. Belirtiler:

   - üç farklı soru **bayt bayt aynı** cevabı veriyordu (birkaç yüz boşluktan
     sonra soru pencereden çıkıyor),
   - üretilen baytlar (0x60, 0xE0 aralığı, `R`) korpusta **hiç geçmiyor**;
     0.1557 kaybına inmiş bir modelin bunları üretmesi mümkün değil,
   - eğitim NLL'si 0.1557 olan kontrol noktası probe'da 14.3 veriyordu.

   Düzeltme `len_trim`; eğitim etkilenmedi (blokları dosyadan tam okuyor).
   **Bu deponun bu motor için yayımladığı bütün prompt → yanıt sayıları bu
   hatalı yoldan üretildi**, yani o sayılar modeli değil hatayı ölçüyordu.

`len_trim`'li ara ölçüm (run 40899100386) NLL'yi 14.316 → 5.890'a indirdi ve
her sorunun farklı cevap üretmesini sağladı, ama bu sayılar **hâlâ** hatalı
yoldan geldi (§8.6'daki 2 ve 3 numaralı hatalar o zaman henüz bulunmamıştı), bu
yüzden hiçbiri burada sonuç olarak kullanılmıyor.

Şablon yerleşimi hakkında "şablonu öğrendi, bağı öğrenmedi" sonucu da o hatalı
araçla çıkarılmıştı. Aynı kontrol noktası, tam düzeltilmiş araçla yeniden
ölçüldü — **run 40923834013 (commit `32dccb9`, eğitim yok)**:

| Ölçüm (şablon yerleşimi, 118 847 232 token) | Değer |
| --- | --- |
| `fact` tam eşleşme | 5/32 |
| `fact` sonraki-bayt | **0.851**, NLL 0.748 |
| `copy` tam eşleşme / sonraki-bayt | 0/200 / 0.173 (NLL 8.936) |
| `next` tam eşleşme / sonraki-bayt | 0/100 / 0.035 (NLL 11.010) |
| Genel | tam eşleşme 0.015, sonraki-bayt 0.223 |

Yani model şablon yerleşiminde de bağı **büyük ölçüde** öğrenmiş: yanıt
baytlarının %85'i doğru, 32 bağın 5'i harfi harfine doğru, kalanlar tek baytlık
sapmalarla (`lakkokvir` ↔ `laskokvir`, `lesnuknen` ↔ `desnuknen`). Buna karşılık
bağlamdan kopyalama (`copy`, görülmemiş değerler) %17 ve harf kaydırma kuralı
(`next`) %3.5 ile başarısız: model **ezberlediği** bağı getirebiliyor,
**bağlamda verilen** yeni bir diziyi kopyalayamıyor. İkisi ayrı yetenek ve ölçüm
ikisini ayırıyor.

Regresyon koruması davranışsal: `tools/verify_prompt_response_corpus.py
--ask-log FILE`, bir ask günlüğünde bütün soruların aynı cevabı verdiğini
görürse düşer; iş akışı bunu ask adımından sonra çalıştırır ve ne ayrıştırdığını
da yazar.

### 8.6 Bağ yerleşimi ve üç çıkarım hatası

Şablon yerleşiminde bağ öğrenilebiliyordu ama yavaş ve eksik: yanıt baytlarının
%85'i doğruyken tam eşleşme 5/32'de kalıyordu, çünkü kaybın çoğu şablonlardan
geliyor ve bağ üzerindeki gradyan seyreltik kalıyor. `--layout bindings` korpusu
yalnızca bağ satırlarından üretir:

```
S: <özne> -> <yanıt>
```

Çerçeve 7 bayt; beş baytlık bir özne için yanıt özneden 12 bayt sonra başlar ve
akışın yaklaşık yarısı, ancak bağı arayarak tahmin edilebilecek içeriktir.
Şablon yerleşimi değişmedi ve kayıtlı hash'leri bayt bayt aynı üretiyor
(20 MB / 64 olgu → `ed1f6dfd…`).

**Run 40899844910 (commit `935002c`, 100 dk, taze model):** 51 069 184 token,
4 oturum, 8 512 token/s, kayıp 0.2509 → 0.1960 → 0.1769 → **0.1670**,
86 465 parametre; öğrenilen erişim 3 969.8 → 1 110.2. Bu koşunun raporladığı
probe sayıları (tam eşleşme 0/32, sonraki-bayt 0.000) hâlâ hatalı çıkarım
yolundan geldi; aşağıdaki üç hata düzeltildikten sonra aynı ağırlıklar ölçüldü.

Sırayla bulunan üç hata — hepsi **çıkarımda**, eğitimde değil:

1. **Dolgu.** `respond`/`score_continuation` soruyu `len(prompt)` ile
   yürüyordu; `prompt` sabit uzunlukta bir tampona bağlı olduğu için model her
   sorudan sonra yüzlerce dolgu boşluğu okuyordu. Sonuç: bütün sorular aynı
   cevabı veriyordu.
2. **Sondaki anlamlı boşluk.** `len_trim(prompt)` çözümü de yanlıştı: bu
   korpustaki her çerçeve bir boşlukla bitiyor (`"… Cevap: "`,
   `"S: <özne> -> "`) ve `len_trim` tam o boşluğu atıyor. Model doğru şekilde
   boşluk tahmin ediyor, ölçüm bunu yanlış ilk yanıt baytı sayıyordu. Çözüm:
   uzunluk **dizgenin yanında** taşınır (`split_row` alan uzunluğunu,
   `get_command_argument` argüman uzunluğunu verir).
3. **İndeks kayması.** `argmax_token` ve `sample_token` logit indeksinden 1
   çıkarıyordu; oysa motor bir token'ı `token_embedding(:, token)` ile gömer ve
   kayıp `dlogits(token)` türevini alır, yani **indeks = bayt değeri**. Bu
   yüzden üretilen her bayt modelin seçtiğinin bir eksiğiydi ve
   `argmax == target` hiçbir zaman doğru olamıyordu.

Bu üçü, ölçümün kendi kendisiyle çelişmesinden yakalandı: aynı logit dizisi
yanıt baytları için **0.002** NLL (yani neredeyse kesin ve doğru) verirken
sonraki-bayt doğruluğu 0.000 ve tam eşleşme 0/32 idi. Doğru bir olasılık
dağılımı ile sıfır isabet aynı dizi için aynı anda doğru olamaz.

**Düzeltilmiş ölçüm — run 40921692429 (commit `cacf09c`, aynı ağırlıklar, eğitim
yok):**

| Ölçüm | Değer |
| --- | --- |
| Korpus | 600 000 000 bayt, 32 bağ, `sha256 03bde545…b9360` |
| Model | 86 465 parametre, 51 069 184 token eğitilmiş |
| Probe `fact` tam eşleşme | **32/32 (1.000)** |
| Probe sonraki-bayt | **1.000**, NLL **0.002** |
| Genel | tam eşleşme 1.000, sonraki-bayt 1.000 |

```
S: töşklüt ->   sümesar      (beklenen sümesar)
S: bördıs  ->   laskokvir    (beklenen laskokvir)
S: sıkül   ->   desnuknen    (beklenen desnuknen)
```

Yani model, eğitildiği bağların tamamını soru-cevap olarak doğru yanıtlıyor.
Bunun ne olduğu ve ne olmadığı §9'da.

Regresyon koruması: `tools/verify_prompt_response_corpus.py --ask-log FILE`
bütün soruların aynı cevabı verdiği bir günlüğü reddeder; iş akışı bunu ask
adımından sonra çalıştırır ve artık ne ayrıştırdığını da yazar.

### 8.7 Kalite geçişi: girdiye bağımlı geçit ve öğrenilebilir bağ tablosu

§8.6'daki 32/32 **ezberlenmiş** bağlar üzerindeydi: 32 bağlı korpusta model
`S: <özne> ->` istemine karşılık gelen yanıtı eksiksiz üretiyor. §8.5'teki
şablon yerleşimi ise 5/32'de kalıyordu. İki fark var ve ikisi de bu geçişte
ele alındı:

1. **Bağ tablosu 2 048 satırdı.** 86 bin parametreli bir model 2 048 satırlık
   bir tabloyu tutamaz; 32 satırlık bir tablo ise tutulabilir. Bu yüzden bu
   geçiş `--facts 32` ile üretilen korpusu kullanır.
2. **Kanal karışımı girdiden bağımsızdı.** Hangi kanaldan ne kadar
   okunacağını söyleyen üç kanal lojistiği ve dört ölçek lojistiği, modelin
   *sabit* parametreleriydi: bir adım "şimdi yerel pencereye bakmalıyım" ya da
   "şimdi yinelemeli bellekten okumalıyım" diyemiyordu. Commit `bf86b75` bu
   lojistiklere girdinin öğrenilmiş bir izdüşümünü ekler
   (`channel_gate_proj`, `scale_gate_proj`); davranış sıfır ağırlıkta eski
   statik lojistiklerle birebir aynıdır, yani bu bir yeniden yorumlama değil
   eklemedir. Ayrıntı ve gradyan türetmesi
   [`FIXED_BUDGET_CONTEXT_ENGINE.md`](FIXED_BUDGET_CONTEXT_ENGINE.md) §2 ve §4.

Buna ek olarak, bu optimizasyon kuralı RMS ile normalize edildiği için ilerleme
token sayısından çok **güncelleme sayısıyla** sınırlı; bu yüzden güncelleme
boyu 512'den 256'ya indirildi (aynı duvar saati, iki kat güncelleme) ve fiziksel
pencere 16'dan 48'e çıkarıldı, yani özne artık yerel pencerenin içinde.

**Sıcak durum kontrolü (negatif sonuç, kayda geçiyor).** "Belki bağlam yok
değil, ama soğuk başlangıç yüzünden kullanılamıyor" hipotezi ölçüldü: aynı
şablon kontrol noktası, probe koşusundan önce duruma 512 bayt korpus
akıtılarak yeniden puanlandı (run 40896847438: `warmup_bytes` 0 → 512, eğitim
yok). Fark yok: `fact` NLL 14.316 → 13.866, sonraki-bayt 0.023 → 0.023,
tam eşleşme 0/128 → 0/128. Yani sorun durumun soğuk olması değildi.

**Run 40927789891 (commit `816a4cc`, talep dosyası `ci/prompt-600mb.json`,
225 dakika, taze model):**

| Ölçüm | Değer |
| --- | --- |
| Korpus | 600 000 000 bayt, 1 025 843 belge, 32 bağ, `sha256 8061d786…32e2` |
| Tüketilen | **160 455 168** token (%26.7), 9 oturum, 11 886 token/s |
| Parametre | 86 913 (geçit +448) |
| Eğitim NLL (oturum ortalamaları) | 0.3591 → 0.2510 → 0.2067 → 0.1833 → 0.1677 → 0.1568 → 0.1488 → 0.1425 → **0.1376** |
| Öğrenilen erişim | 5 499.1 → 4 934.7 token |
| Probe `fact` | tam eşleşme **32/32 (1.000)**, sonraki-bayt **1.000**, NLL **0.006** |
| Probe `copy` | 0/200, sonraki-bayt 0.190, NLL 9.013 |
| Probe `next` | 0/100, sonraki-bayt 0.033, NLL 11.468 |
| Genel | tam eşleşme 0.096, sonraki-bayt 0.251 |

`ask` adımının günlüğü (greedy, `--max-new 32`), üç ayrı soru:

```
Soru: töşklüt nedir? Cevap:  ->  sümesar      (beklenen sümesar)
Soru: bördıs  nedir? Cevap:  ->  laskokvir    (beklenen laskokvir)
Soru: sıkül   nedir? Cevap:  ->  desnuknen    (beklenen desnuknen)
```

Yani model, **kendi eğitim korpusundaki 32 bağın tamamını**, soru cümlesi
biçiminde ve tek bir sıfırdan eğitimle doğru yanıtlıyor; üç soru da tam
eşleşiyor, greedy ile örneklemeli çıktı aynı. §8.5'te "şablon yerleşiminde
5/32'de kalıyor" diye kaydedilen sınır, bu yapılandırmada kalktı.

**Ne ölçülmedi ve neden.** Bu geçişte dört şey birlikte değişti: bağ sayısı
(2 048 → 32), kapasite (8 001 → 86 913), fiziksel pencere (16 → 48) ve kanal
geçidi. Bu yüzden **32/32'yi tek başına geçide atfetmek yanlış olur**: ablasyon
(aynı şekil, geçit yok) koşulmadı. Ölçülen şey paketin kendisidir: bu bütçede
şablon yerleşiminde soru-cevap öğrenilebiliyor.

**Bağlamdan hesaplama hâlâ yok.** `copy` ailesi görülmemiş bir değeri
kopyalamayı ister ve 0/200'de kaldı; sonraki-bayt 0.173 → 0.190 arttı, bu
gürültü düzeyinde bir kıpırdama, bir yetenek değil. `next` (harf kaydırma)
0.033'te. (Bu iki aile farklı yapılandırmalar arasında karşılaştırılıyor:
önceki ölçüm 2 048 bağlı, 8 001 parametreli, 16 pencereli bir modelindi; o
yüzden 0.173 → 0.190 farkı bir kazanç olarak değil, **aynı büyüklük
mertebesinde kaldı** diye okunmalı.) Model, **kendi tablosundaki** bağı getiriyor;
**bağlamda verilen yeni bir diziyi** kullanamıyor. §9 bunu olduğu gibi yazıyor.

### 8.8 Bağlam koşullu yönlendirici

§8.7'deki geçit lojistiği **yalnızca o anki belirtece** bakıyordu:
`channel_logit(c) + channel_gate_proj(c,:)·x`. Bu, "bu belirteç yuvalara
benzeyen bir belirteç" demeyi sağlar; "yerel pencere az önce bir tekrar
buldu, oradan oku" demeyi sağlamaz, çünkü geçit kanalların ne döndürdüğünü
görmez. Bu geçişte kanal karışımına **bağlam koşullu** bir terim eklendi:

```
lojistik_eff(c) = channel_logit(c) + channel_gate_proj(c,:)·x
                + Σ_{s,k} channel_router(c, (k−1)·3 + s) · context(k, s)
```

`channel_router` `3 × 3·width` boyutunda ve **sıfırdan** başlar: sıfırda
davranış §8.7 ile birebir aynıdır (sözleşme testi bunu bit düzeyinde
doğrular; 17 sonlu fark sondasının ikisi artık yönlendirici sütunlarıdır ve
sondadan önce yönlendiricinin üç elemanına sıfırdan farklı değer verilir,
aksi halde kayıp yönlendiriciye göre doğrusal kalır ve testin konusu olan
bağlam türevi sınanmaz).

**Neden bu, `copy`/`next` için bir umut.** Bu iki aile "bağlamda verilen
diziyi kullan" diyor. Modelin bunu yapabilmesi için kararın **kanalların
çıktısına** bağlı olması gerekir: yerel pencere eşleşmeyi bulduğunda
yönlendirici yerel kanalın ağırlığını artırabilir. `channel_gate_proj` bunu
yapısal olarak yapamaz, `channel_router` yapabilir.

**Maliyet ve sözleşme.** `9·width` parametre (86 913 → 87 489 at width 64),
belirteç başına `18·width` ek işlem; ek ayırma yok, belirteç başına `exp` yok,
sınırlar `width` ve sabit 3 (ufuk yok). Gradyan iki yönlü: yönlendirici
sütunları `dL/dlojistik · context`, üç bağlam vektörü ise
`dot(dL/dlojistik, yönlendirici sütunu)` kadar ek gradyan alır. Kontrol
noktası **sürüm 4**; sürüm 3 (bugün diskte bulunması en olası biçim) ve
altındaki dosyalar dönüştürülmeden `FBT_ERR_VERSION` ile reddedilir, bu yüzden
bu geçiş **sıfırdan** eğitilir, önceki koşunun kontrol noktasından devam
etmez.

**Sonuç (bu bölüm ölçüm geldiğinde doldurulacak; şu an yalnızca mekanizma
kayıtlı).** Aynı şekil, aynı bütçe: 32 bağ, `width 64`, 4 kafa, `ffn 192`,
yerel pencere 48, güncelleme boyu 256, `lr 3e-4`, 225 dakika. Ölçüm
sonuçları geldiğinde §8.7'deki tabloyla **yan yana** yazılacak: `fact`
(ezber), `copy` (bağlamdan kopyalama), `next` (bağlamdan kural). Bu bir
ablasyon değildir: geçit + yönlendirici paketi, tek bir koşu.

## 9. Sınırlar

- Korpus **sentetiktir**; model bu veriyle Türkçe "öğrenmez", bu korpusun
  istatistiğini öğrenir.
- §8.6 ve §8.7'deki 32/32 (ve §8.7'deki 32 bağ, şablon yerleşimi), **32 yapay
  bağın** öğrenildiğini gösterir: `S: <özne> ->` ya da `Soru: <özne> nedir?
  Cevap: ` istemine karşılık gelen yanıtı üretmek. Bu bir dil yeteneği değil,
  ölçülebilir bir ezber/arama yeteneğidir; 2 048 bağlı korpusta ve 8 001
  parametreli modelde aynı şey ölçülemedi (§7, §8.5).
- Ezberlenen bağı getirmek ile **bağlamda verilen** yeni bir diziyi kullanmak
  ayrı yetenekler ve ikincisi bu bütçede yok: `copy` (görülmemiş değeri kopyala)
  sonraki-bayt 0.190, tam eşleşme 0/200; `next` (harf kaydırma) 0.033, 0/100
  (§8.7). Bu, mimarinin 87 bin parametrede bağlamdan hesaplama yapamadığını
  gösterir; ölçüm bunu olduğu gibi raporlar.
- §8.7'deki kazanç **tek bir değişkene atfedilemez**: bağ sayısı, kapasite,
  fiziksel pencere ve kanal geçidi aynı geçişte değişti. Ablasyon (aynı şekil,
  geçit yok) koşulmadı; "geçit 32/32'yi getirdi" cümlesi bu belgede **yok**.
- 600 MB'ın **tamamı** hiçbir koşuda tüketilmedi: bağ yerleşiminde 51 069 184
  bayt (%8.5), şablon kalite geçişinde 160 455 168 bayt (%26.7). 32 bağlı
  korpusta kalan bayt aynı tabloyu yineliyor; oran yine de açıkça yazıyor ve
  hiçbir yerde "600 MB tamamlandı" denmiyor.
- §7 ve §8.5'te raporlanan bütün "prompt → yanıt" sayıları, yukarıdaki üç
  hata nedeniyle **modeli değil ölçüm yolunu** ölçüyordu.
- 600 MB tek bir koşuda tüketilemeyecek kadar uzunsa koşu **kaldığı yeri
  söyler** ve zincirleme devam eder; hiçbir yerde "tamamlandı" denmez.
- Bayt düzeyi modelleme, 2048 olgu bağını küçük bir durumda tutmayı
  gerektirir; `fact` doğruluğunun düşük çıkması mimarinin değil, bütçenin
  sonucudur ve ölçüm bunu olduğu gibi raporlar.
- `next` görevi (harf kaydırma) bir *kural*dır; örneklenen 100 satır
  ezberlenemez, ancak çok küçük bir model kuralı öğrenemeyebilir.
