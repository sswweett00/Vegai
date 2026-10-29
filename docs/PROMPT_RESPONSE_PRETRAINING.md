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

Düzeltmeden sonra aynı kontrol noktası (run 40899100386):

| Ölçüm | Bozuk yol | Düzeltilmiş yol |
| --- | --- | --- |
| fact NLL | 14.316 | **5.890** |
| copy NLL | 15.813 | 12.906 |
| next NLL | 19.249 | 17.067 |
| Aynı cevap / farklı soru | evet | hayır (her soru farklı) |

Regresyon koruması davranışsal: `tools/verify_prompt_response_corpus.py
--ask-log FILE`, bir ask günlüğünde bütün soruların aynı cevabı verdiğini
görürse düşer; iş akışı bunu ask adımından sonra çalıştırır (run 40899100386 bu
kapıya takıldı, çünkü cevaplar boştu).

Düzeltmeden sonra tablo ne söylüyor: model **şablonu** öğrenmiş, **bağı**
öğrenmemiş. Yanıt baytlarının NLL'si 5.89 ≈ ln 256 = 5.545, yani yanıt
pozisyonunda dağılım düzgüne yakın. Bunun nedeni ölçülebilir: şablon
yerleşiminde içerik baytları akışın ~%20'si, geri kalanı sabit metin; ortalama
kayıp şablonlarla düşüyor ve bağ üzerindeki gradyan seyreltik kalıyor.

## 9. Sınırlar

- Korpus **sentetiktir**; model bu veriyle Türkçe "öğrenmez", bu korpusun
  istatistiğini öğrenir.
- 600 MB tek bir koşuda tüketilemeyecek kadar uzunsa koşu **kaldığı yeri
  söyler** ve zincirleme devam eder; hiçbir yerde "tamamlandı" denmez.
- Bayt düzeyi modelleme, 2048 olgu bağını küçük bir durumda tutmayı
  gerektirir; `fact` doğruluğunun düşük çıkması mimarinin değil, bütçenin
  sonucudur ve ölçüm bunu olduğu gibi raporlar.
- `next` görevi (harf kaydırma) bir *kural*dır; örneklenen 100 satır
  ezberlenemez, ancak çok küçük bir model kuralı öğrenemeyebilir.
