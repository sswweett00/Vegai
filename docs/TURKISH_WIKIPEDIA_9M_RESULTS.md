# Ölçülmüş gerçek Türkçe Wikipedia 9M native-token sonuçları

Bu kayıt, başarılı immutable Actions artifact’indeki `experiment-report.json`
dosyası doğrulanarak üretildi. Kaynak koşusu ve PR’ye yayımlanan makinece
doğrulanmış görünüm aşağıdadır:

- Actions run: [40154058047](https://github.com/sswweett00/Vegai/actions/runs/40154058047)
- Doğrulanmış PR kaydı: [issuecomment-6392485135](https://github.com/sswweett00/Vegai/pull/1#issuecomment-6392485135)
- Commit: `112ff64cf2d0f75b8e6493d4030ec03957ede260`

Aşağıdaki 11 semantic satır trained native checkpoint’in self-excluded
nearest-neighbour çıktısıdır. 11 yanıt ise ayrı MIT kaynaklı TSV’den
`vegai_chat` tarafından verbatim geri alınmıştır; LLM veya serbest metin üretimi
değildir. Eğitim bir RESONANT başlangıç epoch’udur; önceki epoch’tan öğrenilmiş
semantic feedback veya held-out başarı iddia edilmez.

## Verified real Turkish Wikipedia 9M-native-token evidence

Source workflow: [40154058047](https://github.com/sswweett00/Vegai/actions/runs/40154058047)

### Provenance and strict corpus gate

- Training source: `https://github.com/tdd-ai/trwiki-67` at revision `d22d071c99ff3a6ddd2de1bc412fc7b8f4ca1c02`; declared license `CC-BY-SA-4.0`.
- Training-source LICENSE SHA-256: `87a816969906840bf7af8d4d01cdfad4741b18946365e1f286007935509f2edb`; downloaded archive SHA-256: `cdb7bffaaa4e0891328f4275e0d8d50923afde85602266b1f224e5d4aa6b323a`; raw split SHA-256: `3e20515dc6cb8b5a556235e54d97354a2119309d59470133288744e05d1671a0`.
- Normalization: `tr-ascii-v1`; vocabulary selection: `full-source frequency rank descending then ASCII lexical tie-break`, `8,192` active source tokens.
- Normalized output: exactly **9,000,000** native tokens in `2,322,801` LF records; SHA-256 `cf21e4a7e2ce0de35c8a800c1731313358e510db4bf49a71dc89cd8cc3088bef`.
- Record contract: 2–128 tokens; non-ASCII, CRLF, overlong-record, and tokenizer-truncation counters are all zero.
- Separate grounded TSV source: `https://github.com/Aliumutcan/Turkish_NLP_FAQ_DATASET` at revision `3ff5a943569883d86a4b63ecd88fbfc13891f9f7`; declared license `MIT`.
- FAQ LICENSE SHA-256: `b4e02136739ad38a81faf23943447e13ee62bbdc0bf4e026f9c0c987bd985264`; FAQ dataset SHA-256: `b9ac1d3f9ef5469c85e19126876d3cfce5c4dd1c14b119890e64e773e8ae5167`.

### Native Fortran training

- Checkpoint SHA-256: `88f9370e7a0c53e3abc77ff251917a9335cb2ce7859190f99632bf6f067624bc`.
- Train-child wall time: **263.670648 s**; raw native throughput: **34133.492 tokens/s**. This deliberately excludes retrieval, download, normalization, compilation, and CI overhead.
- Context window: `2`; OMP threads: `4`.
- Native metrics: raw tokens `9,000,000`, active tokens `9,000,000`, prototype updates `9,000,000`, OOV `0`.
- RESONANT scope: One epoch uses the RESONANT initialization contract (identical to HYBRID for epoch one); no learned prior-epoch semantic feedback is claimed.
- Recorded training command:
  ```text
/home/runner/work/Vegai/Vegai/build-turkish-9m/vegai_pretrain train /home/runner/work/_temp/vegai-turkish-wikipedia-9m/experiment/turkish_wikipedia_9m_native_ascii.txt /home/runner/work/_temp/vegai-turkish-wikipedia-9m/experiment/turkish_wikipedia_9m_resonant.vgai --max-vocab 8192 --min-frequency 1 --window 2 --position-shift 41 --context-mode resonant --epochs 1 --strict --parallel-context-build --parallel-min-contexts 32
  ```

### 11 checkpoint semantic nearest-neighbour results

| # | query | nearest semantic neighbour | bipolar similarity |
| ---: | --- | --- | ---: |
| 1 | `turkiye` | `kume` | 0.03491 |
| 2 | `istanbul` | `kafa` | 0.03625 |
| 3 | `ankara` | `manken` | 0.03979 |
| 4 | `tarih` | `tarihce` | 0.06091 |
| 5 | `devlet` | `etkisinin` | 0.04016 |
| 6 | `universite` | `kulturunun` | 0.03430 |
| 7 | `belediye` | `olustugu` | 0.03540 |
| 8 | `sanat` | `tarihce` | 0.04529 |
| 9 | `futbol` | `ragbi` | 0.04138 |
| 10 | `kitap` | `teleskopla` | 0.04675 |
| 11 | `dil` | `terakki` | 0.04382 |

These are self-excluded nearest-neighbour values from the trained checkpoint, not generated answers or held-out semantic accuracy.

### 11 grounded TSV retrieval results

Each answer below is the exact ASCII-normalized answer field retrieved verbatim by `vegai_chat` from the separately versioned TSV; it is **not** free-form LLM generation.

1. **Prompt:** `pasaport ile ilgili is ve islemler nasil yapilir`
   - matched prompt: `pasaport ile ilgili is ve islemler nasil yapilir`
   - similarity `1.00000`; runner-up `0.23218`; margin `0.76782`; grounded `true`
   - TSV source: https://www.meb.gov.tr/sss.php (row `1`)
   - **Literal TSV answer:**
     ```text
https://www.nvi.gov.tr/sss-pasaport
     ```
2. **Prompt:** `ders secme islemi nasil yapilir`
   - matched prompt: `ders secme islemi nasil yapilir`
   - similarity `1.00000`; runner-up `0.23218`; margin `0.76782`; grounded `true`
   - TSV source: https://www.meb.gov.tr/sss.php (row `262`)
   - **Literal TSV answer:**
     ```text
Ilk kayit olan ve kayit yenileten ogrenci bir ust siniftan en fazla dort ders secebilecektir. Ders secimi yapilmadigi takdirde Acik Ogretim Ortaokulu Mudurlugunce yapilacaktir.
     ```
3. **Prompt:** `12 sinif mesleki egitim merkezi programi ogrencileri icin`
   - matched prompt: `12 sinif mesleki egitim merkezi programi ogrencileri icin`
   - similarity `1.00000`; runner-up `0.17053`; margin `0.82947`; grounded `true`
   - TSV source: https://www.meb.gov.tr/sss.php (row `403`)
   - **Literal TSV answer:**
     ```text
Mesleki egitim merkezi programi 12 nci sinif ogrencileri yukardaki aciklamalar dogrultusunda egitim ogretime baslayacaktir.
     ```
4. **Prompt:** `il ilce ozel egitim hizmetleri kurul karari hangi durumlarda alinir`
   - matched prompt: `il ilce ozel egitim hizmetleri kurul karari hangi durumlarda alinir`
   - similarity `1.00000`; runner-up `0.17456`; margin `0.82544`; grounded `true`
   - TSV source: https://www.meb.gov.tr/sss.php (row `680`)
   - **Literal TSV answer:**
     ```text
Ozel egitim ihtiyaci olan bireylerin uygun egitim kurumuna yerlestirilmesi surecinde Ozel Egitim Degerlendirme Kurulu Raporu dogrultusunda il/ilce ozel egitim hizmetleri kurulu karari alinmasi gerekmektedir. Bu kapsamda;. Resmi bir okul veya kuruma ilk kez yapilacak yerlestirmede, . Kademeler arasi gecislerde bir ust kademeye yapilacak yerlestirmede, . Bulundugu kademe icinde yeni duzenlenen Ozel Egitim Degerlendirme Kurulu Raporu dogrultusunda yapilan yonlendirmeye uygun egitim ortamina yerlestirme de il/ilce ozel egitim hizmetleri kurulu karari alinmasi gerekmektedir.
     ```
5. **Prompt:** `evde egitim hizmeti kapsaminda haftalik ders saatlerinde degisiklik yapilabilir mi`
   - matched prompt: `evde egitim hizmeti kapsaminda haftalik ders saatlerinde degisiklik yapilabilir mi`
   - similarity `1.00000`; runner-up `0.20740`; margin `0.79260`; grounded `true`
   - TSV source: https://www.meb.gov.tr/sss.php (row `690`)
   - **Literal TSV answer:**
     ```text
Ozel egitim Hizmetleri Yonetmeliginin 14 uncu maddesi 2 nci fikrasinin (a) bendinde Ogrencinin evde egitim hizmeti kapsaminda alacagi haftalik ders saati, sorumlu oldugu egitim programi esas alinarak ilkogretim kademesinde veya ozel egitim programi uygulanan ortaogretim kademesinde bir okula kayitli olanlar icin haftada 10 ders saatinden, diger ortaogretim kademesinde bir okula kayitli olanlar icin ise haftada 16 ders saatinden az olmayacak sekilde planlanir." hukmu ile ayni maddenin 2 nci fikrasinin (b) bendinde "Ilkogretim programini takip eden ogrenciler icin okutulacak dersler belirlenirken merkezi sistem sinavlarinda sorumlu olacaklari dersler; ortaogretim programlarini takip eden ogrenciler icin okutulacak dersler belirlenirken ise ogrencinin sectigi ders yogunlugu dikkate alinarak planlama yapilir." hukmu yer almaktadir.Bu cercevede, ozel egitim ihtiyaci olan ogrencilerin bireysel ozellikleri ve egitim ihtiyaclari dogrultusunda haftalik ders cizelgesinde yer alan derslerden ilkogretim kademesinde veya ozel egitim programi uygulanan ortaogretim kademesinde bir okula kayitli olanlar icin haftada 10 ders saatinden, diger ortaogretim kademesinde bir okula kayitli olanlar icin ise haftada 16 ders saatinden az olmayacak sekilde il veya ilce ozel egitim hizmetleri kurulunun planlamayi yapmasi gerekmektedir.
     ```
6. **Prompt:** `cocuk kuluplerinde ne tur etkinlikler yapilir`
   - matched prompt: `cocuk kuluplerinde ne tur etkinlikler yapilir`
   - similarity `1.00000`; runner-up `0.18738`; margin `0.81262`; grounded `true`
   - TSV source: https://www.meb.gov.tr/sss.php (row `769`)
   - **Literal TSV answer:**
     ```text
Cocuk kuluplerinde yapilacak etkinlikler,Milli Egitim Bakanligi Okul Oncesi Egitim ve Ilkogretim Kurumlari Cocuk Kulupleri Yonergesinin 7'nci maddesinin birinci fikrasinda yer alan Kulup etkinlikleri, Turk milli egitimin genel amac ve temel ilkeleri ile egitim/ogretim programlarina uygun olarak yurutulur. ve ayni maddenin ikinci fikrasinda belirtilen (Degisik 10/03/2016 tarih ve 2852884 sayili Makam Oluru) Cocuk kulubu etkinliklerinde; cocuk/ogrencilerin yas ve gelisim durumlarina gore geleneksel cocuk oyunlari, halk oyunlari, guzel sanatlar, beden egitimi ve spor calismalari, degerler egitimi, yabanci dil, bilim olimpiyatlari ve proje calismalari ile gelisim alanlarini destekleyecek benzeri faaliyetler yapilir. Ancak, ilkogretim kurumlarinda yaz tatilinde acilacak kulup faaliyetlerinde gunluk iki saati gecmemek uzere egitim ve ogretimi destekleyici calismalara yer verilebilir. hukmu dogrultusunda belirlenir.
     ```
7. **Prompt:** `100 temel eser uygulamasi devam ediyor mu`
   - matched prompt: `100 temel eser uygulamasi devam ediyor mu`
   - similarity `1.00000`; runner-up `0.01660`; margin `0.98340`; grounded `true`
   - TSV source: https://www.meb.gov.tr/sss.php (row `786`)
   - **Literal TSV answer:**
     ```text
Ogrencilerimize okuma aliskanligi kazandirmak amaciyla uygulanmaya baslanan 100 Temel Eser uygulamasi Bakanligimizca yapilan detayli arastirma ve calismalar da dikkate alinarak 2018/17 No'lu Genelge ile uygulamadan kaldirilmis olup ogrencilerimizin ogretmenlerimizin rehberliginde Anayasa'ya, yasalara ve milli egitimin temel amaclarina ters dusmeyecek, milli ve insani degerlere saygili butun eserlerden yararlandirilmasi saglanmistir.
     ```
8. **Prompt:** `okula yeni kayit icin hangi bilgi belge talep edilir`
   - matched prompt: `okula yeni kayit icin hangi bilgi belge talep edilir`
   - similarity `1.00000`; runner-up `0.10522`; margin `0.89478`; grounded `true`
   - TSV source: https://www.meb.gov.tr/sss.php (row `791`)
   - **Literal TSV answer:**
     ```text
Milli Egitim Bakanligi Okul Oncesi Egitim ve Ilkogretim Kurumlari Kanununa gore kayit islemi, 25/4/2006 tarihli ve 5490 sayili Nufus Hizmetleri Kanunu hukumlerince olusturulan ulusal adres veri tabanindaki adres bilgileri esas alinarak, e-Okul sistemi uzerinden yapilir. Kayit islemleri sirasinda herhangi bir belge talep edilmez. hukmu kapsaminda yapilmis olup ogrencilerin adreslerine gore hangi okullara kayitlarinin yapildigini Bakanligimizin e-Okul Yonetim Bilgi Sistemi internet sayfasi uzerinde "https://e-okul.meb.gov.tr/" ogrencilerinin T.C. kimlik numarasi ve dogum tarihi ile sorgulama yapilabilmektedir. e-Okul Yonetim Bilgi Sisteminde ilan edilen okula kesin kayit islemi okul yonetimince yapilacaktir.
     ```
9. **Prompt:** `okullarda ders kitaplari disinda yardimci kaynak kullanilabilir mi`
   - matched prompt: `okullarda ders kitaplari disinda yardimci kaynak kullanilabilir mi`
   - similarity `1.00000`; runner-up `0.20740`; margin `0.79260`; grounded `true`
   - TSV source: https://www.meb.gov.tr/sss.php (row `801`)
   - **Literal TSV answer:**
     ```text
Okullarimizda ders kitaplari disinda yardimci kaynak kitap alinmasi ve kullanimi konusunda ogrencilerimizin zorunlu tutulmamasi gerektigi il milli egitim mudurluklerine gonderilen resmi yazilarda belirtilmistir. Okullarimiza gonderilen aciklamalarda Bakanligimizca hazirlanan ve kitap, haber, gorsel, video, ses, dergi, dokuman, yarisma, uygulama vb modullerin bulundugu, sinif seviyelerine uygun, guvenilir ve dogru e-icerikleri surekli gelistirilmeye devam eden ve ogretmen ve ogrencilerimizin hizmetine ucretsiz olarak sunulan EBA (http://www.eba.gov.tr)'ya yonlendirilmeleri onerilmistir.
     ```
10. **Prompt:** `gercek kisiler icin ek 1 formu`
   - matched prompt: `gercek kisiler icin ek 1 formu`
   - similarity `1.00000`; runner-up `0.17053`; margin `0.82947`; grounded `true`
   - TSV source: https://www.meb.gov.tr/sss.php (row `838`)
   - **Literal TSV answer:**
     ```text
Gercek Kisiler Icin EK-1 Formu
     ```
11. **Prompt:** `iphone 6 hangi ozelliklere sahip olacak`
   - matched prompt: `iphone 6 hangi ozelliklere sahip olacak`
   - similarity `1.00000`; runner-up `0.17456`; margin `0.82544`; grounded `true`
   - TSV source: https://www.ios-tr.com/sss/ (row `920`)
   - **Literal TSV answer:**
     ```text
iOS TurkiyeTabi ki yine A8'in daha hizli ve enerji konusunda daha tasarruflu olmasi bekleniyor.Gecen sene Apple bir surpriz yaparak A7 cipinde 64-bit islemci kullanmisti ve de M7 hareket es islemcisini tanitmisti. Bu sene A8 ile boyle radikal degisikliklere rastlayacagimizi dusunmuyorum.Tabi ki iPhone 6'nin Touch ID kullanacak olmasi da pek zor olmayan bir tahmin. Bunun yaninda Touch ID icin cihazin ana ekran tusunda kullanilan safir kristalin butun ekranda kullanilmasi da bekleniyor.Daha cok tasarimsal degisiklikler barindirmasi beklenen iPhone 6'da beklenen bir diger fark da kamera. 5.5 inclik modelin OIS (Optik Resim Dengeleme) ozelligine sahip olacagi soyleniyor. Ayrica genel olarak tum modellerin en azindan lenslerde ve islemcide yapilacak iyilestirmelerle daha iyi bir kameraya sahip olmasi bekleniyor.RE TARAFINDAN YAZILAN BUYUN YAZILARI GORUNTULEiOS TurkiyeiPhone, iPad ve iPod touch hakkinda her sey
     ```

> Scope: this proves the reported real-data native ingest/update/checkpoint/retrieval run only. It does not establish held-out Turkish NLP accuracy, factual correctness beyond the cited TSV, general language understanding, or general intelligence.
