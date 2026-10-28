# 99 milyar token Türkçe neural pretraining programı

> **Dürüstlük sözleşmesi:** "99 milyar" bu programın **hedefidir**, tamamlanmış
> bir koşu değildir. Bu belgede üç sayı daima ayrı yazılır: (1) bir koşuda
> gerçekten eğitilen token, (2) boru hattının kabul ettiği gerçek veri,
> (3) ölçülen hızdan türetilen projeksiyon. Hiçbiri diğeri yerine geçmez.

## 1. Neden 99B tek seferde "koşulmaz"? (kıdemli mühendis gerçekçiliği)

Bir epoch'un süresi ≈ `token_sayısı / ölçülen token-sn` dir. Vegai'nin
native neural yolu bugün tek host üzerinde, bağlı-çıktı (tied-output)
güncellemesiyle çalışır; dağıtık GPU eğitimi yoktur. Ölçülen hız ne olursa
olsun sonuç aynı kapıya çıkar:

| Ölçülen hız (tahmin tokenı/sn) | 99B için tek-host süre |
|---:|---:|
| 1.000 | ~1.147 gün (~3,1 yıl) |
| 5.125 (bu repo, koşu 40225294843) | ~224 gün |
| 10.000 | ~115 gün |
| 100.000 | ~11,5 gün |
| 1.000.000 | ~1,15 gün |

Gerçek koşunun hızı kanıt dosyasında (`evidence.json.scale_projection`)
ölçüm olarak yayınlanır; yukarıdaki tablo yalnızca büyüklük hissi içindir.
99B'yi gerçekten bitirmek; tam BPTT'li bir mimari, GPU kümesi, karma
hassasiyet ve veri-paralel eğitim gerektirir. Bu repo bunlara sahip
olduğunu iddia etmez; aşağıdaki hat, o ölçeğe **giden yolun doğrulanabilir
ilk basamağıdır**: gerçek veri + gerçek epoch + ölçülebilir hız + devam
edilebilir (resumable) checkpoint sözleşmesi.

## 2. "Aşırı yüksek kaliteli gerçek Türkçe 99B" verisi var mı?

Kamuya açık, lisans-provenance'ı izlenebilir en büyük Türkçe kaynaklar
(2026-10 itibarıyla, yayıncı sayıları):

- **FineWeb-2 `tur_Latn`**: 95,1M belge, 41,9B kelime, ~125 GB filtreli
  metin. Web ölçeğindedir; "küratörlü aşırı yüksek kalite" değildir.
- **FineWeb2-HQ `tur_Latn`**: 8,58M belge (~100 GB disk gösterimi) —
  kalite-skorlu alt küme, 99B'nin çok altındadır.
- **FinePDFs `tur_Latn`**: 1,70M belge, 15,34B token.
- **Türkçe Wikipedia**: ~630 bin madde, ~1,6 GB temiz metin. Kalitesi
  yüksek, ölçeği küçüktür.
- **COSMOS Türkçe** (bu repoda 9 GB kanıt koşusu var): CC-BY-4.0.

Sonuç: 99B token'a ancak birden çok kaynağın (web + PDF + ansiklopedi +
kitap) çakışma-bilinçli birleşimi ve sıkı kalite filtresiyle ulaşılır;
hazır "aşırı yüksek kaliteli 99B Türkçe" paketi yoktur. Bu yüzden program,
veriyi **üreten ve her aşamasını sayan** bir boru hattı olarak tasarlandı.

## 3. Bu PR'da teslim edilen sistem

### 3.1 Eğitim sistemi v2 (native Fortran)

| Alan | Önce | Şimdi |
|---|---|---|
| Optimize edici | düz SGD, sabit LR 0.025 | **AdamW** (bias düzeltme, decoupled WD) + legacy SGD modu |
| LR zamanlama | yok | warmup + inverse-sqrt / warmup + cosine / constant; update-counter'ın saf fonksiyonu |
| Düzenlileştirme | yalnız gradyan kırpma + değer zarfı | + **label smoothing** (NLL raporu daima gerçek), AdamW weight decay (yalnız embedding'lere, `lr*wd < 1` doğrulanır; SGD modunda wd yok) |
| Metrik | mean NLL | + perplexity, **top-1/top-5**, epoch token/sn |
| Checkpoint | v2, yalnız parametre+imleç | **v3**: optimizer config + Adam momentleri bit-birebir; v2 geri-uyumlu yükleme (SGD semantiği korunur) |
| Devam (resume) | moment yok | 2+checkpoint+2 ≡ one-pass **bit-birebir** (CI testi) |
| Değerlendirme | sequence API | + `evaluate` CLI (no-update dosya değerlendirmesi), `init-checkpoint` (başlangıç taban çizgisi) |
| BPE eğitimi | exact-rescan, O(merge × korpus) dosya geçişi | + `tokenizer-train-fast` in-memory apply-merge; exact-rescan ile **bit-birebir aynı merge'ler** (testli) |
| Performans | kayıt başına state realloc; token başına tam-matris finite taraması | workspace yeniden kullanımı; finite-kanıtı matematiksel sınırla (tama tarama yok) |
| CLI yapılandırma | sabit 64/4/32/256 | `--width/--heads/--window/--ffn/--lr/--optimizer/--schedule/...`; resume'da boyut değişimi reddedilir |

Kapsam değişmedi: yalnız bağlı çıktı katmanı eğitilir, çekirdek donuktur.
Bu tam BPTT değildir ve öyle sunulmaz (bkz. `NEURAL_CAUSAL_SYSTEM.md`).

### 3.2 Veri boru hattı (`tools/build_turkish_wiki_corpus.py`)

Resmî trwiki dump'ı → wikitext temizliği → Türkçe kalite filtreleri
(alfabe oranı, fonksiyon-kelime aralığı, kelime uzunluğu, tekrar, liste
benzerliği, doğrudan PII regex reddi) → exact belge ve paragraf dedupe →
belge düzeyinde SHA-bucket train/held-out ayrımı → ≤3600 karakter kayıtlar
→ SHA-256 bağlı `manifest.json` (her aşamanın sayısı ile). Aynı araç,
ileride FineWeb-2/COSMOS gibi kaynaklar için `--text-dir` ve akış
sözleşmesini koruyarak genişletilebilecek şekilde yazıldı.

### 3.3 Koşu ve kanıt (`tools/run_neural_turkish_epoch1.py`)

Tokenizer → eğitimsiz baseline → **tam 1 epoch** → held-out → 10 promptluk
üretim süiti → `evidence.json` + `RESULTS.md`. Her artefakt hash'lenir;
prompt/response'lar düzenlenmeden yazılır. `evidence.json` içindeki `config`
bloğu optimize edicinin tam kimliğini taşır: optimizer, LR, zamanlama,
`weight_decay` (kapsamı ile birlikte: AdamW'de yalnız embedding'ler, SGD'de
uygulanmaz), `adam_beta1/beta2/adam_epsilon`, `gradient_accumulation`
(bkz. §3.5) ve zamanlama ufuklarının hem token hem adım birimindeki değerleri
(`*_updates_token_units`). `--weight-decay` yalnız `--optimizer adamw` ile
kabul edilir.

### 3.4 Kalite-korumalı hız çalışması (ölçüm-güdümlü A/B)

İlk full denemesi (run 40226479496, 200 MB dilim) native `train` çağrısının
18000 sn alt-süreç timeout'una takılmasıyla başarısız oldu; tanı,
publisher'ın PR'a taşıdığı `failure.json` ile kondu. Ardından hız, kaliteyi
düşürmeden artırılmak üzere **bayt-özdeş** probe dilimi (SHA `5215e60c…`,
784.931 tahmin tokenı, aynı held-out kümesi) üzerinde A/B ile ölçüldü:

| Koşu | Değişiklik | token/sn | Held-out NLL | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|
| Baz 40225294843 | `matmul(transpose(E), h)` (token başına tam transpoze kopyası) | 5125.051 | 7.839114 | 0.008929 | 0.039559 |
| A/B#1 40301975287 | sütun-akışlı sıralı dot + paralel softmax exp + Adam bölme katlama | 2275.346 | 7.796473 | 0.008972 | 0.038877 |
| A/B#2 40305596215 | 8 bağımsız kısmi toplam zinciri | 3250.440 | 7.815675 | 0.008918 | 0.038239 |
| **A/B#3 40307841176** | **`matmul(h, E)` (transposesiz, forksuz) + `OMP_WAIT_POLICY=ACTIVE`** | **6408.916** | **7.781494** | **0.009099** | **0.040591** |

Dersler (ölçüldü, tahmin değil): bu boyutta transpoze kopyası L3'te
yaşadığından sandığımız kadar pahalı değil; asıl maliyetler (a) skaler FMA
zincirlerinin gecikme-bağımlılığı ve (b) token başına OpenMP fork-join
thread uyku/uyanma gecikmesi. Nihai paket: transposesiz vektör-matris
matmul, bit-birebir paralel softmax üstelleri (sıralı normalizatör aynı),
Adam moment yinelemeleri bit-birebir kalırken proposal bölmelerinin
döngü-değişmez çarpanlara katlanması, rotary sin/cos ve local-attention
skorlarının bir kez hesaplanması, aktif OpenMP bekleme politikası.
Sonuç: **+%25,1 hız, held-out kalitesinde kayıp yok** (NLL −%0,74 göreli
iyileşme; farklar FP yörünge gürültüsü bandındadır, üstünlük iddia edilmez).
Yeni logits yolu, birim testte transpoze-matmul referansıyla 1e-4 tolerans
ve aynı argmax ile kapılanır.

Dayanıklılık: runner artık epoch'u N checkpoint'li `train-chunk` çağrısına
bölebilir (aynı online sıra/config; timeout yalnız uçuşan parçayı kaybettirir,
ilerleme `chunks.json`'da görünür; verifier mikro koşuda tek-çağrı ile aynı
tahmin-tokenı tüketimini ve tam kaynak-kayıt kapsamasını denetler).

### 3.5 Güncelleme birikimi (mini-batch) — darboğazın ölçülmüş yeri

Ölçüm (run 40307841176, vocab 4096 × width 96, 4 thread): eğitim süresinin
~%80'i `tied_output_update` döngüsünde geçiyordu. O döngü **her tahmin
tokenında** bağlı (tied) embedding matrisini, iki AdamW moment matrisini ve
iki bias moment vektörünü baştan sona akıtıyor: token başına ~17 MB
L3-yerleşik trafik.

İlk hipotez "bellek trafiği darboğaz" idi; ölçüm bunu çürüttü. 32'lik
birikim tek başına yalnız **1,13x** getirdi (3390 → 3833 token/sn, aynı
uzunlukta held-out NLL −0,497, yani batch lehine). Sebep: eleman başına
aritmetik birikimden bağımsızdır — AdamW zinciri her eleman için bir
`sqrt` ve bir bölme demek ve bu, 4-genişlikte real64 ile yapılıyordu. Bu
yüzden `tied_output_update`'in eleman zinciri (moment özyinelemeleri,
`sqrt`, bölme, clamp) **parametre duyarlılığında (real32)** çalışacak şekilde
değiştirildi; momentler de artık real32 saklanıyor (yaygın çerçeveler
`exp_avg`/`exp_avg_sq`'yi parametre dtype'ında tutar). Gradyan toplamı
real64 kalır. Sonuç, tek başına birikimden çok daha büyük: aşağıdaki ölçüm.

Ek olarak batch çekirdeğinin döngü sırası düzeltildi: eskiden her eleman
için batch en içte koşuyordu (B tane bağımlı skaler FMA zinciri, vektörleşmeye
kapalı); şimdi sütun dış birim, batch üyesi başına contiguous `width`
vektörü üzerinde tek bir axpy, ardından sütuna tek vektörel geçiş.

`--gradient-accumulation N` (kütüphanede `neural_train_config_t%gradient_accumulation`)
N tahmin tokenının **güncelleme öncesi parametrelerde** hesaplanmış tam
gradyanlarını ortalar ve tek optimizer adımında uygular — yani sıradan
mini-batch SGD/AdamW. Toplamak değil *ortalamak* şart: aksi hâlde 40 tokenlık
bir kayıt, 1 tokenlık kayıttan 40 kat büyük adım atardı.

Sözleşmeler korunur:

- **Batch kayıt sınırını aşmaz.** Her kayıt sonunda kalan kısmi batch
  flush edilir; her checkpoint sınırı zaten kayıt sınırıdır, dolayısıyla
  `2 + checkpoint + 2` akışı tek-geçiş akışıyla **bit-birebir** aynıdır
  (native test `test_batched_optimizer_contract`, AdamW momentleri dâhil).
- **N = 1 varsayılandır** ve tam olarak eski token-başına yolu çalıştırır;
  mevcut kanıt/test sözleşmeleri değişmez.
- **Birikim katsayısı checkpoint'e yazılır** (format v4). v2/v3 dosyaları
  yüklenebilir; v3, yazıldığı yolu (N = 1) geri yükler. Böylece bir resume
  güncelleme gruplamasını sessizce değiştiremez.
- **LR zamanlaması adım sayar.** Bu yüzden runner'ın `--warmup-updates` /
  `--total-updates` argümanları *tahmin tokenı* birimindedir ve runner
  bunları `--gradient-accumulation`'a bölüp adım cinsine çevirir; token
  birimindeki cosine ufku adım biriminde warmup'ın altına düşerse koşu
  argüman hatasıyla (exit 2) durur.
- Katsayı üst sınırı 512'dir (`NPT_MAX_GRADIENT_ACCUMULATION`): staging
  tamponları O(vocabulary × N), flush çekirdeğinin blok tamponu
  O(width × N). 512'nin üstü hız kazandırmaz, yalnız bellek büyütür.

#### Ölçülen sonuç (tek koşu, tek runner, bayt-özdeş dilim)

CI job 122362411924, vocab 4096 × width 96 × 4 başlık, pencere 48, ffn 384,
400 000 baytlık sentetik Türkçe korpus (paragraf ölçekli kayıtlar, ortalama 833
bayt), 4 thread, aynı seed. Her kol aynı seed'den sıfırdan eğitilir ve aynı
held-out diliminde değerlendirilir:

| kol | token/sn | held-out NLL |
| --- | --- | --- |
| merge-base (main) | 2205,5 | 9,090063 |
| bu dal, N=1 (token-başına) | 4145,7 | 9,089033 |
| bu dal, N=32 (birikim, seçilen) | **7937,2** | **8,442573** |
| bu dal, N=64 | 7843,0 | 8,514667 |
| bu dal, N=128 | 7794,6 | 8,317527 |

- **Hız:** birikim, bu dalın token-başına yoluna göre **1,92x**; merge-base'e
  göre **3,60x** (aynı kodun bir sonraki koşusu: 3,29x). Token-başına yolun kendisi (real32 eleman zinciri, paralel
  bölümler, vektörel softmax) merge-base'in 1,88 katı.
- **Kalite:** eşit tahmin-tokenı sayısında held-out NLL birikim lehine
  **−0,646** (perplexity 8857 → 4629); merge-base'e göre **−0,647**.
- **Faz bütçesi** (seçilen kolda, tahmin başına): decoder adımı **34,6 µs**,
  blok projeksiyonu **23,3 µs**, skorlama **21,8 µs**, optimize edici adımı
  **41,3 µs**; güncellemesiz held-out değerlendirmesi 8052 token/sn.
- **Koşular arası değişkenlik dürüstçe:** aynı job tanımıyla ölçülen oranlar,
  kodun iki hâli için:
  - adım paralelleştirmesi ve bloklu normalizer **öncesi**: 2,78x · 2,81x ·
    2,94x (üç koşu),
  - **son hâli**: 3,60x · 3,29x · 3,24x (üç koşu, ortalama 3,38x).

  Yani son kodun ölçülen aralığı **3,24–3,60x**; her koşuda 3x'in üzerinde. Payın kaynağı merge-base
  kolunun makineye daha duyarlı olması: o kol her tahminde 1,5 MB bağlı
  embedding okuyor, yani bellek bant genişliğine bağımlı ve yavaş/paylaşımlı
  runner'da daha çok düşüyor. Aynı kolun koşular arası farkı 2205–2443 token/sn
  iken bizimkinin 7937–8008 token/sn olması bunu doğruluyor.

Hızı buraya getiren değişiklikler, her biri ölçümle doğrulanmış:

1. **Optimize edici eleman zinciri real32** (momentler dâhil). Darboğaz bellek
   değil aritmetikti: eleman başına bir `sqrt` ve bir bölme, real64'te 4-geniş.
2. **İleri geçişin blok hâlinde birleştirilmesi.** `neural_decoder_step_hidden`
   + `neural_decoder_project_logits`: bağlı embedding blok başına bir kez
   okunur (tahmin başına 1,5 MB okuma kalkar).
3. **Sahnelenen gizli bloğun satır-major olması.** (width, N) düzeninde hem
   projeksiyon hem güncelleme çekirdeği stride-96 okuyordu; satır-major'a
   geçince projeksiyon 90,7 → 23,3 µs.
4. **Softmax normalizer'ı, normalizasyonu ve top-k sıralaması** vektörel/paralel;
   sıralama iki sayılmış karşılaştırmaya indi.
5. **Decoder adımında atıl thread'lerin kullanılması.** Adım tek bir seri
   zincirdi; bağımsız gruplar (query/key/value projeksiyonları, iki FFN dalı,
   elementwise aktivasyon) OpenMP `sections`/`parallel do` ile dağıtıldı.
   Aritmetik değişmediği için sonuç thread sayısından bağımsız.

Denenip **ölçümle reddedilen** iki optimizasyon (kodda da not düşülmüş):

- Elle yazılmış vektörel `matvec`, contiguous `project_heads`, tek-eksponansiyel
  `tanh`: adım 54,4 → 111,9 µs (job 122344621975). gfortran'ın `MATMUL`/
  `DOT_PRODUCT`'ı bu elle formlardan daha iyi vektörleşiyor.
- Sözlüğü L1'e sığan dilimlere bölmek: güncelleme 33,8 → 98,7 µs (job
  122350793122). Kare biriktirici `TILE × width` otomatik dizi; ikinci boyut
  çalışma zamanı değeri olunca gfortran axpy'yi vektörleştirmiyor. Ayrıca
  hipotez de yanlıştı: N=32'de sahnelenen blok 12 KB, yani zaten L1'de.

Hız iddiası ölçümle kapılanır: `make neural-throughput-check`
(`tools/benchmark_neural_batch_throughput.py`) bayt-özdeş sentetik korpusta
aynı seed/shape/zamanlamayla kolları koşar (N=1, `--sweep` ile verilen birikim
faktörleri ve verilirse merge-base binary'si), token/sn oranlarını, **eşit token
sayısında** held-out NLL farkını, faz başına mikrosaniye bütçesini ve "saniye
başına NLL iyileşmesi"ni raporlar; referans kol hiç iyileşmediyse oran yerine
mutlak değeri yazar (0,000x yanıltıcıydı). Runner'lar arası token hızı 2 kattan
fazla oynadığı için karşılaştırma **aynı job içinde** yapılır.
`.github/workflows/neural-batch-throughput.yml` merge-base'ten ikinci bir binary
kurar, faktör süpürmesini (32,64,128) çalıştırır ve kapıları
(birikim/bu-dal ≥1,25x, birikim/merge-base ≥2,0x, NLL sapması ≤0,05) PR'da
zorlar; JSON kanıtı artifact olarak yayımlanır.

### 3.6 NOVA-1 — sıfırdan yazılmış güncelleme kuralı (proje varsayılanı)

Eğiticinin güncelleme kuralı artık AdamW değil: `src/neural_pretraining.f90`
içinde, harici hiçbir kütüphane/çerçeve kullanılmadan yazılmış **NOVA-1**
(`NPT_OPTIMIZER_NOVA`). AdamW ve eski SGD yolu seçilebilir karşılaştırma /
eski checkpoint yolu olarak duruyor (`--optimizer nova|adamw|sgd`); CLI,
`neural_train_config_t` ve benchmark varsayılanı NOVA.

**Kural.** Bağlı (tied) çıktı gradyanı bir tahmin için `r(v) · h`'dir, yani
rank-1: her sütun aynı `h` yönünü kendi `r(v)` katsayısıyla alır. Bu yüzden
eleman başına ikinci moment, sütun başına sabit olan bir büyüklüğü
`width × vocabulary` eleman üzerinde `sqrt`+bölme ile yeniden türetmek olur.
NOVA-1 onu sütun düzeyinde tutar: sütunun batch üzerinden ortalama gradyanı
`g(v)`, enerjisi `z(v) = b2·z + (1-b2)·mean_b(r_b(v)²·||h_b||²/width)` — yani
her tahminin **kendi** enerjisinin batch ortalaması; `||mean_b g_b||²` gibi
`1/B²` ile ölçeklenir (yani adımın izlediği **ortalama** gradyanın ölçeğini
izler; `1/B²` kaldırıldığında etkin adım `1/√B` gibi küçülüyor ve toy sözleşme
`test_nova_optimizer_contract` B=4'te kayıp düşüşünü kaybediyordu). `||mean_b
g_b||²`'ten farkı yalnız üyeler arası çapraz terimler, yani gürültü. B=1'de
token-başına yolun kullandığı ifadenin aynısı ve sütun başına genişlik boyu
indirgeme gerektirmiyor (`O(B)` skaler iş). İlk adımda kendi gradyanından
başlatılır, yüzden bias düzeltmesi yok. Taşınan hızla
uyumu `a(v) = g(v)·u(v)`, hız `u(v) ← γ·u(v) + g(v)` (`γ = b1`, `a(v) < 0`
ise `γ = b1·NPT_NOVA_RESTART_DAMP`) ve adım
`θ(:,v) ← (1 - lr·wd)·θ(:,v) - s(v)·u(v)`, `s(v) = lr/(sqrt(z(v)) + eps)`.
Uyum terimi Adam/RMSprop'ta yoktur: hızıyla çelişen sütun, körü körüne
güvenilmek yerine sönümlenerek yeniden başlatılır — ağır momentumun aşırı
salınım hatası budur — ve bedeli eleman başına değil sütun başına bir iç
çarpımdır. Eleman zinciri tek çarpma + tek FMA; hiçbir eleman için
transandantal yok. Durum: tek hız dizisi (`nova_u_embedding`) + sütun başına
enerji (`nova_energy`), yani eleman başına **16 B** durum trafiği (AdamW 28 B),
bellek `4·width·vocabulary + 8·vocabulary` (AdamW `8·(width·vocabulary + vocabulary)`).
Checkpoint **v5** bu durumu optimizer kimliğiyle birlikte yazar; v2/v3/v4
dosyaları yüklenmeye devam eder, farklı optimizer durumu yüklemek
`NPT_ERR_FORMAT`'tır.

#### Ölçüm — aynı ikili, aynı birikim faktörü, interleaved turlar

`tools/benchmark_neural_batch_throughput.py --compare-optimizer adamw` aynı
koşu içinde NOVA'yı ve AdamW'yu **aynı seed, korpus, şekil ve birikim
faktörüyle** eğitir; tek fark kuraldır. Paylaşımlı runner'ın sürüklenmesi tek
turda ölçülen farktan büyük olduğu için (aynı kod iki koşuda 0,963x ve
1,242x verdi) turlar dönüşümlü koşulur ve **medyan** raporlanır.

| koşu (commit) | N | NOVA tok/sn | AdamW tok/sn | oran | held-out NLL farkı (NOVA−AdamW) |
| --- | --- | --- | --- | --- | --- |
| d54011e | 32 | 7507,5 · 7555,3 | 8080,6 · 8103,4 | **0,931x** | **−0,0322** |
| 3627026 | 64 | 7621,3 · 7630,1 | 8042,2 · 6598,9 | **1,042x** | **−0,1016** |

- **Hız:** ölçülen bant **0,93–1,04x**, yani adım başına AdamW ile parite
  bandında; NOVA örnekleri kendi içinde çok daha kararlı (7621–7630'a karşı
  6598–8080). Adım süresinin baskın kısmı bu şekilde (vocab 4096 × width 96,
  N=64) paylaşılan gradyan toplama (batch × width) olduğu için optimizer
  farkının üst sınırı yapısal olarak küçüktür; NOVA'nın kazancı eleman başına
  iş ve durum trafiğindedir.
- **Kalite:** her iki koşuda da NOVA lehine; eşit birikimde fark birikim
  büyüdükçe açılıyor (−0,032 → −0,102), çünkü sütun enerjisi batch ortalaması
  üzerinden ölçülüyor ve `1/B` ölçekli ikinci momentin aksine gürültüden
  etkilenmiyor.
- **Uçtan uca:** aynı koşuda merge-base'e göre **3,119x** (yukarıdaki
  3,24–3,60x bandıyla aynı sınıf). Faz bütçesi (tahmin başına): decoder
  **33,5 µs**, blok projeksiyonu **23,8 µs**, skorlama **23,2 µs**, optimize
  edici **46,8 µs**.

Denenip **ölçümle reddedilen** iki NOVA optimizasyonu (kodda not düşülmüş):

- **Hız dizisini token-major saklamak** (`nova_u_embedding(vocabulary, width)`):
  sütun unit-stride olur, `DOT_PRODUCT` geçici kopya üretmez — beklenti buydu.
  Ölçüm tersini söyledi: aynı koşuda batched kol 0,623x'e düştü, token-başına
  kol 12 kat yavaşladı. Geri alındı (21d396d).
- **Enerji ve uyumu tek skaler tarama içinde birleştirmek:** iki biriktiricili
  döngü ve strided hız okuması vektörleşmiyor; oran 0,931x → 0,823x (179d3f8,
  geri alındı). Normalizasyon geçişini güncelleme taramasına katlamak
  (toplam iki tarama) ise 1,042x veren hâldir.

Sözleşme native testtedir: `test_nova_optimizer_contract` (kayıp düşüşü, tek
hız dizisi, AdamW'dan küçük kaynak planı, v5 turu, parça+devam
bit-birebirliği) ve `test_adamw_optimizer_contract` (eski yol bozulmadı).

### 3.7 Darboğaz denetimi — ölçülen faz bütçesi, düzeltilenler, reddedilenler

Faz bütçesi artık **her kol için** yayımlanıyor (`benchmark_neural_batch_throughput.py`
annotation'ı: `phases <kol>[d… p… s… u… t…]`). Bu, "hangi faz darboğaz" sorusunu
tahmin olmaktan çıkarıp ölçüme bağlıyor; aynı kodun kollar arasında ~%10–14
oynadığı da bu tablodan görülüyor, yani tek ölçümle iddia kurulmuyor.

Üretim şekli (vocab 4096 × width 96 × ffn 384 × 4 thread, N=32), aynı koşu
(run 40663284104, hızlı sınıf runner):

| faz | µs/tahmin | pay | ölçülen darboğaz |
| --- | ---: | ---: | --- |
| decode | 16,2 | %24 | FFN'in 768 transandantalı + 4 yoğun matris-vektör; `/8` vektör genişlikte donanım sınırına yakın |
| project | 6,5 | %10 | E bir blokta bir kez okunuyor; kalan maliyet `(count, vocab)` çıktısında satır-aralıklı yazma |
| score | 12,7 | %19 | softmax + true-token sıralaması, tahmin başına 5–6 kez O(vocabulary) tarama |
| update | 31,0 | %47 | blok gradyanının **skaler hızda** biriktirilmesi (aşağıya bakın) |

**Düzeltilenler**
1. `stable_softmax`: `maxval(real(logits, real64))` her tahminde 4096 elemanlık
   real64 geçici dizi ayırıp dolduruyordu → `real(maxval(logits), real64)`
   (bit-birebir, geçici yok). Ayrıca bölme sonrası "hepsi sonlu mu" taraması
   bölmenin kendi geçişine katlandı (durum bayrağı üzerinde reduction).
   Ölçüm: bu runner'da gürültü bandında; kazanç yapısal (bir geçiş ve bir
   ayırma eksik).
2. Benchmark artık her kolun faz bütçesini ve optimizer A/B'sini **seçilen**
   birikim faktöründe, dönüşümlü turların medyanıyla yayımlıyor.
3. Uçtan uca oran artık runner sınıfına göre raporlanıyor: merge-base kolu
   bellek-bağımlı olduğu için hızlı host'ta orantısız hızlanıyor (referans
   sınıfta 2,1–2,5k token/sn → 3,24–3,60x; hızlı host'ta 8,5k token/sn → 1,70x).
   Oran her iki durumda da yazılıyor; kalibrasyon bandının dışında kapı
   "fail" değil "notice" üretiyor.

**Denenip ölçümle reddedilenler** (kodda not düşülmüş)
- **Blok gradyanını MATMUL ile kurmak** (aynı çarpım, aynı sıra — aritmetik
  olarak birebir): rank-1 ve aralıklı ilk operand yüzünden intrinsik her sütun
  için kopya + sonuç ayırıyor, adım başına 4096 kez. Aynı koşu içinde
  normalize edilmiş güncelleme 61,8 → 134,2 µs/tahmin (×2,2 yavaş).
- **FFN'i satır-paralel yapmak** (`project_rows`, `dot_product` ile satır
  başına): aynı koşuda aynı model/şekil ile decoder adımı 79,9 → 105,8
  µs/tahmin (%32 yavaş). Tek bloklu `MATMUL` satır başına `DOT_PRODUCT`'tan
  iyi. İki OpenMP section'ın boş bıraktığı iki lane, kaybedilenden ucuz.
- **Hız dizisini token-major saklamak** (daha önce, §3.6): batched kol 0,623x.

**Açık kalan, ölçülmüş darboğazlar** (yapılacaklar, iddia değil)
1. **Güncelleme fazının biriktirmesi skaler hızda.** 12,6M çarpma-toplama
   (4096 sütun × 32 batch × 96) 4 lane'de kabaca çevrim başına bir işlem
   hızında koşuyor; düz `!$omp simd` axpy bunu vektörleştirmiyor. Denenmesi
   gereken sıradaki adım, biriktiriciyi sabit boyutlu (derleme zamanı) bir
   tamponla bloğa almak — bu depoda iki kez çalışan reçete (§3.5'teki
   "runtime boyutlu biriktirici vektörleşmiyor" notu).
2. **NOVA'nın sütun başına kalan iç çarpımı.** Enerji artık üye enerjilerinin
   batch ortalaması olduğu için (yukarı bakın) genişlik boyu indirgeme
   kalmadı; uyum için sütun başına bir `dot_product` duruyor. Ölçülen sonuç
   (enerji değişikliğinden **önce**): NOVA'nın güncelleme fazı AdamW'ninkinden
   hızlı değildi (30,6'a karşı 18,5 µs/tahmin, aynı koşu) ve toplam oran
   8 koşuda 0,82–1,24x, medyan ~0,93x. Yani "güncelleme fazında kanıtlanabilir
   hız" hedefi bu hâliyle **karşılanmış değil**; enerji değişikliğinin etkisi
   bir sonraki ölçümle doğrulanacak.
3. **Skorlama fazının O(vocabulary) taramaları** (softmax'in 4 geçişi +
   sıralama için iki `count`): vocab 32k'ya çıkınca faz lineer büyür.
4. **Korpus okuma bayt bayt**: `utf8_stream_read_record` akış üzerinden tek
   bayt okuyor; bu şekilde 833 baytlık kayıt başına ~800 okuma çağrısı var.
   Faz bütçesinin dışında kaldığı için bu tabloda görünmüyor ama 900 MB'lık
   hatta kayıt başına sabit bir maliyet.

## 4. Koşu tarifi

Probe (hız ölçümü) ve full (bütün dump boru hattı + boyutlandırılmış epoch
dilimi) modları `neural-turkish-epoch1-evidence.yml` içindedir:

1. `mode=probe`: 6.000 belge, 3 MB epoch dilimi — token/sn ölçülür.
2. Epoch bütçesi, hedef duvar süresine göre hesaplanır:
   `bayt ≈ token/sn × saniye × (bayt/token)`. **Güvenlik payı zorunludur:**
   ilk full denemesi, uzun koşuda etkin hızın probe hızının <%56'sına
   düşebildiğini gösterdi; full boyutlandırma ölçülen hızın yarısı
   varsayılarak yapılır (ör. 6409 token/sn ölçümü → ~3200 plan hızı).
3. `mode=full` + `train_byte_budget`: bütün dump filtrelenir (gerçek
   toplam sayılır), epoch 1 bu dilimde tanımlı ve hash'lidir.
4. Kalan veri, `train-chunk` + attested imleç ile sonraki koşularda aynı
   checkpoint'ten devam edebilir; çift-uygulama yapılmaz.

## 5. 99B'ye kalan mühendislik (yol haritası, iddia değil)

1. **Veri**: FineWeb-2 `tur_Latn` + FinePDFs + trwiki + COSMOS birleşimi;
   kaynaklar-arası paragraf dedupe ve kalite skorlaması (bu hattın çok-
   kaynak genişlemesi). Hedef bileşim ~99B BPE token'ı ancak böyle verir.
2. **Model**: tam BPTT (dikkat/retention/FFN gradyanları), daha derin
   mimari, karma hassasiyet.
3. **Sistem**: GPU/çok-düğüm veri paralelliği, shard-akışlı eğitim,
   arıza toleranslı checkpoint pencereleri.
4. **Değerlendirme**: bağımsız insan-anotasyonlu Türkçe benchmarklar,
   talimat verisi yönetişimi ve güvenlik değerlendirmesi.

Her adım kendi kanıt artefaktıyla gelmeden "99B eğitildi" denmez.
