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
uygulanmaz) ve `adam_beta1/beta2/adam_epsilon`. `--weight-decay` yalnız
`--optimizer adamw` ile kabul edilir.

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
