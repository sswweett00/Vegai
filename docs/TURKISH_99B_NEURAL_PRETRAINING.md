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
| 1.000 | ~3.140 yıl |
| 10.000 | ~314 yıl |
| 100.000 | ~31 yıl |
| 1.000.000 | ~3,1 yıl |

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
| Düzenlileştirme | yalnız gradyan kırpma + değer zarfı | + **label smoothing** (NLL raporu daima gerçek), AdamW weight decay |
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
prompt/response'lar düzenlenmeden yazılır.

## 4. Koşu tarifi

Probe (hız ölçümü) ve full (bütün dump boru hattı + boyutlandırılmış epoch
dilimi) modları `neural-turkish-epoch1-evidence.yml` içindedir:

1. `mode=probe`: 6.000 belge, 3 MB epoch dilimi — token/sn ölçülür.
2. Epoch bütçesi, hedef duvar süresine göre hesaplanır:
   `bayt ≈ token/sn × saniye × (bayt/token)`.
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
