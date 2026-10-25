# Ölçülmüş gerçek Türkçe Wikipedia 99M native-token sonuçları

Bu belge, başarılı immutable GitHub Actions artifact’indeki
`experiment-report.json` dosyasını doğrulayıp PR’ye yayımlayan sonuçtan
türetilmiştir. Sayılar fixture, simülasyon veya sentetik corpus sonucu değildir.
Canonical makinece üretilmiş ham sonuçlar (11 semantic output ve 11 literal TSV
response dahil) aşağıdadır:

- Kaynak Actions run: [40178556527](https://github.com/sswweett00/Vegai/actions/runs/40178556527)
- Doğrulanmış artifact özeti ve ham output: [issuecomment-6396514266](https://github.com/sswweett00/Vegai/pull/1#issuecomment-6396514266)
- Evidence artifact: `turkish-wikipedia-99m-evidence-40178556527` (Actions run’da 30 gün saklanır)
- Checkpoint artifact: `turkish-wikipedia-99m-checkpoint-40178556527`
- Evidence commit: `aa1ce1ec8ecf63fd863dded7affff9b36986616d`

> Sonuç özeti: Bu gerçek 99M-token koşu native Fortran eğitim/inference
> zincirini başarıyla tamamladı; ancak held-out local-association sonucu ve
> semantic nearest-neighbour satırları modelin genel Türkçe semantic retrieval
> kalitesinin **zayıf** olduğunu gösteriyor. Bu checkpoint LLM değildir ve
> serbest metin üretmez.

## Kaynak ve tam corpus doğrulaması

Eğitim kaynağı resmi Türkçe Wikipedia `20261001` snapshot’ıdır;
`CC-BY-SA-4.0` olarak kaydedildi. Resmi durum manifesti:
[`dumpstatus.json`](https://dumps.wikimedia.org/trwiki/20261001/dumpstatus.json),
SHA-256 `e75a526cbefc916860b5de91c1d21510afce2c9b4eee74ae6fa2922c811d21f8`.

| Resmi XML bzip2 shard | SHA-256 | Byte |
|---|---|---:|
| [`p1–p1500000`](https://dumps.wikimedia.org/trwiki/20261001/trwiki-20261001-pages-articles1.xml-p1p1500000.bz2) | `7d74056a25d57d9521a54e1a12da34762e53d67852bd63edf3df143f2e14d12a` | 360,284,735 |
| [`p1500001–p3000000`](https://dumps.wikimedia.org/trwiki/20261001/trwiki-20261001-pages-articles1.xml-p1500001p3000000.bz2) | `e6f6405bcfb90f38e8ca192aeddffb0cd6246d2c352230ca31190d9d25cd2fcb` | 376,984,510 |
| [`p3000001–p4500000`](https://dumps.wikimedia.org/trwiki/20261001/trwiki-20261001-pages-articles1.xml-p3000001p4500000.bz2) | `aa59de09f7ef6dd37684ca9ceb4d27cd91101d059cb57dfd3d7baaef0967c622` | 248,686,484 |
| [`p4500001–p4754320`](https://dumps.wikimedia.org/trwiki/20261001/trwiki-20261001-pages-articles1.xml-p4500001p4754320.bz2) | `5abc39c38fdfe9d9c54ccf1e12d2f34df76b1afb43576dff66acc970c2a5d5e5` | 72,431,606 |

- `tr-ascii-v1` normalization ile **tam 99,000,000** native token üretildi.
- Çıktı SHA-256: `8725ac8fdaa9b55a46cc5eb8b4f836e457abcd85b9856391e8ec1fd91ca0b675`.
- Fiziksel kayıt sayısı: `1,237,356`; her kayıt `2–128` token.
- Non-ASCII byte, CRLF record, overlong record ve tokenizer truncation: **0**.
- 262,144 source-frequency-ranked vocabulary tipi kullanıldı. Seçilmeyen
  source tipleri yalnız aynı article içinde elendi; token üretilmedi,
  tekrar edilmedi ve article sınırı geçilmedi.
- Article-level split: `686,436` train page, `6,841` source-disjoint held-out
  page. Held-out sayfalar vocabulary seçimine ve model update’lerine girmedi.

## Gerçek native Fortran eğitim

```text
OMP_NUM_THREADS=4 vegai_pretrain train TURKISH_WIKIPEDIA_99M_ASCII.txt checkpoint.vgai \
  --max-vocab 262144 --min-frequency 1 --window 2 --position-shift 41 \
  --context-mode resonant --epochs 2 --strict \
  --parallel-context-build --parallel-min-contexts 32
```

| Ölçüm | Gerçek değer |
|---|---:|
| Epoch | 2 |
| Raw token / active token / prototype update | 198,000,000 / 198,000,000 / 198,000,000 |
| OOV | 0 |
| Sadece native train-child wall time | 4,577.195837 s (yaklaşık 76 dk 17 sn) |
| Native train throughput | 43,257.926 token/s |
| Checkpoint SHA-256 | `92b96a476c8595b4f6306b69193e46c0e5ff7d7c9ea2ce2245d139bd2c3adda1` |

Süre yalnız `vegai_pretrain train` child process’ini kapsar; indirme,
extraction, corpus oluşturma, compilation, inspect, retrieval, hashing ve CI
bu throughput hesabına dahil değildir. RESONANT için ikinci epoch yalnız
birinci epoch sonunda commit edilmiş frozen semantic snapshot’ı topical feedback
olarak kullandı.

## Held-out gerçek checkpoint inference: kalite sonucu

Source-disjoint 6,841 held-out article üzerinde `1,007,170` active-token
occurrence ile 128 uygun query değerlendirildi. Bu ölçüm checkpoint’in gerçek
top-1 nearest-neighbour output’unu held-out local co-occurrence association ile
karşılaştırır; insan anotasyonlu semantic benchmark veya genel NLP puanı
değildir.

| Metrik | Değer |
|---|---:|
| Eligible query | 128 |
| Exact held-out association Top-1 agreement | **0.007812** (yaklaşık 1 / 128) |
| Tahmin edilen komşunun held-out context penceresinde görülme oranı | **0.085938** (yaklaşık 11 / 128) |
| Held-out association rank MRR | **0.018416** |

Bu değerler yüksek semantic retrieval kalitesi göstermiyor. Özellikle exact
Top-1 yaklaşık `%0.78` ve MRR `0.018416`; dolayısıyla bu koşudan güçlü genel
Türkçe anlama, semantic search veya dil modeli kalitesi iddiası çıkarılamaz.

## 11 gerçek checkpoint semantic nearest-neighbour output’u

| # | Query | Self-excluded nearest neighbour | Bipolar similarity |
|---:|---|---|---:|
| 1 | `turkiye` | `terakki` | 0.04663 |
| 2 | `istanbul` | `kahverengili` | 0.05200 |
| 3 | `ankara` | `destekcileriydi` | 0.04736 |
| 4 | `tarih` | `bulgurdur` | 0.06396 |
| 5 | `devlet` | `aranjordur` | 0.05371 |
| 6 | `universite` | `superbike` | 0.05151 |
| 7 | `belediye` | `prores` | 0.10742 |
| 8 | `sanat` | `montilla` | 0.06311 |
| 9 | `futbol` | `manifolddur` | 0.04114 |
| 10 | `kitap` | `gostergedir` | 0.05420 |
| 11 | `dil` | `vardir` | 0.18860 |

Bunlar checkpoint’ten alınmış gerçek self-excluded nearest-neighbour output’larıdır.
Çoğu komşu query ile anlamlı görünmediğinden tablo, held-out metriğin zayıf
sonucuyla tutarlıdır; bunlar generated answer değildir.

## 11 grounded TSV retrieval output’u

Aşağıdaki sonuçların tamamında `vegai_chat`, ayrı MIT-lisanslı
`Aliumutcan/Turkish_NLP_FAQ_DATASET` revision
`3ff5a943569883d86a4b63ecd88fbfc13891f9f7` içindeki aynı prompt’u buldu.
FAQ data-file SHA-256:
`b9ac1d3f9ef5469c85e19126876d3cfce5c4dd1c14b119890e64e773e8ae5167`.
Bu, modellenmiş serbest cevap üretimi değil; literal, ASCII-normalized TSV
alanının native retrieval ile geri getirilmesidir. Tam literal response text’leri
ve source URL’leri canonical [PR evidence comment](https://github.com/sswweett00/Vegai/pull/1#issuecomment-6396514266)
içindedir.

| # | Prompt | TSV row | Best | Runner-up | Margin | Grounded |
|---:|---|---:|---:|---:|---:|---|
| 1 | `pasaport ile ilgili is ve islemler nasil yapilir` | 1 | 1.00000 | 0.17493 | 0.82507 | true |
| 2 | `yurtdisinda ogrenim gormek icin avrupa birligi ve dis iliskiler genel mudurlugunce verilen herhangi bir burs var midir` | 2 | 1.00000 | 0.44263 | 0.55737 | true |
| 3 | `avrupa birligi ve dis iliskiler genel mudurlugunce hangi burs islemleri yapilmaktadir` | 3 | 1.00000 | 0.44263 | 0.55737 | true |
| 4 | `yurtdisinda lise egitimini tamamlamis ogrencilerin yuksekogretim kurumlarina kabulu nasil yapilmaktadir` | 8 | 1.00000 | 0.18237 | 0.81763 | true |
| 5 | `yurtdisindaki muadil okullarla kardes okul iliskisi nasil tesis edilir` | 9 | 1.00000 | 0.11780 | 0.88220 | true |
| 6 | `ab egitim ve genclik programlari nelerdir milli egitim bakanligi'nin bu programlarda fonksiyonu nedir` | 16 | 1.00000 | 0.37634 | 0.62366 | true |
| 7 | `halk egitim merkezlerinde veya mesleki egitim merkezlerinde kursiyer olarak egitim gorenlerden ab egitim ve genclik programlari kapsaminda yurutulen proje faaliyetleri kapsaminda yurt disina cikacak kisilere hizmet damgali pasaport verilebilir mi` | 18 | 1.00000 | 0.37634 | 0.62366 | true |
| 8 | `ulusal uluslararasi projelere yonelik duyurulari nasil takip edebilirim` | 20 | 1.00000 | 0.18237 | 0.81763 | true |
| 9 | `pestalozzi programi nedir` | 22 | 1.00000 | 0.12805 | 0.87195 | true |
| 10 | `daha once bir comenius okul ortakligi projesinde yer aldim bu seminere katilmama engel midir` | 27 | 1.00000 | 0.13574 | 0.86426 | true |
| 11 | `istek turunun basvuru konusu ne olmalidir` | 39 | 1.00000 | 0.11865 | 0.88135 | true |

Bu 11/11 exact-match grounding başarısı FAQ retrieval pipeline’ının ilgili
kaynak satırlarını geri verdiğini kanıtlar. Ancak prompt’lar knowledge TSV’de
zaten bulunduğu için bunu Wikipedia ile eğitilen semantic modelin generalization
başarısı, factuality ölçümü veya LLM cevabı diye sunmak doğru değildir.

## Sınır ve dürüst yorum

Bu run şunları doğrular: açık lisans/provenance bağlı gerçek Turkish Wikipedia
metninden tam 99M native-token corpus, iki gerçek native Fortran epoch, üretilmiş
checkpoint, gerçek checkpoint inspect ve source-disjoint held-out inference.

Şunları doğrulamaz: serbest metin generation, genel Türkçe language
understanding, insan semantic-similarity benchmark başarısı, cited TSV dışı
factual correctness veya genel zekâ. Özellikle bu run’ın gerçek held-out ve
semantic-neighbour sonuçları, modelin mevcut HDC/VSA retrieval temsilinin
kalitesinin bu görevde zayıf kaldığını açıkça göstermektedir.
