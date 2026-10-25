# Gerçek 9 GB COSMOS Türkçe Nedensel Ön-eğitim Sonucu

Bu kayıt, **25 Ekim 2026** tarihinde çalışan tek gerçek evidence run'ının
artifact'i ve immutable PR evidence comment'inden türetilmiştir. Sentetik
fixture, nominal byte sayısı veya tekrar eden corpus değildir.

- Workflow: [40208842546](https://github.com/sswweett00/Vegai/actions/runs/40208842546)
- Evidence comment: `6400688649`
- Uygulama revision: `e65e9c7` (sonraki publisher/docs commit'lerinden önceki
  gerçek training snapshot'ı)
- Dataset: `ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0`
- Resolved immutable dataset revision:
  `8c9a64084b6cc2665462971d860d599b574d82ea`
- Dataset-card lisansı: `CC-BY-4.0`
- Dataset-card SHA-256:
  `8906a49d554dd78e4c9be02e8584472894492832816cc214291eed21c4d9f72b`

## Doğrulanmış corpus sınırı

| Partition | Tam byte | SHA-256 | Ayrım |
|---|---:|---|---|
| Native training | 9,000,000,000 | `143c928ca7c4df12c9312769362654a9d52d337e93d55a6bbb7fd5e7be70ea32` | gerçek kaynak-türevi, exact-document ve exact-record SHA-256 dedupe |
| Frozen held-out | 50,000,000 | `930a514f88d39a7b047b48c0208555ac3c041f385c28010c6f8055bc80ad3183` | train'den source-document-disjoint |

Controller Unicode NFC ve whitespace canonicalization yaptı; transliteration,
translation, padding, cyclic repetition veya sentetik filler kullanmadı. Tam
9 GB sınırında yalnız valid UTF-8 source-contiguous son substring kullanma
sözleşmesi vardı. Direct email/telefon/IPv4 regex'li belgeler reject edildi.
Bunlar bir complete PII audit ya da near-duplicate-free garantisi değildir.

## Tam bir native update epoch

| Tam update epoch | Native causal update saniyesi | Raw token | Raw token/s |
|---:|---:|---:|---:|
| 1 | 193.132585526 | 1,235,844,700 | 6,398,944.521113074 |

Bu süre yalnız native `causal_lm_train_file` update geçişinin çevresindeki
Fortran `system_clock` ölçümüdür. Vocabulary heavy-hitter pass, corpus
download/parse/dedupe, derleme, checkpoint serialization, held-out evaluation,
generation ve CI orchestration throughput hesabından çıkarılmıştır.

## Source-document-disjoint held-out n-gram ölçümü

| Metrik | Gerçek değer |
|---|---:|
| Top-1 next-token accuracy | 0.7560882309106265 |
| Bounded-successor coverage | 1.0 |

Bu değerler yalnız bounded causal heavy-hitter n-gram sketch'inin known-token
continuation davranışını ölçer. Factual correctness, Türkçe soru-cevap kalitesi,
instruction following, semantik akıl yürütme veya neural-LM perplexity değildir.
Özellikle successor coverage'ın 1.0 oluşu, final user-facing kalite kanıtı
olarak yorumlanmamalıdır; bounded table'nin held-out next-token target'larını
kapsaması anlamına gelir.

## Gerçek native prompt/response çıktıları

Aşağıdaki metinler workflow artifact'indeki native `vegai_causal_lm generate`
çıktılarının **aynen** kopyasıdır. Bunlar daha iyi görünmesi için düzenlenmedi.

1. **Prompt:** `Türkiye'nin başkenti`

   **Response:** `ve ve ve ve ve ve ve ile ve ve ve ve ile ve ve ve ve ve ve ve ve ve ve ve ve ve ve ve ile ile ile ile ve ve ve ve ve ve ve`

2. **Prompt:** `Yapay zeka verilerden`

   **Response:** `ve`

3. **Prompt:** `Bilimsel bir açıklama için`

   **Response:** `ve ve ve ve ve ve ve ile ve ile ve ve ve ile ve ve ve ve ve ve ve ile ile`

4. **Prompt:** `Trabzon Karadeniz kıyısında`

   **Response:** `ve ve ve ve ve ve ve ve ve ve ve ve ve ve ve ve ve ve ve`

5. **Prompt:** `Türkçe bir cümlede`

   **Response:** `ve ve ve ve ve ve ve ve ve ve ve ve ve ile ve ve ve ile ve ve ile ve ve ve ile ile ile ve ve ve ve ile ve ve ve ve ve ve ve ve ve ve ve ve ve ve ve ve`

## Dürüst kalite kararı

Bu run, seçilen açık lisanslı Türkçe kaynaktan tam 9 GB, real/source-traceable
bir corpus üretildiğini ve native Fortran causal modelin bu corpus üzerinde
**tam bir update epoch** çalıştırıp gerçek continuation ürettiğini kanıtlar.
Önceki HDC/TSV retrieval yolunun sahip olmadığı causal generate yüzeyi artık
vardır.

Bununla birlikte yukarıdaki fonksiyon-kelimesi tekrarları, bu küçük bounded
n-gram modelin open-ended Türkçe cevap kalitesinin **yetersiz** olduğunu açıkça
gösterir. Bu sonuç genel zeka, başarılı Türkçe asistan, factual answer veya
kaliteli prompt-response başarısı olarak sunulamaz. Kaliteyi yükseltmek için
neural/subword causal architecture, daha güçlü context/state, calibrated
decoding ve ayrı lisans/provenance-audited instruction/answer supervision
ayrıca gerekir; corpus'u büyütmek tek başına bu sınırı ortadan kaldırmaz.
