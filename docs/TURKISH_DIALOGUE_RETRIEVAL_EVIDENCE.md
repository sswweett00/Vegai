# Türkçe diyalog retrieval evidence — content-disjoint calibration run

Bu kayıt, native Fortran batch evaluator ve canonical-prompt leakage filtresi
sonrasında üretilen gerçek CI çalışmasının kalıcı özetidir. Bir generative LLM
kalite kartı değildir.

| Alan | Değer |
|---|---|
| Evidence run | [GitHub Actions 40197729219](https://github.com/sswweett00/Vegai/actions/runs/40197729219) |
| Branch head | `a19ca193fe5176e9680b36de5870b931fa6e3b86` |
| Workflow merge commit | `6165f191e207281d8fe6926e1b337b69a1e51da1` |
| Kaynak | `3nesdeniz/turkish-daily-dialogues-5k` @ `ccd9f05c2f97684bbd9a55d818528da9dfb6bd5a` |
| Kaynağın beyan ettiği lisans | `CC-BY-4.0` |
| Kaynağın beyan ettiği doğa | `synthetic-ai-assisted-deterministic` |
| Native evidence artifact | `turkish-dialogue-retrieval-evidence-40197237783` (30 gün CI retention) |

## Çalıştırılan gerçek native yol

1. Immutable package compiler, train splitten `4,036` unique canonical UTF-8
   TSV key ve `25,268` ASCII HDC training record üretti.
2. `vegai_pretrain` gerçek Fortran executable'ı iki epoch resonant HDC
   checkpoint eğitti.
3. `vegai_dialogue_eval` checkpoint ve knowledge TSV'yi her ölçümde bir kez
   yükledi; stateless, strict three-column TSV case stream'ini native Fortran'da
   ölçtü. Raporlar checkpoint/knowledge/evaluation SHA-256 ve byte count
   identity'lerini içerir; raw prompt/response içermez.
4. Hybrid score-floor yalnız source-disjoint **validation** setinde önceden
   ilan edilmiş adaylar arasından seçildi. Frozen test TSV calibration aracına
   verilmedi. Daha sonra aynı testte önceden ilan edilmiş `0.65` baseline ve
   validation ile seçilen policy ölçüldü.

## Leakage denetimi

Source conversation ID split'i tek başına yeterli değildi. Native tokenizer
projection'ında aynı canonical prompt train knowledge key'iyle çakıştığında,
held-out `expected_prompt: null` etiketi meşru retrieval'ı hatalı biçimde
unsafe grounding sayardı. Package compiler bu çakışmaları evidence setinden
çıkarıp manifest'e bağladı:

| Filtre sonucu | Sayı |
|---|---:|
| Train canonical key | 4,036 |
| Validation raw pair | 1,684 |
| Validation canonical duplicate drop | 559 |
| Validation train-key overlap exclusion | 869 |
| Kalan content-disjoint validation abstention case | **256** |
| Test raw pair | 1,682 |
| Test canonical duplicate drop | 577 |
| Test train-key overlap exclusion | 861 |
| Test selected-validation-key overlap exclusion | 15 |
| Kalan frozen content-disjoint test abstention case | **229** |

Builder, content filtering sonucunda validation veya testte en az `128` case
kalmazsa package yayınını fail-closed durdurur. Bu kaynaktaki template tekrarının
büyüklüğü ayrıca kaydedilir; bu filtre source corpus'un insan-doğal sohbet
kalitesini kanıtladığı anlamına gelmez.

## Ölçülen sonuçlar

### Full train exact-key retrieval contract

| Metrik | Sonuç |
|---|---:|
| Case | 4,036 |
| Correct grounded prompt | 4,036 |
| Grounded-prompt accuracy | 1.0 |
| Grounded precision | 1.0 |
| False abstention | 0 |

Bu yalnız published train TSV key'inin aynı checkpoint/KB ile literal olarak
bulunabildiğini gösterir. Held-out generalization değildir.

### Validation-only policy seçimi

- Semantic weight: `0.25`
- Lexical weight: `0.75`
- Hybrid margin floor: `0.03`
- Önceden ilan edilmiş validation false-grounded-rate cap: `0.01`
- Seçilen en düşük güvenli hybrid score floor: **`0.95`**

### Frozen content-disjoint test: aynı checkpoint, aynı 229 case

| Policy | Rendered true abstain | OOV safe rejection | False grounded | Rendered abstain recall | Non-fabrication recall (OOV dahil) |
|---|---:|---:|---:|---:|---:|
| Predeclared baseline, score floor `0.65` | 124 | 0 | 105 | 0.5414847161572053 | 0.5414847161572053 |
| Validation-calibrated, score floor `0.95` | 229 | 0 | 0 | 1.0 | 1.0 |

Dolayısıyla bu spesifik **abstention/leakage-safety** contract'inde false
grounding `105 → 0`, rendered abstention recall
`0.5414847161572053 → 1.0` oldu. Karşılaştırma test sonucuyla threshold seçmek
için kullanılmadı; selection yalnız validation ile yapıldı.

## Dürüst yorum ve sınırlar

Bu iyileşme kullanıcıya daha ilgili veya daha doğal yeni cevaplar verdiğini
kanıtlamaz. Aksine selected `0.95` policy, bu 229 content-disjoint null-label
case'in tamamında abstain etti. Bu güvenli refusal davranışı için gerçek bir
kazançtır; unseen-answer relevance, factuality, helpfulness, paraphrase
coverage veya natural conversation quality ölçümü değildir.

Kaynak açıkça sentetik/AI-assisted'tır. `vegai_pretrain` HDC contextual
prototype eğitir; causal next-token loss, logits, decoder ve sampling yoluna
sahip değildir. `vegai_chat` yalnız operator TSV'den literal response retrieve
der. Bu nedenle bu evidence, Vegai'nin doğal free-form Türkçe sohbet ürettiği
veya yeterli miktarda aynı sentetik veriyle bunu yapacağı iddiasını desteklemez.

Gerçek doğal sohbet için hâlâ en az şunlar gerekir:

1. izin/attribution/PII politikası belirlenmiş, insan yazımı veya açıkça
   governed Türkçe dialogue corpus;
2. native causal decoder, tokenizer, next-token objective, optimizer,
   checkpoint/resume ve bounded sampling implementation'ı;
3. train/validation/test arasında canonical/template/semantic contamination
   denetimi;
4. source-disjoint paraphrase-to-grounded-answer labels, answer relevance ve
   factuality annotation'ları;
5. Turkish human preference/naturalness evaluation ve final test'ten bağımsız
   calibration.

İlgili operational contract: [TURKISH_DIALOGUE_DATA_GOVERNANCE.md](TURKISH_DIALOGUE_DATA_GOVERNANCE.md)
ve [DIALOGUE_RETRIEVAL.md](DIALOGUE_RETRIEVAL.md).
