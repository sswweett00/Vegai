# Held-out HDC semantic-neighbour kalite kapısı

`tools/pretraining_quality_evaluation.py`, derlenmiş gerçek
`vegai_pretrain inspect --query-file` yolunu operator-owned, versioned bir
held-out annotation seti ile ölçer. Amaç HDC checkpoint'in **belirli hedef
domain'deki self-excluded Top-1 semantic-neighbour retrieval** davranışını
ölçülebilir hale getirmek ve başarısız aday checkpoint'i release öncesinde
fail-closed durdurmaktır.

> **Dürüst kapsam:** Bu araç yalnız verilen checkpoint ve verilen held-out
> semantic-neighbour annotation'ları için Top-1 hit/similarity ölçer. Geçen bir
> gate; genel dil anlama, dünya bilgisi, factuality, güvenlik, kodlama,
> multimodal reasoning, calibration veya LLM/LMM üstünlüğü kanıtı değildir.
> Vegai'nin HDC/VSA pretraining'i Transformer loss/perplexity tabanlı bir LLM
> training sistemi değildir.

## Neden ayrı bir release gate?

Mevcut production yolunda aşağıdaki veri ve transaction kontrolleri zaten
vardır:

- strict ASCII/LF ingest, exact dedupe ve content-addressed immutable shard;
- deterministic train/validation split ve record-level exact leakage audit;
- vocabulary capacity/resource preflight;
- checkpoint SHA-256 journal'i ve exactly-once continual commit;
- immutable dataset/binary/training-policy run fingerprint'i.

Bunlar bir checkpoint'in doğru **üretilip korunmasını** sağlar; checkpoint'in
operator'ın hedef domainindeki semantic ilişkileri yeterli öğrendiğini tek
başına göstermez. Bu katman o boşluğu hardcoded cevap, LLM judge veya
in-corpus benchmark sonucu olmadan; operator'ın ayrı held-out annotation'ları,
açık eşikler ve gerçek checkpoint inspection ile kapatır.

## Girdi sözleşmesi

Araç standart kütüphane kullanır; ilave pip paketi, model weight'i veya network
isteği yapmaz. Tüm JSON nesnelerinde duplicate key ve unknown field fail-closed
reddedilir. Case/policy input'ları, binary, checkpoint ve evaluator source
byte'ları SHA-256 ile bağlanır; external `inspect` çağrısı boyunca değişirse
sonuç yayınlanmaz.

### 1. Case manifest: `vegai-pretraining-quality-cases/v1`

Tek JSON object aşağıdaki **tam** alanları taşır:

```json
{
  "schema_version": "vegai-pretraining-quality-cases/v1",
  "evaluation_id": "support-domain-r2026-10",
  "declared_split": "held-out",
  "annotation_provenance": {
    "source": "governed evaluation registry: support-r2026-10",
    "license": "approved-internal-evaluation-license",
    "retrieved_at": "registry-event-2026-10-24T00:00:00Z",
    "method": "versioned human-reviewed semantic-neighbour annotation protocol v1"
  },
  "cases": [
    {
      "schema_version": "vegai-pretraining-quality-case/v1",
      "id": "support-001",
      "query_token": "service_ticket",
      "acceptable_neighbours": ["incident", "support_queue"]
    }
  ]
}
```

Kurallar:

- `declared_split` tam olarak `held-out` olmalıdır; başka değer kabul edilmez.
- `annotation_provenance.source`, `license`, `retrieved_at`, `method` zorunlu,
  CR/LF içermeyen en fazla 1.024 UTF-8 byte metindir. Bu metadata report'a
  kopyalanmaz; case-manifest SHA'sı üzerinden immutable olarak bağlanır.
- Case ID ve `evaluation_id`, bounded ASCII identifier'dır. ID'ler ve query
  token'ları case set içinde unique olmalıdır; bu implicit weighting'i önler.
- Query/acceptable token, Vegai tokenizer ile birebir uyumlu lowercase
  ASCII `[a-z0-9_']`, 1–48 byte olmalıdır. Query, kendi acceptable set'inde
  olamaz; inspect zaten query'yi nearest-neighbour adaylarından çıkarır.
- Her case 1–128 farklı acceptable token taşıyabilir. Bu, bir semantic ilişkinin
  tek keyfi eş anlamlıya indirilmesini önler.
- Case manifest'in kendisi raw training record içermez; erişim kontrollü
  annotation artifact'i olarak tutulmalıdır. Report bu tokenları kopyalamaz.

`held-out` burada annotation/evaluation governance beyanıdır: query token'ın
training vocabulary'de bulunması normal ve inspect için zorunludur. Araç raw
corpus'u tekrar okumadığından, annotation'ın training, threshold tuning veya
model-selection sürecine sızmadığını matematiksel olarak kendi başına kanıtlayamaz.
Bu split isolation; source registry, annotation workflow, reviewer ve retention
politikasıyla upstream'te sağlanmalıdır. Production dataset'in exact
train/validation **raw-record** leakage audit'i bu semantic/annotation isolation
kontrolünün yerine geçmez.

### 2. Immutable policy: `vegai-pretraining-quality-policy/v1`

```json
{
  "schema_version": "vegai-pretraining-quality-policy/v1",
  "policy_id": "support-release-policy-r2026-10",
  "minimum_cases": 200,
  "minimum_top1_accuracy": 0.82,
  "minimum_top1_wilson_lower_bound_95": 0.75,
  "minimum_mean_similarity": 0.12,
  "minimum_case_similarity": -0.10,
  "maximum_incorrect_cases": 36
}
```

Tüm alanlar zorunludur; evaluator içinde gizli default threshold yoktur.

| Alan | Anlamı |
|---|---|
| `minimum_cases` | Gate'e dahil edilmesi gereken minimum unique case sayısı (`1..4096`) |
| `minimum_top1_accuracy` | `correct_top1_cases / case_count` alt sınırı (`0..1`) |
| `minimum_top1_wilson_lower_bound_95` | Bernoulli Top-1 hit oranının sabit two-sided %95 Wilson lower bound alt sınırı (`0..1`); küçük case setindeki mükemmel raw oranı tek başına yeterli sayılmaz |
| `minimum_mean_similarity` | Top-1 observed bipolar similarity aritmetik ortalaması alt sınırı (`-1..1`) |
| `minimum_case_similarity` | Her bir case için gereken en düşük similarity (`-1..1`) |
| `maximum_incorrect_cases` | Kabul edilen maksimum Top-1 yanlış sayısı; `minimum_cases`ten küçük olmalıdır |

Bu eşikler target-domain baseline/candidate deneyleri, bağımsız held-out
calibration ve risk değerlendirmesiyle seçilmelidir. Repository fixture veya
başka bir domain'e ait eşik production policy değildir.

### 3. HMAC redaction key

`--redaction-key-file` boş olmayan, en fazla 4 KiB local secret regular-file
target ister (secret-manager symlink'i hedefiyle birlikte snapshot edilir). Key
case ID/query/expected/observed tokenların yerine report'ta
namespace-separated `HMAC-SHA-256` identifier üretir. Key byte'ı, key path'i ve
key hash'i report/run manifest'e yazılmaz.

Key'i repository'ye koymayın; secret manager veya erişim kontrollü volume
kullanın. Key rotasyonu semantic gate kararını değiştirmez ancak report'taki
HMAC ID'lerini değiştirir; cross-report case-level correlation gerekiyorsa
retention/audit döneminde aynı key yönetilmelidir.

## Doğrudan checkpoint değerlendirmesi

```bash
python3 tools/pretraining_quality_evaluation.py \
  --pretrain-bin ./build/vegai_pretrain \
  --checkpoint production-runs/support-r2026-10/state/generations/generation-00000000000000000042-....vgai \
  --cases /secure/evaluations/support-r2026-10-cases.json \
  --policy /secure/evaluations/support-r2026-10-policy.json \
  --redaction-key-file /secure/secrets/support-quality-hmac.key \
  --output production-runs/support-r2026-10/quality-report.json \
  --scratch-dir /fast-local-ssd/vegai-quality-scratch \
  --timeout-seconds 120
```

`--scratch-dir` verilmezse OS temporary directory kullanılır. Private evaluation
aynı anda checkpoint'in bir kopyası, binary'nin bir kopyası ve küçük query
metadata'sı için yer ister; büyük checkpoint'lerde SSD/local-volume kapasitesini
ayrı planlayın. Evaluator private checkpoint snapshot'ını şu an 2 GiB ile
fail-closed sınırlar; daha büyük artifact için rastgele limiti aşmak yerine
storage/threat-model ve versioned evaluator policy'sini gözden geçirin. Scratch
path report'a yazılmaz.

Exit code sözleşmesi:

| Code | Sonuç |
|---:|---|
| `0` | Input/provenance geçerli, real inspect tamamlandı ve tüm eşikler geçti |
| `3` | Geçerli report atomik yazıldı, fakat en az bir quality eşiği geçmedi |
| `2` | Schema, input identity, CLI output, timeout veya provenance hatası; güvenilir report yayınlanmadı |

`inspect` stdout'u durable report'a konmaz. Bir batch `--query-file` çağrısı
kullanılır; evaluator initially fingerprinted binary ve checkpoint byte'larından
private temporary snapshot üretir, child'ı bu snapshot'larla çalıştırır ve
orijinal dependency'leri ardından tekrar doğrular. Query block sayısı, sıra,
token grammar ve similarity `[-1,1]` sınırı strict parse edilir. Child stdout
16 MiB ile in-memory bounded capture edilir; cap aşımı child'ı terminate eder ve
report yayınlanmaz. Eksik/reordered/unavailable/ambiguous block fail-closed olur.

## Privacy-conscious report

`vegai-pretraining-quality-report/v1` şunları taşır:

- evaluator, pretrain binary, checkpoint, cases ve policy için SHA-256 + byte
  count provenance;
- `declared_split`, HMAC-redacted evaluation ID, case count;
- `correct_top1_cases`, `incorrect_top1_cases`, Top-1 accuracy, sabit %95
  Wilson lower bound ve mean/min/max bipolar similarity;
- her case için HMAC-redacted ID/query/acceptable-set/observed-neighbour,
  boolean Top-1 hit ve similarity;
- inspect output SHA-256/byte count/elapsed time; explicit threshold listesi
  ve gate failure listesi.

Raw case ID, token, acceptable label, observed label, HMAC key, HMAC-key path,
raw inspect stdout ve operator filesystem path'i report'a yazılmaz. Bu leakage
yüzeyini azaltır; sayı/metriklerin kendisinin hassas olup olmadığı ve report
retention/access policy'si yine operatör sorumluluğundadır.

## Production integration

`tools/production_pretrain.py`, üç quality input'un **hepsi** verildiğinde
gate'i immutable run policy/fingerprint'e bağlar ve completed final checkpoint
sonrasında evaluator'ı çalıştırır:

```bash
python3 tools/production_pretrain.py \
  --pretrain-bin ./build/vegai_pretrain \
  --dataset production-datasets/support-r2026-10/dataset.json \
  --run-dir production-runs/support-r2026-10-hdc \
  --max-vocab 20000 --min-frequency 1 --context-mode hierarchical --epochs 1 \
  --quality-cases /secure/evaluations/support-r2026-10-cases.json \
  --quality-policy /secure/evaluations/support-r2026-10-policy.json \
  --quality-redaction-key-file /secure/secrets/support-quality-hmac.key \
  --quality-scratch-dir /fast-local-ssd/vegai-quality-scratch \
  --require-quality-gate
```

Varsayılan quality-case safety bound `4096`dır; daha düşük bir operational
bound için `--quality-max-cases N` verin. Bound aşan manifest training başlamadan
reddedilir ve bound immutable run identity'ye girer. `--quality-scratch-dir`
verilirse preflight, resource-plan'ın estimated maximum checkpoint'i + binary +
bounded query metadata için local free-space gate uygular; evaluator öncesi
actual final checkpoint boyutuyla tekrar doğrular. Verilmezse aynı kontrol OS
temp directory üzerinde yapılır.

- `run.json.fingerprint.policy.quality_gate`, evaluator source SHA, case/policy
  SHA+byte count, case count, declared split, thresholds, `--quality-max-cases`
  safety bound'u, scratch directory ve timeout'u bağlar.
  Case/policy/evaluator değiştiyse `--resume-run` reddedilir.
- Redaction key yalnız output privacy aracıdır ve fingerprint'e yazılmaz; ancak
  invocation sırasında valid/stable olmalıdır.
- Completed candidate'da `quality-report.json` atomik yazılır. Production
  wrapper report'un schema'sını, duplicate key olmamasını, evaluator/binary/
  checkpoint/case/policy SHA bağlarını, redacted case row'larından yeniden
  hesaplanan aggregate/Wilson eşik kararını ve exit-status ↔ gate kararını
  doğrular.
- Gate geçerse `result.json.quality_gate.status=passed` ve
  `release_eligible=true` olur. Gate eşiklerden kalırsa eğitim state'i
  recoverable kalır fakat `failure.json` ve latest `result.json`
  `status=failed`, `quality_gate.status=failed` ile atomik yayınlanır; process
  non-zero biter. Böylece trained-but-unqualified checkpoint release sonucu
  gibi görünmez.
- `--max-shards` ile henüz tüm immutable train shard'ları uygulanmadıysa gate
  `deferred` olur ve `release_eligible=false` kalır; final complete invocation
  gate'i çalıştırır. Bu operational partial batch'i model-quality failure ile
  karıştırmaz.
- Quality input vermeyen eski operational workflow desteklenir, fakat
  `quality_gate.status=not-configured` ve `release_eligible=false` taşır.
  `--require-quality-gate` ile bu ungated mod açıkça reddedilir.

`elapsed_train_controller_seconds` quality inspect/report süresini içermez;
yalnız real `continuous_pretrain.py` child controller wall-clock interval'idir.
Quality report'taki `inspect.elapsed_seconds` ayrı bir evaluation ölçüsüdür ve
training throughput olarak yorumlanmamalıdır.

## Testler ve sınırlar

```bash
make quality-check
# veya checked CMake build sonrası:
ctest --test-dir build --output-on-failure -R pretraining_quality_evaluation
```

`tools/verify_pretraining_quality_evaluation.py` şu negatif/positive contract'leri
doğrular: strict schema/duplicate JSON key rejection, HMAC report privacy,
production wrapper aggregate-metric recomputation/tamper rejection, intentional
threshold/Wilson failure'da report publication, bounded stdout ve timeout
termination, redaction-key TOCTOU rejection ve gerçek compiled HDC checkpoint
ile batch inspect ABI.
`tools/verify_production_pretraining.py` ayrıca gerçek exactly-once production
run üzerinde quality pass/fail/deferred manifest davranışını, policy-swap resume
rejection'ını ve recoverable state davranışını doğrular.

Bu mekanizma signed remote attestation, external benchmark bağımsızlığı,
semantic-near-duplicate detection, evaluator key escrow, multi-host consensus,
object-store immutability veya live traffic enforcement sağlamaz. Bunlar
production threat model'inde ayrı kontrol katmanlarıdır.
