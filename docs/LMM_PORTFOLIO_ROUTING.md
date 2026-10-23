# Domain-expert LMM portföy routing

## Amaç ve dürüst sınır

Tek bir LMM'in dil, kod, matematik, hukuk, sağlık, multimodal reasoning, latency,
maliyet ve güvenlikte her zaman en iyi olduğu varsayımı güvenilir değildir.
`tools/lmm_portfolio.py`, bu sorunu model adına veya pazarlama iddiasına göre
değil; operatorün yönettiği **hedef-domain held-out ölçüm kartları** ile çözen
fail-closed bir kontrol düzlemidir.

Bu katman şunları yapar:

- gerçek immutable model manifest'ini, gateway argv'sini ve geçmişte başarılı
  olmuş citation/abstention quality report'unu birbirine bağlar;
- domain, data residency, safety tier, modality, context, output budget,
  latency ve maliyet sınırını geçmeyen modeli reddeder;
- eligible modeller arasından deterministik olarak en yüksek measured lower
  bound'a sahip modeli seçer;
- zor görevlerde aynı provider/model family tekrarını önleyen bağımsız panel
  planı üretebilir;
- `generate` modunda yalnız `single-best` seçimini gerçek gateway'e aktarır;
  model yanıtına prompt, template veya hardcoded cevap eklemez.

Bu katman **evrensel zekâ veya bütün LMM'lerden üstünlük kanıtı değildir**.
Evaluation card içindeki broad quality/reliability/safety değerlerinin anlamı,
suite tasarımı, contamination kontrolü ve confidence interval'ı operatorün
sorumluluğundadır. Router yalnız bu değerlerin doğru model, report ve policy
ile tutarlı bağlandığını denetler. Citation quality report ise yalnız supplied
evidence marker contract'ini ölçer; factual entailment yerine geçmez.

## 1. Bileşenler

| Bileşen | Sorumluluk |
|---|---|
| `tools/lmm_gateway_template.py` | Shell kullanmayan `{request}` / `{evidence}` argv template'i; duplicate/redirected placeholder, `--output` ve response-retention seçeneğini reddeder |
| `tools/lmm_portfolio.py route` | Ölçülmüş domain expert veya bağımsız panel seçimi; raw request taşımayan `vegai-lmm-route-decision/v1` raporu |
| `tools/lmm_portfolio.py generate` | `single-best` kararıyla private request/evidence snapshot'ı üzerinden gerçek gateway çağrısı; model ve input provenance'ını yeniden doğrular |
| `tools/lmm_quality_evaluation.py` | Candidate'in citation/abstention held-out release report'unu üretir; portföy seçimi bu report'un manifest SHA'sını doğrular |
| `tools/verify_lmm_portfolio.py` | Synthetic worker ile domain selection, gerçek dispatch, panel bağımsızlığı, raw-text privacy ve fail-closed negative cases |

Tüm gerçek modeller, gateway command'leri, evaluation suite'leri ve scorecard
sayısal değerleri operator tarafından sağlanır. Depo model weight veya uydurma
benchmark sonucu içermez.

## 2. Sıkı veri sözleşmeleri

### 2.1 Portfolio — `vegai-lmm-portfolio/v1`

Portfolio, relative dosya yollarını kendi JSON dosyasının dizinine göre çözer.
Absolute path de kullanılabilir. Tüm candidate'ler aynı `cost_unit` ile ölçülmüş
olmalıdır; örneğin `usd-microunits` veya kuruluşun tanımladığı
`internal-credit-microunits`.

```json
{
  "schema_version": "vegai-lmm-portfolio/v1",
  "id": "production-domain-portfolio-2026a",
  "cost_unit": "usd-microunits",
  "candidates": [
    {
      "id": "coding-expert-v3",
      "domains": ["coding", "general"],
      "independence_group": "provider-a-family-x",
      "data_residencies": ["TR", "EU"],
      "safety_tier": 3,
      "grounding_mode": "evidence-or-abstain",
      "model_manifest": "/secure/models/coding-v3.json",
      "gateway_command_json": "/secure/gateway/coding-v3.json",
      "evaluation_card": "/secure/evals/coding-v3-card.json",
      "quality_report": "/secure/evals/coding-v3-grounding.json"
    }
  ]
}
```

`independence_group`, gerçek bağımsızlık iddiasını temsil eden operator-owned
bir family/provider etiketi olmalıdır. Aynı hosted modelin farklı endpoint veya
quantization'ları farklı group diye gösterilmemelidir.

### 2.2 Evaluation card — `vegai-lmm-evaluation-card/v1`

Her card tam olarak bir model manifest SHA'sına ve bir başarılı held-out
citation report SHA'sına bağlanır. Quality report'un
`dependencies.gateway_command.sha256` alanı da candidate'in çalıştıracağı
command JSON hash'iyle bire bir aynı olmalıdır; farklı decoding, timeout,
endpoint veya worker seçeneği için quality report yeniden üretilir. Her domain
için yalnız bir measurement kabul edilir; aynı domain'deki suite seçimi/risk
analizi operatorün yayınlanmış eval protokolünde yapılmalıdır.

```json
{
  "schema_version": "vegai-lmm-evaluation-card/v1",
  "id": "coding-v3-card-2026a",
  "cost_unit": "usd-microunits",
  "model_manifest_sha256": "<64-lowercase-hex>",
  "quality_report_sha256": "<64-lowercase-hex>",
  "measurements": [
    {
      "domain": "coding",
      "suite_id": "governed-coding-heldout-2026a",
      "suite_report_sha256": "<64-lowercase-hex>",
      "cases": 1200,
      "quality_lower_bound": 0.91,
      "reliability_lower_bound": 0.88,
      "safety_lower_bound": 0.96,
      "p95_latency_ms": 850,
      "cost_microunits_per_1k_output_tokens": 2200
    }
  ]
}
```

`lower_bound` alanları point estimate değil, operatorün yayımladığı güven aralığı
alt sınırları olmalıdır. `suite_report_sha256`, ayrı immutable eval artifact'ini
bağlar; router bu artifact'in istatistiksel yöntemini kendiliğinden doğrulamaz.

### 2.3 Routing task — `vegai-lmm-routing-task/v1`

Task, kullanıcı metnini taşımaz; gerçek `vegai-lmm-request/v1` ayrı `--request`
dosyasındadır. Bu ayrım policy report'unda raw prompt tutmayı önler.

```json
{
  "schema_version": "vegai-lmm-routing-task/v1",
  "id": "secure-coding-assist",
  "domain": "coding",
  "data_residency": "TR",
  "minimum_safety_tier": 3,
  "strategy": "single-best",
  "minimum_independent_candidates": 1,
  "required_modalities": ["text"],
  "require_evidence": true,
  "minimum_context_window_tokens": 32768,
  "minimum_quality_lower_bound": 0.85,
  "minimum_reliability_lower_bound": 0.80,
  "minimum_safety_lower_bound": 0.90,
  "max_p95_latency_ms": 1500,
  "max_cost_microunits_per_1k_output_tokens": 5000
}
```

`minimum_context_window_tokens`, request için operatorün hesapladığı worst-case
capacity gereksinimidir. Gateway, gerçek tokenizer/context packing ile ikinci
bir fail-closed kontrol yapar. Request image/audio/video asset ID'si taşıyorsa
`route`/`generate` çağrısındaki `--asset-registry`, request'in gereken asset'lerini
çözmek zorundadır; seçilmiş gateway report'u aynı registry SHA-256'sını döndürmek
zorundadır.

## 3. Gateway command sözleşmesi

Her candidate command JSON string array olmalı ve aşağıdaki koşulları aynı anda
sağlamalıdır:

- tam olarak bir `--request {request}` ve bir `--evidence {evidence}`;
- tam olarak bir `--print-json`;
- candidate manifest'iyle byte aynı yolu gösteren tam olarak bir `--model`;
- declared `grounding_mode` ile aynı `--grounding-mode` (yoksa `off`);
- `--output` ve `--include-response-in-report` **olmamalı**.

Örnek:

```json
["python3", "tools/lmm_gateway.py",
 "--model", "/secure/models/coding-v3.json",
 "--request", "{request}", "--evidence", "{evidence}",
 "--worker-command-json", "/secure/workers/coding-v3.json",
 "--grounding-mode", "evidence-or-abstain", "--print-json"]
```

Placeholder'lar yalnız private temporary file path ile değiştirilir; shell,
string interpolation veya model output'una dayalı argv oluşturulmaz.

## 4. Deterministik seçim

Candidate önce şu hard gate'lerden geçmelidir:

1. task domain ve data residency candidate tarafından açıkça ilan edilmiş olmalı;
2. model manifest required modality, context ve `max_new_tokens` ihtiyacını
   karşılamalı;
3. candidate safety tier ve measured lower bound'lar task threshold'larını
   geçmeli;
4. p95 latency ve aynı `cost_unit` içindeki cost cap aşılmamalı;
5. `require_evidence=true` ise evidence bundle boş olmamalı ve grounding mode
   `off` olmamalı;
6. evaluation card, immutable model manifest ve başarılı
   `vegai-lmm-quality-report/v1` aynı manifest SHA'sına bağlı olmalı.

Eligible modeller şu sabit sırayla karşılaştırılır:

1. yüksek `quality_lower_bound`;
2. yüksek `reliability_lower_bound`;
3. yüksek `safety_lower_bound`;
4. düşük p95 latency;
5. düşük output-token maliyeti;
6. lexical candidate ID tie-break.

`independent-panel` stratejisi aynı sıralamayı kullanır ancak her seçilen
candidate farklı `independence_group` taşımak zorundadır. Bu komut cevapları
birleştirmez, çoğunluk oylamasını factuality kanıtı diye sunmaz ve gizli prompt
üretmez. Bunun yerine source-to-answer trace, human review veya operator-owned
adjudicator workflow için güvenli, provenance-bound bir panel planı verir.

## 5. Operasyon akışı

1. Her gerçek model için `lmm_quality_evaluation.py --fail-on-gate` ile frozen
   citation/abstention suite çalıştırın.
2. Ayrı governed target-domain suite, safety/red-team ve latency/cost
   ölçümlerini yapın; immutable suite report digest'i içeren evaluation card
   yayınlayın.
3. Model manifest, gateway command, evaluation card ve quality report'u
   portfolio'ya ekleyin.
4. Her gerçek request için önce route kararı üretin:

```bash
python3 tools/lmm_portfolio.py route \
  --portfolio /secure/portfolio.json --task /secure/coding-task.json \
  --request request.json --evidence evidence.json \
  --output lmm-runs/coding-route.json
```

5. Task `single-best` ise aynı bounded karar üzerinden gerçek model çağrısı:

```bash
python3 tools/lmm_portfolio.py generate \
  --portfolio /secure/portfolio.json --task /secure/coding-task.json \
  --request request.json --evidence evidence.json \
  --output lmm-runs/coding-generation.json --print-json
```

Generation report default olarak response SHA-256/byte count saklar; raw text
ancak açık `--include-response-in-report` ile eklenir. `--print-json`, response'u
operatorün stdout consumer'ına iletir; held-out evaluation için bu metni kalıcı
loglara yazmamak operatorün sorumluluğundadır.

## 6. Fail-closed ve provenance davranışı

Router report yayınlamaz ve exit `2` ile durur, eğer:

- hiç candidate bütün sınırları geçemezse;
- panel için yeterli bağımsız group yoksa;
- model manifest, quality report, evaluation card veya gateway command değişirse;
- output, router input'u, model/tokenizer, candidate command/worker config ya da
  declared model/asset package root'u içindeki immutable bir path'i overwrite
  etmeye çalışırsa;
- quality report başarısızsa veya başka model manifest'e aitse;
- command başka bir `--model` kullanıyorsa;
- generate aşamasında gateway farklı request/evidence/asset-registry/model/
  grounding mode provenance'ı döndürürse veya gateway tool/tokenizer/worker/
  model-artifact runtime identity'si held-out quality report'tan farklıysa.

Report; router, portfolio, task, request, evidence, asset registry, model,
command, evaluation card ve quality report identity'lerini taşır. Böylece
counterfactual/shadow routing karşılaştırması için denetlenebilir input zinciri
oluşur. Ancak real provider'ın endpoint arkasındaki weight revision'ını manifest
dışında değiştirmediği ayrıca provider attestation ile kanıtlanmalıdır.

## 7. Doğrulama

```bash
python3 tools/verify_lmm_portfolio.py
make lmm-check
```

Verifier; en yüksek measured coding candidate'in seçilmesini, gerçek selected
gateway dispatch'ini, raw prompt/response privacy default'unu, iki farklı
independence group ile paneli, threshold altında güvenli reddi, başka model
yoluna bağlı veya held-out report'ta ölçülmemiş command rejection'ını,
immutable worker-config output collision'ını ve held-out runtime'dan sapmış
worker config rejection'ını test eder. Synthetic worker yalnız protocol
fixture'idir; gerçek domain score veya LMM üstünlüğü sonucu değildir.
