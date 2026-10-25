# Doğrulanmış araç orkestrasyonu

## Amaç ve dürüst sınır

`tools/lmm_tool_orchestrator.py`, gerçek bir LMM'in (veya yetkili bir operatörün)
önerdiği **araç planını** çalıştırır. Araç sonucu, LMM yanıtı veya gizli düşünce
zinciri üretmez. Bunun yerine planı typed sözleşmelerle doğrular, yalnız operator
registry'sindeki salt-okunur araçları shell olmadan çalıştırır, ayrı verifier
programının bağlandığı sonucu kontrol eder ve sonucu
`vegai-lmm-evidence/v1` bundle'ı olarak yayınlar.

Bu katman; hesaplama, code-analysis, arama, veri tabanı sorgusu veya domain
worker gibi gerçek araçlarla çalışan LMM'lere daha fazla **doğrulanabilir işlem
kapasitesi** verir. Bir araç çağrısının veya verifier'ın geçmesi şunları
kanıtlamaz:

- modelin planının semantik olarak iyi, eksiksiz veya genel olarak zeki olduğunu;
- dış kaynağın güncel/faktüel olduğunu;
- verifier bağımsızlık grubunun gerçek hayatta bağımsız implementasyon/audit
  olduğu iddiasını;
- OS sandbox, network isolation, authorization veya side-effect güvenliğini;
- araç evidence'ine dayanan son model cümlesinin doğru olduğunu.

Bunlar için target-domain held-out tool-use evaluation, gerçek sandbox/container
politikası, kaynak doğrulama, red-team ve human approval gerekir.

## Neden plan / execute / verify ayrımı?

```text
actual LMM or operator
        │  immutable typed plan (no hidden CoT required)
        ▼
tool registry + input/output contracts ──► execute (no shell)
                                             │
                                             ▼
                                      independent verifier
                                             │
                                             ▼
              provenance-bound EvidenceBundle ──► lmm_gateway.py
```

1. **Plan:** request ve registry digest'ine bağlı invocation sırası ve typed
   argümanlar seçilir.
2. **Execute:** registry içindeki executor, private input/output dosyalarıyla
   timeout altında çalışır.
3. **Verify:** executor'dan farklı implementation hash'i ve farklı declared
   `independence_group` taşıyan verifier, exact result digest için `pass`
   attestation üretir.
4. **Ground:** yalnız pass eden sonuç, gateway'e verilebilecek source/license/hash
   taşıyan evidence item'a dönüşür.

Plan yalnız observable action graph'ıdır; modelin private chain-of-thought'unu
istemek veya saklamak gerekmez. Bir önceki pass eden output'a JSON Pointer ile
bağlanan sonraki invocation, çok-adımlı hesaplama/araç akışını sağlar.

## Hızlı kullanım

Önce registry'nin, schema'ların ve trusted executor/verifier implementation
files'larının aynı versioned operator directory'sinde olduğunu varsayın:

```bash
python3 tools/lmm_tool_orchestrator.py execute \
  --registry governed-tools/registry.json \
  --request request.json \
  --plan model-proposed-plan.json \
  --evidence-output tool-evidence.json \
  --report-output tool-execution-report.json

python3 tools/lmm_gateway.py \
  --model /secure/model-package/model.manifest.json \
  --model-root /secure/model-package \
  --request request.json \
  --evidence tool-evidence.json \
  --output generation-report.json
```

İlk komut successful olduğunda `tool-evidence.json`, normal
`vegai-lmm-evidence/v1` sözleşmesidir. Gateway bu evidence'i pinned source data
olarak pack eder; model, gerekiyorsa kendi output'unda
`[[vegai-source:tool-<invocation-id>]]` marker'ını üretmelidir. Orchestrator
marker veya final natural-language answer eklemez.

Aynı evidence/report output path'i ikinci kez kullanılırsa işlem overwrite etmez.
Geçersiz plan, tool/verifier hatası veya input değişimi için exit `2` döner ve
başarılı evidence/report pair'i yayınlanmaz.

## 1. Registry sözleşmesi

Root schema: `vegai-lmm-tool-registry/v1`

```json
{
  "schema_version": "vegai-lmm-tool-registry/v1",
  "id": "operator-tools-v1",
  "tools": [
    {
      "id": "checked-calculator",
      "source": "operator-reviewed arithmetic worker",
      "license": "operator-owned",
      "side_effect": "none",
      "input_schema": {"path": "calculator-input.json", "sha256": "..."},
      "output_schema": {"path": "calculator-output.json", "sha256": "..."},
      "executor": {
        "implementation": {"path": "calculator_executor.py", "sha256": "..."},
        "command": ["python3", "{implementation}", "--input", "{input}", "--output", "{output}"],
        "independence_group": "calculator-executor-v1"
      },
      "verifier": {
        "implementation": {"path": "calculator_verifier.py", "sha256": "..."},
        "command": ["python3", "{implementation}", "--input", "{input}", "--result", "{result}", "--output", "{output}"],
        "independence_group": "calculator-verifier-v1"
      },
      "timeout_seconds": 30,
      "maximum_result_bytes": 262144,
      "max_invocations_per_plan": 4
    }
  ]
}
```

Kurallar:

- registry en çok 64 tool taşır; tool id'leri unique ASCII identifier'dır;
- implementation ve schema `path` alanları registry directory'sine göre relative
  olur; `..`, absolute path ve bütün symlink traversal'ları reddedilir;
- her member'ın gerçek SHA-256'sı manifestteki digest ile eşleşir ve publish
  öncesi yeniden hash'lenir;
- executor/verifier implementation hash'leri ve declared independence groups
  farklı olmak zorundadır. Bu syntactic ayrım gerçek organizational independence
  veya formal proof değildir;
- command JSON string array'dir. Shell, interpolation ve partial placeholder
  yoktur. Executor yalnız `{implementation}`, `{input}`, `{output}`; verifier
  bunlara ek olarak `{result}` placeholder'larını tam bir argv item olarak birer
  kez taşır;
- process child'ın `stdin/stdout/stderr`i tutulmaz. Tool sonucu yalnız bounded,
  private output JSON dosyası üzerinden alınır; `maximum_result_bytes` en fazla
  1 MiB'dır çünkü pass eden output doğrudan EvidenceItem text'ine dönüşür;
- `side_effect` bu sürümde yalnız `none` olabilir. E-posta, ödeme, deployment,
  dosya mutasyonu veya dış state değiştiren tool'lar **bilinçli olarak desteklenmez**.

`PATH`, locale ve UTC dışında child'a ambient environment aktarılmaz; credential,
API key, HOME veya user prompt otomatik aktarılmaz. Bu bir least-privilege
başlangıcıdır; tool'un network veya filesystem erişimini kernel düzeyinde
sınırlamaz.

## 2. Typed value schema

Input ve output schema root'u `vegai-lmm-tool-value-schema/v1` olur:

```json
{
  "schema_version": "vegai-lmm-tool-value-schema/v1",
  "id": "calculator-input-v1",
  "root": {
    "type": "object",
    "fields": {
      "left": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
      "right": {"type": "integer", "minimum": -1000000, "maximum": 1000000}
    },
    "required": ["left", "right"],
    "additional_properties": false
  }
}
```

Desteklenen node türleri:

| `type` | Zorunlu alanlar | Sınır |
|---|---|---|
| `string` | `min_bytes`, `max_bytes` | UTF-8, CR/NUL/control reject |
| `integer` | `minimum`, `maximum` | signed 64-bit aralık |
| `number` | `minimum`, `maximum` | finite IEEE numeric input |
| `boolean` | yalnız `type` | strict JSON bool |
| `array` | `max_items`, `items` | en çok 256 item |
| `object` | `fields`, `required`, `additional_properties:false` | en çok 64 named field |

Schema derinliği 8, toplam schema node sayısı 256 ile bounded'tır. Bilinmeyen
field veya type coercion kabul edilmez. Bu contract tool implementasyonunun
semantik doğruluğu değildir; yalnız input/output boundary'yi auditable yapar.

## 3. Plan ve güvenli dataflow

Root schema: `vegai-lmm-tool-plan/v1`.

```json
{
  "schema_version": "vegai-lmm-tool-plan/v1",
  "id": "request-42-calculation",
  "request_sha256": "<canonical GenerationRequest digest>",
  "registry_sha256": "<exact registry file digest>",
  "invocations": [
    {
      "id": "first",
      "tool_id": "checked-calculator",
      "input": {"left": 2, "right": 3},
      "idempotency_key": "<unique 64-char digest>"
    },
    {
      "id": "second",
      "tool_id": "checked-calculator",
      "input": {
        "left": {"$vegai_ref": "first#/value"},
        "right": 7
      },
      "idempotency_key": "<another unique 64-char digest>"
    }
  ]
}
```

`request_sha256`, raw request file hash'i değil canonical
`GenerationRequest.digest()` değeridir; bu whitespace/JSON formatting farkını
ayırır ve planı gerçek user/session request'ine bağlar. `registry_sha256` ise
exact registry bytes'ını pinler.

`$vegai_ref` yalnız şu formatta tam bir object olabilir:

```json
{"$vegai_ref": "<previous-invocation-id>#/RFC6901/json/pointer"}
```

- Referans yalnız listedeki **önceki**, pass eden invocation output'una gider;
  current/future/unknown invocation, cycle veya invalid pointer reddedilir.
- Referans çözüldükten sonra hedef tool input schema'sı tekrar doğrulanır.
- Her invocation id ve idempotency key plan içinde unique olmalıdır.
- Tool başına registry `max_invocations_per_plan` sınırı uygulanır.

Bu idempotency key yalnız immutable plan intent'i bağlayan bir key'dir. Bu sürüm
mutating tool çalıştırmadığından external side-effect idempotency garantisi
sunmaz.

## 4. Executor ve verifier protocol'leri

Executor'a private dosyada aşağıdaki object verilir:

```json
{
  "schema_version": "vegai-lmm-tool-invocation/v1",
  "id": "first",
  "tool_id": "checked-calculator",
  "request_sha256": "...",
  "idempotency_key": "...",
  "input": {"left": 2, "right": 3}
}
```

Executor, belirtilen output path'e yalnız şu exact root'u yazmalıdır:

```json
{
  "schema_version": "vegai-lmm-tool-result/v1",
  "id": "first",
  "tool_id": "checked-calculator",
  "output": {"value": 5}
}
```

Orchestrator output'u declared schema'ya göre doğrular, canonical JSON SHA-256
result digest'ini üretir ve separate private snapshot'u verifier'a verir.
Verifier'ın output'u aşağıdaki şekildedir:

```json
{
  "schema_version": "vegai-lmm-tool-verification/v1",
  "id": "first",
  "tool_id": "checked-calculator",
  "result_sha256": "<canonical exact result digest>",
  "verdict": "pass"
}
```

`verdict` tam olarak `pass` değilse, verifier input/result snapshot'ı değişirse,
result identity uyuşmazsa veya child timeout/nonzero exit ile biterse bütün plan
fail-closed olur.

## 5. Evidence ve privacy-conscious execution report

Başarılı planın her pass eden invocation'ı bir `EvidenceItem` olur:

- `id`: `tool-<invocation-id>`;
- `text`: typed `output` object'in canonical JSON text'i;
- `source` / `license`: immutable registry metadata'sı;
- `sha256`: text digest'i;
- `score`: `1.0`; semantic confidence veya factual truth score değildir.

Evidence dosyası model context'i için tasarlanmıştır ve tool result'ını içerir.
Bunun aksine execution report raw request, raw plan arguments, raw result veya
child stdout/stderr yazmaz. Report şunları taşır:

- redacted request/registry/plan file identities;
- logical request digest, registry/plan SHA-256;
- invocation/tool id, input/result/verification digest'leri;
- executor/verifier implementation digest'leri, elapsed milliseconds ve
  derived evidence id;
- evidence artifact identity ve açık sınırlar.

Output çifti no-replace yayınlanır. Report publication başarısız olursa, henüz
orchestrator'ın yazdığı değişmemiş evidence artifact geri alınmaya çalışılır.

## 6. Fail-closed ve deployment sınırları

Aşağıdaki durumlarda exit `2` ve no successful output alınır:

- duplicate JSON key, strict schema/type/size limit ihlali;
- request/registry binding mismatch, unknown tool veya too-many invocation;
- absolute/escaping/symlink registry member, digest swap veya input TOCTOU;
- shell benzeri/partial/duplicate placeholder command;
- forward/cyclic reference, type mismatch veya tool output schema mismatch;
- executor/verifier timeout, nonzero exit, output missing, verifier fail;
- existing/protected output path.

Araç programı 900 saniyeye kadar registry timeout'una uyar, child output'unu
retention yapmaz ve private temporary work directory kullanır. Ancak child'ın
memory/CPU/network/filesystem limitleri, process namespace'i, syscall policy,
remote service identity'si ve human confirmation production sandbox/runtime
katmanında uygulanmalıdır. Bunları mevcut olduğunu iddia etmez.

## 7. Doğrulama

```bash
python3 tools/verify_lmm_tool_orchestrator.py
make tool-check
```

Synthetic verifier, gerçek model veya domain answer kullanmadan şunları test eder:

- Unicode typed input ile two-step predecessor reference (`first#/value`);
- distinct executor/verifier implementation ve exact result binding;
- evidence/request provenance ve raw args/resultsiz execution report;
- output collision, forged request binding, forward reference, verifier-fail,
  verifier snapshot mutation, forged implementation digest ve duplicate JSON-key
  negative gate'leri.

Bu contract testinin geçmesi, operatorün gerçek tool'larının güvenli, doğru veya
her domain'de yararlı olduğunu göstermez. Her gerçek registry için separately
held-out tool-use suite, attack simulation, tool availability/SLO testleri ve
human authorization policy gerekir.
