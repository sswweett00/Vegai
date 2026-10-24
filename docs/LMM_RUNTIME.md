# Vegai LMM entegrasyon düzlemi

Bu belge, Vegai HDC çekirdeğinin yerine geçtiğini iddia etmeyen; gerçek bir
harici LMM/LLM/VLM runtime'ını güvenli, Unicode-aware ve provenance-bound
şekilde bağlayan yeni entegrasyon düzlemini tanımlar.

> **Dürüst sınır:** Depoda model ağırlığı, Transformer tensor runtime'ı,
> autodiff/optimizer, lisanslı web-scale corpus veya evrensel multimodal model
> bulunmaz. Bu katman, operatorün sahip olduğu gerçek model package/backend ile
> çalışır. Backend yoksa `lmm_gateway.py` cevap üretmez, fail-closed olur.

Bu ayrım önemlidir: HDC/Retrieval yolunun deterministic, source-owned özellikleri
korunurken; serbest doğal dil üretimi, Unicode/subword tokenization, uzun sohbet
context'i ve medya reasoning işi gerçek neural modele devredilir.

## 1. Yeni mimari

```text
strict session JSONL ─────┐
                           ├── lmm_session.py ── vegai-lmm-request/v1
Unicode documents JSONL ──┼── lmm_knowledge.py ─ vegai-lmm-evidence/v1
image/audio/video bytes ──┼── lmm_assets.py ──── asset registry
model/operator tool plan ─┼── lmm_tool_orchestrator.py ─ verified tool evidence
                           │
model manifest + tokenizer ┴── lmm_gateway.py
                                  │
                  ┌───────────────┼────────────────┐
                  │               │                │
        OpenVINO GenAI text   HTTPS OpenAI-      trusted local
          model package       compatible API    multimodal worker
                  │               │                │
                  └──── actual model-generated response ────┘
```

LMM düzlemi şu bileşenlerden oluşur:

| Bileşen | Sorumluluk |
|---|---|
| `tools/lmm_contract.py` | Versioned schema, Unicode doğrulama, model/tokenizer/asset manifest'i, exact token-budget context packer, worker protocol ve atomic report araçları |
| `tools/lmm_model_manifest.py` | Gerçek local model package'ini recursively hash'leyip manifest üretir ve inference öncesi artifact doğrulaması yapar |
| `tools/lmm_session.py` | Hash-chain ile korunmuş uzun sohbet geçmişi; history'yi `vegai-lmm-request/v1` haline getirir |
| `tools/lmm_knowledge.py` | Operator-owned Unicode JSONL belgelerinden SQLite FTS5 indeks üretir; kaynak/izin/hash taşıyan evidence bundle döndürür |
| `tools/lmm_assets.py` | Görüntü/ses/video dosyalarını content-addressed store'a alır; MIME magic, source/license ve immutable hash kaydeder |
| `tools/lmm_gateway.py` | Context + evidence + verified asset'leri gerçek backend'e yollar; model/request/evidence/assets provenance report üretir |
| `tools/benchmark_lmm_gateway.py` | Sadece gerçek gateway child-process sürelerini ve backend token sayısını ölçer |
| `tools/lmm_gateway_template.py` | Shell kullanmayan shared `{request}`/`{evidence}` gateway argv template doğrulaması; redirect/duplicate placeholder ve response-retention'i reddeder |
| `tools/lmm_quality_evaluation.py` | Held-out gerçek gateway çağrılarında model-emitted citation/abstention contract'i, privacy-conscious report ve fail-closed release threshold'ları |
| `tools/lmm_portfolio.py` | Immutable manifest + passed quality report + target-domain evaluation card ile deterministic expert model routing veya bağımsız panel planı; `single-best` gerçek dispatch'i |
| `tools/lmm_release_governor.py` | Frozen portfolio/task baseline-candidate identity'leri üzerinde aggregate-only shadow → bounded canary → promote/rollback karar/state chain'i ve external traffic directive |
| `tools/lmm_release_registry.py` | Verified governor artifact'lerini canonical strict-LF local hash-chain ledger'da single-host expected-state CAS ile serialize eder; privacy-safe authoritative head raporu verir |
| `tools/lmm_tool_orchestrator.py` | Request+registry-bound typed read-only tool planı, prior-result reference, no-shell private execution, distinct verifier ve gateway-ready evidence publication |
| `tools/verify_lmm_runtime.py`, `tools/verify_lmm_portfolio.py`, `tools/verify_lmm_tool_orchestrator.py`, `tools/verify_lmm_release_governor.py`, `tools/verify_lmm_release_registry.py` | Runtime boundary ile domain-routing/privacy/provenance/fail-closed dispatch, verified tool plan ve release-governance/authoritative-head entegrasyon testleri |

## 2. Gerçek serbest doğal dil üretimi

`lmm_gateway.py`ın kendisi yanıt şablonu veya hardcoded cevap içermez. Yanıt,
aşağıdaki backend'lerden gelen gerçek model output'udur:

### 2.1 `openvino-genai`

Intel/OpenVINO yolunda bir operator, lisanslı ve OpenVINO GenAI ile uyumlu yerel
model package sağlar. Gateway `LLMPipeline` üzerinden gerçek generation çağrısı
yapar. Bu adapter bugün **text generation** içindir; model manifest'i image,
audio veya video capability ilan etse bile OpenVINO text adapter'ı bu asset'leri
kabul etmez. Multimodal OpenVINO/VLM için aşağıdaki worker protocol kullanılmalıdır.

```bash
python3 -m venv .venv-lmm
. .venv-lmm/bin/activate
pip install -r requirements-lmm-openvino.txt

python3 tools/lmm_gateway.py \
  --model /secure/model-package/model.manifest.json \
  --model-root /secure/model-package \
  --request request.json \
  --device GPU \
  --output lmm-runs/run-001.json
```

`--device GPU`, `NPU` veya `CPU` bir **istenen provider** bilgisidir. Gateway,
başarılı çalışmayı gerçek model generation çağrısı olarak kaydeder; ancak bağımsız
profiling/evidence olmadan bir donanım hızlanma oranı iddia etmez.

### 2.2 `openai-compatible`

HTTPS chat-completions endpoint'i olan gerçek bir model server'ı kullanılabilir.
API anahtarı yalnız `--api-key-env` ile environment'tan alınır; report'a,
command manifest'ine veya Git'e yazılmaz. OpenAI-compatible standart content
formatında text ve inline boyut sınırına uyan **image** part'leri taşınır.
Audio/video protocol'u provider'lar arasında standart olmadığı için bu backend
onları fail-closed reddeder. Remote provider'ın gerçek ağırlık/revision'ı
cryptographically doğrulanamaz; manifest'teki `model_id`/`revision` operatörün
configuration claim'idir ve provider-side model attestation yerine geçmez.

### 2.3 `worker`

`worker` backend'i, image/audio/video dahil tüm asset'leri destekleyen gerçek
yerel VLM/AV modelini bağlamak için tasarlanmıştır. Gateway shell kullanmaz;
trusted worker command'i bir JSON string-array dosyasından okunur ve stdin'e
tek bir `vegai-lmm-worker-request/v1` objesi gönderilir. Worker stdout'a tam
bir `vegai-lmm-worker-response/v1` döndürmelidir.

Worker request, asset'in raw bytes'ını değil doğrulanmış local path, SHA-256,
MIME, modality ve context-token cost'unu taşır. Varsayılan
`--worker-assets-mode snapshot`, her istenen asset'i hash kontrolüyle private,
geçici bir dosyaya stream-copy eder; worker yalnız bu request snapshot'ını görür.
Toplam snapshot varsayılan olarak 4 GiB ile sınırlıdır ve daha büyük, planlı video
istekleri için `--max-worker-snapshot-bytes` açıkça yükseltilmelidir. Böylece
worker'ın decode ettiği medya bytes'ı manifest hash'ine bağlanır. Büyük resident
deployment'ta salt-okunur/immutable asset store ve worker'ın kendi SHA-256
kontrolü kanıtlanmışsa `--worker-assets-mode direct` copy maliyetini kaldırır;
bu modda path değişmezliği operatörün sorumluluğudur.

### 2.4 Optional kanıt/citation output contract'i

Gateway model yanıtına hiçbir citation veya abstention metni eklemez. Operatorün
system policy'si/fine-tune'u aşağıdaki marker'ları **model output'u içinde**
üretmelidir:

```text
[[vegai-source:document-id]]
[[vegai-abstain]]
```

`--grounding-mode require-citations`, nonempty evidence bundle varken en az bir
verilen document ID'sinin citation'ını zorunlu kılar. `--grounding-mode
evidence-or-abstain`, evidence varsa citation; evidence yoksa tam bir abstention
marker'ı ister. Bilinmeyen citation ile malformed citation/abstention marker'ı
fail-closed olur; report marker gözlemini ve mevcut evidence ID'lerini taşır. Bu bir citation provenance
kontrolüdür; modelin cümlesinin gerçekten doğru olduğu, kaynak metni anladığı
veya iyi uncertainty calibration yaptığı anlamına gelmez.

## 3. Unicode ve gerçek subword tokenization

HDC çekirdeğinin ASCII tokenizer'ı değiştirilmemiştir. LMM düzlemi ayrı bir
Unicode sözleşmesi kullanır:

- text UTF-8, CR/NUL olmayan ve bounded Unicode string'dir;
- model package `tokenizers-json` tokenizer path'i ve SHA-256'sini manifest'e
  bağlayabilir;
- `requirements-lmm.txt` içindeki Rust tabanlı `tokenizers` paketi ile gerçek
  BPE/WordPiece/Unigram `tokenizer.json` çalıştırılır;
- model tokenizer'ı yoksa yalnız `utf8-byte` bootstrap tokenizer kullanılabilir.
  Bu path lossless test/adapter bootstrap içindir; semantic subword model olduğu
  iddia edilmez.

Modelin tokenizer JSON'u ve model artifact'leri immutable manifest'te hash'li
olmalıdır. Model-specific normalizasyon tokenizer JSON tarafından yapılır;
gateway modelden bağımsız “Türkçe düzeltme” veya kelime hardcode etmez.

## 4. Uzun context ve doğal sohbet durumu

### 4.1 Hash-chained session

Her conversation message `id`, `role`, content part'leri ve `pinned` alanını
taşır. `lmm_session.py append`, önceki event SHA-256 head'ini zorunlu kılar ve
kilit + atomic replace ile ledger yayınlar; `compile` aynı lock ile çalışır ve
raporladığı head'in publish boyunca değişmediğini tekrar doğrular. Böylece stale
writer veya kısmi append sessizce yeni history/request üretmez.

```bash
# İlk system message
python3 tools/lmm_session.py append \
  --ledger lmm-sessions/support.jsonl --message system-message.json \
  --expected-head GENESIS

# Son append çıktısındaki head değeriyle kullanıcı turu eklenir.
python3 tools/lmm_session.py append \
  --ledger lmm-sessions/support.jsonl --message user-message.json \
  --expected-head PREVIOUS_HEAD

python3 tools/lmm_session.py compile \
  --ledger lmm-sessions/support.jsonl --expected-head CURRENT_HEAD \
  --request-id support-turn-19 --max-new-tokens 512 --temperature 0.2 --top-p 0.95 \
  --output request.json
```

Ledger ham history'yi korur; model “summary” üretip eski mesajları sessizce
değiştirmez.

### 4.2 Deterministic token-budget packer

`lmm_gateway.py` model manifest'indeki `context_window_tokens`, tokenizer,
message overhead ve output reservation değerlerini kullanır:

1. pinned policy/evidence mesajları ve son user turn zorunlu tutulur;
2. kalan eski konuşma, en yeni tam user-led turn'lerden geriye doğru alınır;
3. bütçeye sığmayan eski turn tamamen elenir; assistant cevabı user turn'ünden
   koparılmaz;
4. included/omitted message id + hash'leri report'a yazılır.

Bu, modelin gerçek context window'una sığmayan history için deterministic ve
denetlenebilir bir sınır sağlar. “Sonsuz context” veya semantik memory iddiası
değildir. Daha yüksek kaliteli history compression için ayrı bir model,
evaluation seti ve loss-aware policy gerekir.

## 5. Dünya bilgisi: latent iddia değil, kaynaklı evidence

`lmm_knowledge.py`, `vegai-lmm-document/v1` JSONL'den SQLite FTS5 Unicode
index üretir. Her document şu alanları zorunlu taşır:

```json
{
  "schema_version": "vegai-lmm-document/v1",
  "id": "doc-001",
  "text": "Unicode kaynak metni",
  "source": "operator source registry",
  "license": "license identifier",
  "retrieved_at": "2026-10-23",
  "metadata": {"language": "tr"}
}
```

İndeks Unicode diacritic-insensitive search ve CJK single-codepoint/bigram
fallback kullanır. Build JSONL'i streaming okur; duplicate document ID kontrolü
Python'da dev bir ID set'i tutmak yerine SQLite primary key üzerinde yapılır.
Bu, Türkçe, Arapça, Japonca/Çince/Korece gibi farklı script kümelerinde lexical
retrieval için bir başlangıçtır; tam morphology veya semantic embedding değildir.

```bash
python3 tools/lmm_knowledge.py build \
  --input governed-documents.jsonl --output lmm-knowledge/domain.sqlite
python3 tools/lmm_knowledge.py query \
  --index lmm-knowledge/domain.sqlite --query "guncel soru" \
  --limit 8 --output evidence.json
```

Evidence bundle, text SHA-256, source/license ve index identity ile birlikte
gateway'e verilir. Gateway bunları açık delimiters ile pinned context'e ekler ve
report'a document id listesini yazar. Kaynak metni prompt injection içerebilir;
delimiter tek başına güvenlik garantisi değildir. Source review, ACL, license,
PII ve task-specific factual evaluation operator sorumluluğundadır.

## 6. Görüntü, ses ve video input plane

Asset ingestion örneği:

```bash
python3 tools/lmm_assets.py ingest \
  --input image.png --store lmm-assets --registry lmm-assets/registry.jsonl \
  --id invoice-image-001 --source 'operator upload' --license 'operator-owned' \
  --retrieved-at '2026-10-23' --context-tokens 576

python3 tools/lmm_assets.py verify \
  --store lmm-assets --registry lmm-assets/registry.jsonl
```

Araç PNG/JPEG/WebP/GIF, WAV/FLAC/Ogg/MP3/M4A ve MP4/MKV/WebM/MOV için extension
ve magic-byte kontrolü yapar, SHA-256 ile content-addressed local store'a kopyalar
ve registry read-modify-write işlemini lock + atomic replace ile serialize eder.
Aşağıdakileri manifest'e bağlar:

- modality ve MIME;
- byte count/SHA-256;
- source/license/retrieval zamanı;
- model-specific context token cost.

Bu katman medya **decoder/encoder/fusion modeli değildir**. Bir worker veya
endpoint manifest'te ilgili capability (`image`, `audio`, `video`) ilan etmezse
request çalışmaz. Bu fail-closed davranış “dosyayı kabul ettik, demek ki model
anladı” türü yanlış multimodal iddiaları engeller.

## 7. Immutable model package

Her `vegai-lmm-model/v1` manifest'i en az şunları bağlar:

- engine (`openvino-genai`, `openai-compatible`, `worker`);
- model id/revision/source/license;
- context window, maximum output, message/reply overhead;
- capability matrix;
- tokenizer type ve `tokenizer.json` hash'i;
- local model artifact path/SHA-256/byte count listesi;
- prompt format (`chatml-v1` veya structured `json-v1`).

OpenVINO local model için artifacts listesi boş olamaz. Manifest'i elle hash
kopyalamak yerine şunu kullanın:

```bash
python3 tools/lmm_model_manifest.py create \
  --model-root /secure/openvino-model --output /secure/manifests/model.json \
  --model-id approved-model --engine openvino-genai \
  --context-window-tokens 8192 --max-output-tokens 1024 \
  --tokenizer-kind tokenizers-json --tokenizer-path tokenizer.json \
  --source 'approved model registry/revision' --license 'approved-license' --revision 'release-id'
python3 tools/lmm_model_manifest.py verify \
  --manifest /secure/manifests/model.json --model-root /secure/openvino-model
```

Gateway generation öncesi ve sonrası model artifact'lerini tekrar hash'ler.
Model/asset paket kökleri ile package içi artifact yollarında symlink kabul
edilmez; release manifest'i oluşturulurken tüm normal dosyalar tekrar hash'lenir
ve değiştirilmiş package doğrulamada fail-closed olur. Büyük model package'inde
bu integrity maliyetini ölçün; düşük latency gereken deployment'ta immutable,
read-only model mount ve resident server ile bu preflight'ı ayrı release adımına
alın, fakat hash/provenance kaydını kaldırmayın.

## 8. Performans ölçümü

Rust `tokenizers`, SQLite FTS5 ve content-addressed asset referansları Python
kontrol plane'ini hafif tutar. Varsayılan worker media snapshot'ı güvenlik için
stream-copy maliyetini gateway süresine dahil eder; immutable store'lu ölçülmüş
production ortamı ancak `--worker-assets-mode direct` ile bunu kaldırmalıdır.
Asıl throughput seçilen neural model, quantization, context length, device
driver ve worker/endpoint'in resident olup olmamasına bağlıdır.

```bash
cat > benchmark-command.json <<'JSON'
["python3", "tools/lmm_gateway.py", "--model", "model.json", "--request", "request.json", "--worker-command-json", "worker-command.json", "--print-json"]
JSON
python3 tools/benchmark_lmm_gateway.py \
  --command-json benchmark-command.json --warmups 1 --repetitions 5 \
  --output lmm-runs/benchmark.json
```

Benchmark yalnız gateway child command'inin süresini ölçer. Compiler, fixture
oluşturma ve Python aggregation süreleri sample dışındadır. Command model'i her
sample'da yeniden yüklüyorsa bu cold-load maliyeti **bilerek** sample içindedir;
resident worker/API için command resident servise bağlanmalıdır. Report, yalnız
backend'in döndürdüğü token count varsa output token/s verir; GPU/NPU hızlanma
iddiası yapmaz.

## 9. Held-out model kalite ve release gate

`tools/lmm_quality_evaluation.py`, gerçek gateway command'ini her held-out case
icin shell kullanmadan ve JSONL'i streaming okuyarak çalıştırır. Her
`vegai-lmm-quality-case/v1` JSONL kaydı,
model request'ini, backend'e verilecek evidence bundle'ı, izinli/zorunlu citation
ID'lerini ve gerekirse abstention beklentisini taşır. Case'nin request/evidence
metni ile backend'in ham yanıtı report'a kopyalanmaz; report yalnız SHA-256 veya
opsiyonel `--case-id-hmac-key` HMAC case kimliği, response hash/byte sayısı,
citation ID'leri ve generation-report hash'i taşır. Gateway/model/tokenizer/
worker configuration identity'leri case'ler arasında aynı kalmalıdır; evaluator
bu stable runtime identity'yi de report'a bağlar.

Komut vector'ü `--request {request}` ve `--evidence {evidence}` biçiminde tam
olarak birer placeholder ile
`--print-json` içermelidir. Sabit model/backend/worker seçenekleri bu
operator-owned JSON array'de kalır:

```bash
cat > heldout-gateway-command.json <<'JSON'
["python3", "tools/lmm_gateway.py", "--model", "/secure/model.json",
 "--request", "{request}", "--evidence", "{evidence}",
 "--worker-command-json", "/secure/worker-command.json",
 "--grounding-mode", "evidence-or-abstain", "--print-json"]
JSON
python3 tools/lmm_quality_evaluation.py \
  --cases governed-heldout.jsonl \
  --gateway-command-json heldout-gateway-command.json \
  --report lmm-runs/heldout-quality.json --fail-on-gate
```

Default gate; case-pass rate, required-citation recall, invalid-citation case
rate ve her case'in abstention beklentisi doğruluğu için sırasıyla `%100`,
`%100`, `%0`, `%100` eşiklerini uygular. `--fail-on-gate` report yayınlandıktan sonra başarısız gate
icin exit `3` verir. Bu metrikler operatorün kaliteli source/case tasarımıyla
anlamlıdır; prose factuality, safety, preference, semantic reasoning veya
LLM'lerden üstünlük yerine geçmez.

## 10. Ölçülmüş domain expert routing ve bağımsız panel

`tools/lmm_portfolio.py`, tek bir modelin her task'ta en iyi olduğu varsayımını
kabul etmez. Operatorün `vegai-lmm-portfolio/v1` dosyasında model manifest,
gateway command, passed `vegai-lmm-quality-report/v1` ve target-domain
evaluation card bağlanır. Routing task; domain, residency, safety, modality,
context, latency, cost ve measured lower-bound threshold'larını tanımlar.
Uygun candidate yoksa fallback cevap yazmak yerine fail-closed durur.

`single-best`, en yüksek deterministic measured ranking ile seçilen gerçek
backend'i private input snapshot üzerinden çağırabilir. `independent-panel`,
farklı `independence_group` değerine sahip birden çok eligible candidate planı
üretir fakat cevapları otomatik birleştirmez veya çoğunluk oylamasını factuality
kanıtı diye göstermez. Bu yaklaşım, operator-owned human/adjudicator/tool
workflow'unda bağımsız critique için güvenli başlangıçtır. Tam şema, örnek ve
sınırlar için [LMM_PORTFOLIO_ROUTING.md](LMM_PORTFOLIO_ROUTING.md)'ye bakın.

### 10.1 Offline route sonrası shadow/canary release governor

`lmm_portfolio.py`nin selected candidate'i, gerçek user traffic'e otomatik
promote edilmez. `tools/lmm_release_governor.py`, aynı immutable task ve
portfolio üzerinden baseline/candidate'i yeniden yükler; model manifest,
gateway command, evaluation card ve passed held-out quality report identity'ni
state'e bağlar. Policy quality, reliability, safety, cost ve latency için ayrı
bounded live metric ister. Consentli shadow evidence pass etmeden bounded
canary directive'i; conservative all-metric non-regression kanıtı olmadan da
full promotion directive'i vermez.

Governor input'u raw session/request/evidence/response değildir: strict LF
aggregate telemetry ve opaque replay IDs'dir. Decision raw ID/path içermez;
external controller için `shadow-mirror-only`, `bounded-canary`,
`candidate-promoted` veya `baseline-only-rollback` directive'i üretir. Traffic
enforcement, telemetry integrity, signed/distributed release head, provider
revision attestation, privacy/tenant isolation ve emergency kill switch bu
Python control plane'in dışında kalır. Tam contract:
[LMM_RELEASE_GOVERNANCE.md](LMM_RELEASE_GOVERNANCE.md).

### 10.2 Local authoritative release registry

`tools/lmm_release_registry.py`, governor'ın deliberate multi-child artifact
özelliğini canlı controller için tek-host authoritative head'e çevirir. `record`
current state SHA-256'yi CAS token olarak ister (`GENESIS` yalnız initial
event'tir); policy/governor/registry implementation identity'sini, direct
predecessor'u, immutable consumed observation-batch dependency'sini ve exact
traffic directive'i yeniden doğrular. Accepted events strict-LF JSONL
hash-chain'e exclusive local lock ve atomic publish ile yazılır. `inspect`
policy argument'iyle historical ledger'ı privacy-safe digest/phase/directive
olarak tekrar doğrular.

Bu layer model veya telemetry güvenilirliği vermez ve route config'i uygulamaz.
Aynı zamanda signing, remote append-only retention, distributed lock/consensus
ve HA failover çözümü değildir: multi-host controller signed remote registry ve
transactional traffic-control boundary kullanmalıdır. Local registry sayesinde
rejected competing/stale child'ın bu host ledger'ında canonical head olması
engellenir; gerçek provider/traffic-plane apply sonucu ayrıca denetlenmelidir.

## 11. Typed, verified tool-use plane

`tools/lmm_tool_orchestrator.py`, LMM'in veya operatörün output olarak verdiği
observable action planını actual answer'dan ayrı işler. Plan exact canonical
request digest'i ve registry file SHA-256'sına bağlıdır. Her read-only tool;
hash-pinned input/output schema, executor implementation, no-shell argv ve
ayrı implementation hash'i taşıyan verifier ile declared olur. Prior pass eden
result'tan RFC 6901 JSON Pointer ile sonraki typed invocation input'una dataflow
verilebilir; forward/cyclic reference kabul edilmez.

Başarılı result, raw plan/request/result taşımayan execution report ve normal
`vegai-lmm-evidence/v1` artifact'ı üretir. Evidence, lmm gateway'e verilebilir;
model response'u yine backend'in kendi output'udur ve citation marker'ını model
üretir. Orchestrator final prose, hidden chain-of-thought veya tool sonucu
uydurmaz. Salt-okunur sınır, timeout ve credential-minimized environment vardır;
ancak container/OS sandbox, network policy, side-effect confirmation, real
verifier independence ve held-out tool-use evaluation operator/deployment
katmanında zorunludur. Tam contract:
[VERIFIED_TOOL_ORCHESTRATION.md](VERIFIED_TOOL_ORCHESTRATION.md).

## 12. Doğrulama

```bash
python3 tools/verify_lmm_runtime.py
python3 tools/verify_lmm_portfolio.py
python3 tools/verify_lmm_tool_orchestrator.py
python3 tools/verify_lmm_release_governor.py
python3 tools/verify_lmm_release_registry.py
```

Bu beş test birlikte şunları denetler:

- Türkçe, Arapça ve Japonca Unicode retrieval;
- `tokenizers` kuruluysa gerçek `tokenizer.json` BPE yolu;
- UTF-8 bootstrap round-trip;
- hash-chain session append/compile;
- token-budget context eviction ve full-turn korunması;
- content-addressed image/audio/video registry ve hash-checked worker snapshot;
- model-package manifest create/rehash ve değiştirilmiş artifact fail-closed rejection;
- evidence ve asset SHA provenance'inin worker request/report'a taşınması;
- local OpenAI-compatible endpoint'e gerçek text/image request payload'u;
- model-emitted citation/abstention marker gateway gate'i ve privacy-conscious
  held-out quality/release evaluator (başarılı ve bilinçli başarısız gate);
- measured domain-expert portfolio ranking, bağımsız panel, selected gateway
  dispatch, raw-text privacy ve başka manifest'e bağlı command fail-closed rejection;
- request/registry-bound typed tool plan, predecessor reference, separate verifier,
  no-replace evidence/report, raw argument/result privacy ve negative tool gates;
- immutable portfolio identity, shadow pass → bounded canary transition,
  conservative promotion/rollback, aggregate-only privacy, replay/forged
  telemetry/state semantic/tamper ve required quality/reliability/safety/cost/
  latency policy gates;
- authoritative local `GENESIS`/state-head CAS, competing-child rejection,
  consumed batch/directive binding, strict-LF hash-chain, stale-lock, ledger
  corruption, policy mismatch ve path/raw-ID privacy gates;
- modality capability yoksa fail-closed rejection;
- gerçek gateway child process benchmark sınırı.

`requirements-lmm.txt` CI'da BPE path'ini doğrular. `requirements-lmm-openvino.txt`
model ağırlığı içermez; yalnız operatorün kendi OpenVINO model package'ini
çalıştırması için optional provider bağımlılığıdır.

## 13. Hâlâ yapılmamış ve iddia edilmeyenler

Bu entegrasyon aşağıdakileri otomatik olarak sağlamaz:

- pretraining ile sıfırdan dev Transformer eğitimi;
- autodiff, AdamW/Lion, distributed optimizer veya BF16/FP8 kernels;
- model/data parallelism, collective recovery veya sharded checkpoint;
- universal factual doğruluk, güvenlik alignment veya copyright/PII temizliği;
- audio/video semantics veya visual reasoning; bunlar gerçek declared VLM/AV
  worker/model gerektirir;
- LMM benchmark, human preference, red-team veya multimodal task score'u;
- kernel/container sandbox, network egress policy, real verifier independence,
  mutating-tool authorization/idempotency veya tool-use benchmark'ı;
- signed/distributed deployment registry/CAS, live traffic enforcement,
  trustworthy online telemetry, provider weight attestation, tenant isolation
  veya automatic production safety approval. (Local registry yalnız single-host
  serialization sağlar.)

Bu yüzden bu katman “LMM eklendi” demek için yeterli bir operational boundary
sağlar; “Vegai artık evrensel LMM'den daha iyi” sonucu çıkarmaz. Gerçek kalite
iddiası, seçilen model, lisanslı data, hedef donanım ve held-out evaluation ile
ayrıca kanıtlanmalıdır.
