# Vegai kalite güvencesi, LMM sınırı ve iyileştirme planı

Bu belge, Vegai'nin tüm çalışma yüzeyinin (Fortran çekirdeği, HDC pretraining,
checkpoint/continual transaction, veri katmanı, grounded retrieval, production
araçları, donanım sözleşmesi ve CI) kalite denetimini tanımlar. Amaç ölçülebilir
güvenilirliği yükseltmektir; ölçülmemiş bir "LMM'den daha iyi" iddiası üretmek
değildir.

## 1. Dürüst kalite hedefi

Vegai, Transformer tabanlı generative LMM/LLM değildir. Packed 16.384-bit HDC
semantic prototype'ları, exact symbolic operations ve operator-owned knowledge
base retrieval kullanır. Bu yüzden aşağıdaki iki ifade birbirinden ayrılmalıdır:

- **Desteklenmeyen evrensel iddia:** serbest doğal dil üretimi, dünya bilgisi,
  multimodal muhakeme, next-token kalitesi veya tüm benchmark'larda LMM'leri
  geçmek.
- **Ölçülebilir Vegai hedefi:** belirli, versioned bir domain corpus/checkpoint/
  knowledge base üzerinde deterministic çalışma; kaynağa bağlı cevap; exact
  provenance; leakage/deduplication kontrolleri; held-out grounded retrieval
  accuracy ve güvenli abstention.

Dar bir retrieval domain'inde, uygun quality gate'lerle Vegai; source disclosure,
aynı input'ta checkpoint byte determinism ve grounding başarısız olduğunda
cevap uydurmama bakımından **uygun şekilde yapılandırılmamış** generative bir
sisteme göre daha denetlenebilir olabilir. Bu, daha iyi genel dil modeli olduğu
anlamına gelmez.

## 2. Uçtan uca mimari denetimi

| Alan | Mevcut doğrulanabilir güç | Kalite riski / sınır | Uygulanan veya gerekli kontrol |
|---|---|---|---|
| Bit cebiri | Fortran 2023 fixed-size HDC, SplitMix64 oracle, XOR/unbinding, deterministic ties | Model kapasitesi HDC cardinality ve vocabulary ile sınırlı | Native property tests, bağımsız Python oracle |
| Symbolic memory / Resonator | Exact cleanup ve multi-factor certificate | Heuristic resonator aşaması tek başına certificate değildir | Exact verification, ambiguity telemetry |
| Pretraining | İki-pass streaming vocabulary ve deterministic source-ordered Hebbian update | Transformer loss/perplexity yok; semantic quality corpus'a bağımlı | Strict ingest, checkpoint/resume, mode comparison diagnostics |
| Data quality | JSONL/JSON/TXT/Parquet/Arrow schema boundary, exact dedupe, split audit | Exact dedupe semantic duplicate, PII/license veya contamination çözmez | `training_dataset.py`, provenance ve upstream governed pipeline |
| Production transaction | Immutable shard, SHA-256 ledger, exact leakage audit, exactly-once state | Source/domain quality kendi başına garanti edilmez | `production_dataset.py`, `production_pretrain.py`, health audit |
| Grounded chat | Her response operator TSV'den gelir; exact Hamming scan, margin/similarity abstention | Held-out task accuracy ve threshold calibration yoksa güvenilirlik bilinmez | **Yeni:** versioned held-out evaluator + release gates |
| Optional LMM integration | Gerçek external model backend'i için Unicode/subword manifest, long-context packer, knowledge evidence, multimodal asset boundary, held-out citation/abstention release gate ve measured domain-expert routing/panel | Model weights/quality/provider güvenliği Vegai tarafından sağlanmaz; scorecard broad quality kanıtı değildir, multimodal capability model-specific'tir | **Yeni:** `lmm_gateway.py`, `lmm_quality_evaluation.py`, `lmm_portfolio.py`, session/knowledge/assets contracts, artifact hash ve fail-closed capability/quality/routing gate |
| Verified tool use | Real LMM/operator action planını request+registry SHA ile bağlar; typed predecessor dataflow, no-shell executor, separate verifier ve gateway evidence üretir | Declared independence gerçek audit/sandbox/factual truth değildir; side-effect tool bilinçli olarak yoktur | **Yeni:** `lmm_tool_orchestrator.py`, strict schema/reference/verifier/privacy/no-replace negative gates |
| Algorithm/optimizer selection | Real external algorithm/optimizer config, training contract ve artifact identity'sini repeated held-out bounded metric ile karşılaştırır | Hoeffding sınırı independence/bound/protocol varsayımlarına bağlıdır; neural optimizer execute etmez veya universal-best kanıtlamaz | **Yeni:** `evidential_optimizer.py`, all-objective baseline improvement, robust Pareto, no-replace decision ve negative gate verifier |
| Hardware / performance | Portable/native CPU profiles, OpenMP byte equality, resource plan, optional OpenVINO truth contract | GPU/NPU inventory compute kanıtı değildir; universal throughput yok | Host-specific benchmark ve provider execution evidence |
| Security / governance | Fail-closed ASCII contracts, hashes, atomic publication | Prompt/response factuality, PII, licensing, adversarial corpus ve human review dışarıdadır | Versioned source registry, review, retention, red-team policy |

Bu tablo "her alan tamam" anlamına gelmez. Her satırda modelin gerçekten
verebildiği garanti ile operatörün sağlaması gereken kalite ayrı tutulur.

## 3. Yeni held-out grounded retrieval evaluation katmanı

Yeni araç: `tools/dialogue_evaluation.py`

Araç, gerçek derlenmiş `vegai_chat ask` komutunu her test case için **stateless**
(`--history-turns 0`, `OMP_NUM_THREADS=1`) çalıştırır. Böylece test sıra etkisi
ve history contamination olmadan şu iki davranışı ölçer:

1. beklenen knowledge-base prompt'a doğru grounding;
2. knowledge-base cevabı verilmemesi gereken case'te doğru abstention.

Araç response prose'unu yeniden üretmez, scorelamaz veya kaynak koda intent
kuralı eklemez. Yalnız `vegai_chat`in mevcut source-disclosing output
sözleşmesinden `grounded`, `matched prompt`, similarity ve margin telemetrilerini
strict olarak parse eder.

### 3.1 Case şeması

Şema: `vegai-dialogue-evaluation-case/v1`

Her strict LF JSONL kaydı aşağıdaki alanların hepsini taşır:

```json
{
  "schema_version": "vegai-dialogue-evaluation-case/v1",
  "id": "education-exact",
  "prompt": "okul ders",
  "expected_prompt": "okul ders"
}
```

Beklenen abstention açıkça `null` ile yazılır:

```json
{
  "schema_version": "vegai-dialogue-evaluation-case/v1",
  "id": "single-token-abstention",
  "prompt": "tarim",
  "expected_prompt": null
}
```

Kurallar:

- `id`, `prompt`, `expected_prompt` (veya açık `null`) zorunludur;
- unknown field, duplicate id, blank row, CRLF, non-printable/non-ASCII byte,
  1.024 byte'tan uzun prompt ve 16 KiB'tan uzun physical JSONL row reddedilir;
- evaluator prompt/expected prompt'u report'a kopyalamaz; varsayılan SHA-256
  veya opt-in keyed HMAC-SHA-256 digest'i ile bağlar;
- test query/annotation seti threshold calibration setinden ayrı, versioned ve
  erişim kontrollü tutulmalıdır; `expected_prompt` bilerek publish edilmiş KB
  key'ine referans verir. Test case'leri KB/threshold tuning veya training
  corpusuna geri beslenmemelidir.

Örnek sentetik fixture:
`examples/dialogue/mini_turkish_hdc_evaluation.jsonl`. Bu fixture'daki exact
prompt case'leri runtime contract smoke testidir; production natural-language
retrieval kalitesi için ayrı, domain-representative held-out query/annotation
seti gerekir.

### 3.2 Ölçümler

Report şeması: `vegai-dialogue-evaluation-report/v1`

| Metrik | Tanım |
|---|---|
| `grounded_prompt_accuracy` | Expected-grounded case'lerde tam beklenen prompt'a eşleşen oran |
| `grounded_precision` | Grounded sonucu verilen case'lerde doğru prompt oranı |
| `grounded_recall` | Beklenen grounded case'lerin doğru prompt'a ulaşılan oranı |
| `abstain_precision` | Abstain edilen case'lerin gerçekten abstain beklenen oranı |
| `abstain_recall` | Abstain beklenen case'lerin abstain edildiği oran |
| `grounded_coverage` | Tüm testte grounded sonuç oranı |

`wrong_grounded_prompt`, `false_grounded`, `false_abstain` ve
`true_abstain` sayıları ayrıca yayınlanır. Böylece yalnız bir aggregate score
ile yanlış güven üretilmez. Release gate'ler hem `--min-*` oranları hem de
özellikle yüksek riskli domain'lerde `--max-wrong-grounded-prompt-cases`,
`--max-false-grounded-cases`, `--max-false-abstain-cases` mutlak üst sınırlarını
kullanabilir.

### 3.3 Provenance ve privacy

Rapor şunların path/SHA-256/byte count kimliğini bağlar:

- evaluator script'i;
- `vegai_chat` binary;
- checkpoint;
- knowledge-base TSV;
- evaluation JSONL;
- optional case-HMAC key (path'i redact edilerek).

Evaluation sırasında bu dependency'ler tekrar hash'lenir; herhangi biri
değişirse run fail-closed olur. `--output`, bu immutable input'lardan biriyle
aynı path'e verilemez. Per-case raporda raw held-out prompt, expected prompt veya operator
response yoktur; yalnız case id, prompt digest, expected/matched prompt digest
ve outcome bulunur. Varsayılan SHA-256, public sentetik fixture için
reproducible'dır fakat kısa/low-entropy prompt'lara dictionary attack engeli
sağlamaz. Hassas held-out setlerde, repository dışında saklanan nonempty
`--case-hmac-key-file` ile keyed HMAC-SHA-256 kullanılmalıdır; key report'a
kopyalanmaz, yalnız key dosyasının SHA-256 kimliği bağlanır ve key path'i
redact edilir. Case id'leri de hassassa report erişimi ayrıca
kısıtlanmalıdır.

## 4. Release gate kullanımı

Önce **calibration/validation** setinde similarity ve margin threshold'larını
seçin. Ardından threshold'ları değiştirmeden frozen held-out test setinde gate
çalıştırın. Aynı test setinde threshold seçmek test leakage'dir.

```bash
python3 tools/dialogue_evaluation.py \
  --chat-bin ./build/vegai_chat \
  --checkpoint releases/domain-a/model.vgai \
  --knowledge releases/domain-a/knowledge.tsv \
  --evaluation governed-evaluation/domain-a-test.jsonl \
  --case-hmac-key-file /secure/keys/domain-a-dialogue-report.key \
  --output releases/domain-a/dialogue-quality.json \
  --max-entries 4096 \
  --min-similarity 0.18 --min-margin 0.04 \
  --min-grounded-prompt-accuracy 0.95 \
  --min-grounded-precision 0.97 \
  --min-grounded-recall 0.90 \
  --min-abstain-precision 0.95 \
  --min-abstain-recall 0.90 \
  --max-false-grounded-cases 0
```

Eşikler örnektir; repository bunları universal hedef veya doğal dil kalitesi
iddiası olarak sunmaz. Domain owner; risk, false-grounding maliyeti,
abstention UX'i, test set büyüklüğü ve insan denetimine göre threshold
belirlemelidir.

- Exit `0`: verilen gate'ler geçti.
- Exit `3`: evaluation teknik olarak tamamlandı fakat en az bir kalite gate'i
  geçmedi; JSON report yine yazılır.
- Exit `2`: schema, immutable input, `vegai_chat` contract'i veya process
  çalıştırmasında hata oluştu.

Bu ayrım, "test çalıştı fakat kalite yetersiz" ile "test güvenilir şekilde
çalışmadı" durumunu operasyonel olarak ayırır.

## 5. Kaliteyi gerçekten artıran çalışma sırası

1. **Domain tanımı:** Allowed intent/knowledge scope, source authority, update
   owner, risk sınıfı ve abstention policy belirlenir.
2. **Governed data:** Lisans/rıza/PII/secret/benchmark contamination kontrolleri
   upstream yapılır. `training_dataset.py prepare` exact dedupe ve split audit
   sağlar; semantic near-dedupe yerine geçmez.
3. **Deterministic training:** Strict HDC export veya immutable production
   package kullanılır; dataset/binary/policy hashes kaydedilir.
4. **KB curation:** TSV prompt'ları active vocabulary'e tam bağlı, source
   reviewed ve versioned olur. Response factuality ayrıca domain owner
   sorumluluğundadır.
5. **Threshold calibration:** Ayrı validation setinde false-grounding ile
   abstention trade-off'u ölçülür. Eşik seçim log'u saklanır.
6. **Frozen held-out gate:** Bu araçla release report ve minimum metrikler
   üretilir. Gate geçmezse checkpoint/KB/threshold publish edilmez.
7. **Post-release monitoring:** OOV oranı, abstention oranı, source revision,
   operator feedback ve policy violation'lar privacy-preserving telemetri ile
   izlenir. Yeni data eski held-out testte sızıntı üretmeyecek şekilde versioned
   edilir.

## 6. LMM integration sınırı ve hâlâ kanıtlanmayan alanlar

Yeni optional LMM plane; Unicode/subword tokenizer JSON'u, gerçek model backend
çağrısı, hash-chained long-context, source evidence, multimodal asset referansı,
held-out citation/abstention release gate'i ve measured domain-expert routing
sağlar. Ancak bunlar aşağıdakileri **otomatik** eklemez:

- Vegai içinde Transformer attention, autodiff, optimizer veya gradient training;
- model weight'leri, model family kalitesi veya neural token likelihood kanıtı;
- image/audio/video encoder ve multimodal fusion; bunlar declared gerçek worker/
  VLM backend'inin görevidir;
- distributed model/data/pipeline parallelism;
- factual truth oracle, safety alignment veya red-team otomasyonu;
- LMM benchmark/perplexity/human preference karşılaştırması;
- evaluation card sayılarının geçerli target-domain üstünlük veya panel
  adjudication kanıtı olduğu iddiası.

Bu yetenekler için [LMM_SCOPE_AND_GAP_ANALYSIS.md](LMM_SCOPE_AND_GAP_ANALYSIS.md),
[LMM_RUNTIME.md](LMM_RUNTIME.md) ve [LMM_PORTFOLIO_ROUTING.md](LMM_PORTFOLIO_ROUTING.md)
belgelerindeki model/data/evaluation
mimarisi gerekir. HDC retrieval kalite gate'i yalnız açıkça tanımlanmış operator
knowledge-base domain'inde ölçülebilir güvenilirlik sağlar; LMM backend report'u
ise yalnız gerçek backend çağrısı/provenance kanıtıdır.

## 7. Kod entegrasyon haritası

| Dosya | Rol |
|---|---|
| `tools/dialogue_evaluation.py` | Strict JSONL reader, immutable input binding, compiled-chat executor, metric calculator, privacy-preserving report ve gate exit policy |
| `tools/verify_dialogue_evaluation.py` | Gerçek Fortran binaries ile positive/abstention, strict schema, raw-prompt privacy ve intentional gate failure integration testi |
| `examples/dialogue/mini_turkish_hdc_evaluation.jsonl` | 8 exact grounded + 1 expected abstention sentetik fixture |
| `CMakeLists.txt` | `vegai_dialogue_evaluation` CTest kaydı |
| `docs/DIALOGUE_RETRIEVAL.md` | Retrieval kullanımına held-out calibration/release gate eklenir |
| `docs/VALIDATION.md` | CI doğrulama matrisi ve test kapsamı güncellenir |
| `tools/lmm_contract.py`, `tools/lmm_model_manifest.py`, `tools/lmm_gateway.py` | Gerçek model backend'i için Unicode/subword, immutable artifact package, context, response provenance ve fail-closed gateway |
| `tools/lmm_session.py`, `tools/lmm_knowledge.py`, `tools/lmm_assets.py` | Hash-chain dialogue, source-grounded world-knowledge evidence ve image/audio/video input plane |
| `tools/lmm_quality_evaluation.py`, `tools/lmm_gateway_template.py` | Gerçek gateway command'i üzerinde privacy-conscious held-out citation/abstention evaluation, explicit release thresholds ve shared no-shell template validation |
| `tools/lmm_portfolio.py` | Immutable model/gateway/quality-card binding, measured target-domain expert selection, independent panel planı ve generate-time provenance recheck |
| `tools/lmm_tool_orchestrator.py` | Request+registry-bound typed read-only tool planı, prior-result dataflow, no-shell executor, distinct verifier ve gateway-ready evidence publication |
| `tools/verify_lmm_tool_orchestrator.py` | Synthetic independent worker ile typed multi-step plan, provenance/privacy, collision ve negative tool gates testi |
| `tools/evidential_optimizer.py` | External algorithm/optimizer identity'leri için held-out bounded observation, Hoeffding confidence, all-objective baseline gate, robust Pareto ve atomic no-replace selection |
| `tools/verify_evidential_optimizer.py` | Synthetic protocol ile provenance, confidence/selection, insufficient/non-dominating gate, privacy ve no-replace contract testi |
| `tools/verify_lmm_runtime.py`, `tools/verify_lmm_portfolio.py` | Model-weightsiz LMM runtime/quality/routing contract testleri |

Bu layer, core HDC checkpoint formatını, deterministic training semantics'ini
veya response üretim modelini değiştirmez. Böylece kalite işlemi model
semantiğinden ayrık, test edilebilir ve geri alınabilir kalır.
