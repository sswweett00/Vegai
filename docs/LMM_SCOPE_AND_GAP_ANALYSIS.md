# LMM/LLM kapsam analizi ve Vegai üretim sınırı

Bu belge, repodaki kaynak, CLI, checkpoint, test ve operational tool'ların
incelenmesinden sonra mevcut yetenek ile gerçek bir büyük dil/multimodal model
(LMM/LLM) training system'i arasındaki sınırı açıklar. Amaç capability
inflation'ı önlemektir: orchestration eklemek, HDC modelini Transformer veya
multimodal neural runtime yapmaz.

## İncelenen mevcut sistem

| Alan | Vegai'de doğrulanmış mevcut durum |
|---|---|
| Core representation | Fortran 2023 packed 16.384-bit hypervector/VSA (`hyper_types`, `symbolic_memory`) |
| Tokenless byte-event field | Raw octet'i BPE/subword vocabulary olmadan bounded order/distribution HDC field'e indirger; exact Walsh/inverse/Parseval integer contract'i vardır, semantic/lossless/physical-world claim'i yoktur (`tokenless_field`) |
| Learning | iki-pass ASCII corpus, frequency vocabulary, deterministic HDC context bundle ve source-ordered Hebbian prototype update (`hdc_pretraining`) |
| Factor reasoning | bounded exact certificate ile Resonator Network multi-factor factorization |
| Determinism | fixed seeds, bounded allocation/error contracts, serial/OpenMP checkpoint byte equality, checkpoint/resume |
| Continual data | strict ASCII/LF SHA-256 immutable shards, private snapshot/candidate generation/atomic state journal ile exactly-once commit |
| CPU operation | portable/native Fortran build, OpenMP immutable-context preparation, conservative RAM/checkpoint preflight |
| Optional Intel boundary | OpenVINO yalnız supported positional/hybrid/hierarchical immutable packed-HDC context preparation için; explicit CPU policy native fused kernel'dir. Center-dependent coherent cross-fold mode CPU/OpenMP-only'dir; GPU/NPU execution ayrıca doğrulanmalıdır |
| Yeni production control plane | disk-backed exact raw-record dedupe, content-addressed shards, deterministic held-out split, exact leakage gate, max-vocabulary capacity preflight, binary/data/policy provenance |
| Grounded diyalog retrieval | checkpoint semantic HV + operator TSV prompt/response index, strict UTF-8 Türkçe→ASCII checkpoint projection, exact Hamming legacy veya opt-in lexical/HDC hybrid reranking, evidence/margin abstention ve bounded turn history; cevap TSV'den gelir |
| Grounded retrieval quality gate | versioned UTF-8 held-out JSONL, stateless real-binary exact prompt/abstention metrikleri, semantic/hybrid policy + immutable input digest binding, privacy-conscious report ve explicit release threshold'ları |
| Optional LMM integration plane | Unicode/subword model-tokenizer manifest, hash-chained long dialogue, trust-role-isolated FTS evidence, content-addressed image/audio/video asset registry, strict structured-output contract, model-emitted citation/abstention release gate, measured domain-expert router/independent-panel planı, typed independently verified read-only tool planı ve gerçek OpenVINO/API/worker backend gateway'i; model ağırlığı depoda değildir |
| LMM release control | Frozen portfolio/task/model/gateway/quality identity'leriyle aggregate-only shadow → canary → promote/rollback governor; verified artifact'ler için strict-LF canonical local hash-chain, single-host expected-state CAS ve privacy-safe head (`lmm_release_governor.py`, `lmm_release_registry.py`) |
| Algorithm/optimizer evidence governor | Immutable algorithm/optimizer/training/artifact identity'lerini repeated held-out bounded observation ile Bonferroni-Hoeffding/Pareto gate'ten geçirir; actual optimizer veya neural training runtime değildir (`evidential_optimizer.py`) |

Bu sistem, bounded semantic prototype/retrieval ve symbolic factor reasoning için
üretim kalitesinde operation sözleşmeleri taşıyabilir. Model kapasitesi ve
öğrenme semantiği HDC slot/vocabulary/context yapısına bağlıdır; neural parameter
sayısı, next-token likelihood veya generative model kapasitesi değildir.

`tokenless_field` bu sınırı değiştirmez. Learned BPE/subword tokenizer'a
alternatif olarak ham octet event'lerini doğrudan işleyebilir ve finite
Walsh-Hadamard identity'lerini exact olarak kanıtlayabilir. Ancak sabit 16.384
bitlik bir field arbitrary byte stream için lossless değildir; byte-level
aritmetik, doğal dil veya multimodal semantic understanding, gerçek dünya
factuality ya da fiziksel ölçüm doğruluğu değildir. Bunun için source identity,
calibration/uncertainty ve target-domain evaluation ayrı gerekir. Ayrıntılı
sınır [TOKENLESS_SIGNAL_FIELD.md](TOKENLESS_SIGNAL_FIELD.md)'dedir.

`evidential_optimizer.py` de native AdamW/Lion/Transformer optimizer eklemez.
Gerçek external algorithm/optimizer candidate'larının config/artifact identity'si
ve aynı held-out benchmark üzerindeki repeated bounded ölçümleri için güçlü bir
selection gate sağlar. Bu gate geçilirse yalnız declared domain/metric altında
baseline improvement evidence'i vardır; optimizer'ın evrensel en gelişmiş
olduğu, bağımsızlık varsayımının gerçek olduğu veya gerçek dünya kalite
üstünlüğü kanıtlanmış olmaz. Ayrıntı:
[EVIDENTIAL_OPTIMIZER.md](EVIDENTIAL_OPTIMIZER.md).

`vegai_chat` bu sınırı degistirmez: user-visible metin, operatorun versioned TSV
bilgi bankasindan retrieval ile gelir; model yeni paragraf/token sample etmez.
Similarity/margin gate'i gecilmezse abstain eder. Bu, hardcoded reply yerine
data-driven grounded UX icin yararlidir; LLM sohbet zekasi, factual guarantee
veya safety policy yerine gecmez.

Yeni held-out evaluator bu dar retrieval contract'inde bilinmeyen kalite
durumunu azaltir: expected prompt veya expected abstention'i gerçek compiled
chat binary'si ile olcer ve release threshold'unu fail-closed uygular. Bu
HDC evaluator yalnız retrieval contract'ini ölçer. Ayrı LMM citation/abstention
gate'i eklense de; perplexity, multimodal task, contamination, factuality veya
safety/red-team evaluation yapmaz.

`tools/lmm_gateway.py` ile ayrı bir LMM integration plane eklendi. Bu plane,
gerçek bir operator-supplied OpenVINO GenAI modelini, HTTPS OpenAI-compatible
endpoint'i veya local multimodal worker'ı bağlayabilir; Unicode/subword tokenizer
ve immutable model artifact manifest'iyle context pack eder. Retrieved evidence,
source-free trusted safety mesajından sonra pinned user-data olarak taşınır;
optional strict JSON response contract'i tool/UI tüketicisi için shape gate
sağlar. Image/audio/video ancak gerçek backend capability'si ilan edildiğinde
worker'a geçer.
`lmm_quality_evaluation.py`, gerçek configured gateway üzerinde held-out
citation/abstention contract'ini release threshold ile ölçer; response'u
hardcode etmez veya factuality iddia etmez. `lmm_portfolio.py`, yalnız aynı
model manifest'e bağlı passed quality report ve target-domain measurement card
ile expert route/panel seçimi yapar; scorecard'ın broad quality iddiasını kendi
başına kanıtlamaz. Bu bir LMM runtime'ın tümünü Fortran HDC çekirdeğine
dönüştürmez ve ağırlık/model kalitesi iddiası taşımaz. Ayrıntılı contract için
[LMM_RUNTIME.md](LMM_RUNTIME.md) ve [LMM_PORTFOLIO_ROUTING.md](LMM_PORTFOLIO_ROUTING.md)'ye bakın.

## Gerçek LMM/LLM için eksik zorunlu katmanlar

Aşağıdakiler "flag" veya wrapper ile eklenebilecek küçük eksikler değildir;
yeni data plane, model runtime ve test yüzeyi gerektirir.

| Zorunlu LMM katmanı | Vegai'deki durum | Neden mevcut HDC katmanıyla ikame edilemez |
|---|---|---|
| Unicode/subword tokenizer | Optional `tokenizers-json` model package ile var; HDC `text_tokenizer` hâlâ ASCII-byte'dır | BPE/Unigram tokenizer gerçek model package'ine aittir; geniş tokenizer quality/normalization test suite'i model owner tarafından sağlanmalıdır |
| Tensor/matrix runtime | Native Vegai'de yok; harici actual backend bağlanabilir | Transformer linear layers, embeddings, attention ve activations packed-HV XOR/majority işlemleri değildir |
| Autodiff + optimizer | Native Vegai'de yok; EPO yalnız external candidate evidence selection yapar | Backpropagation, gradient accumulation, AdamW/Lion, clipping, loss scaling ve optimizer state gerekir |
| Transformer/sequence objective | Harici model backend'de olabilir; repo model weight/architecture sağlamaz | Causal/encoder-decoder attention, positional encoding, logits, cross entropy ve sampling/generation gerçek model package'inde kanıtlanmalıdır |
| Multimodal encoders/fusion | Asset registry + worker protocol var; native encoder/fusion yok | image/audio/video decoding, vision/audio encoder, alignment/fusion, modality batching ve safety policy gerçek VLM/AV worker tarafından sağlanmalıdır |
| Distributed training | Yok | data/model/pipeline parallelism, collectives, gradient synchronization, elastic fault recovery ve sharded optimizer/checkpoint gerekir |
| Neural mixed precision | Native Vegai'de yok | BF16/FP16/FP8 numerical policy, master weights, loss scaling ve hardware kernels gerekir |
| LMM evaluation gate | Held-out gerçek gateway citation/abstention contract + release threshold var; broad model-quality gate yok | held-out perplexity/task suites, contamination policy, modality quality, safety/red-team ve calibration gerekir |
| Release authority / traffic enforcement | Aggregate-only governor + local canonical hash-chain, single-host expected-state CAS ve privacy-safe head var | signed remote append-only registry, distributed lease/consensus, provider attestation, controller-to-route-config transaction ve live traffic enforcement gerekir |
| Tool-use/reasoning control plane | Typed read-only plan, request+registry binding, no-shell executor, prior-result reference, declared separate verifier ve gateway evidence var | OS sandbox/network policy, mutation authorization, real independent audit, tool availability/retry, semantic plan/result correctness ve held-out tool-use score gerekir |
| GPU/NPU neural training backend | Optional OpenVINO GenAI text inference adapter var; neural training/joint dispatch yok | adapter inventory/OpenVINO context dispatch, neural training kernel veya CPU+GPU+NPU joint compute kanıtı değildir |
| Dataset governance at LMM scale | Kısmi; document/asset source-license contract eklendi | semantic near-duplicate, licensing/PII/consent, benchmark contamination ve multimodal rights pipeline ayrıca gerekir |

## Native causal Türkçe model için architecture decision

Mevcut çalışma için karar nettir: HDC retrieval katmanı Türkçe UTF-8,
morfoloji-duyarlı lexical kanıt ve versioned/licensed dialogue package ile
iyileştirilebilir; **HDC checkpoint'e decoder etiketi eklenerek causal model
olduğu iddia edilmeyecektir.**

Gerçek native Fortran causal model kararı ancak ayrı bir design/implementation
ile alınabilir. En az şunlar artifact/test olarak mevcut olmalıdır: Unicode
subword tokenizer ABI ve frozen vocabulary; causal mask/position contract'i;
embedding-attention-MLP tensor kernels; forward cross-entropy, backward ve
optimizer-state ABI; deterministic checkpoint/resume; bounded decoder
sampling/stop policy; train/validation/test source-disjoint data registry;
perplexity yanında Turkish dialogue relevance/factual grounding/abstention ve
human preference evaluation. Bu maddeler uygulanana kadar `vegai_chat`in
literal TSV response'u retrieval olarak etiketlenir.

Native data-plane başlangıcı
[TURKISH_DIALOGUE_DATA_GOVERNANCE.md](TURKISH_DIALOGUE_DATA_GOVERNANCE.md)'nde
belgelenmiştir: pinned CC-BY-4.0 Turkish Daily Dialogues 5K kaynağı, immutable
local package builder, strict split isolation ve abstention-only source-disjoint
set sağlar. Kaynağın sentetik/küçük olması özellikle kaydedilir; causal training
veya insan-benzeri konuşma iddiası değildir.

## Bu değişiklikte bilinçli olarak eklenenler

`production_dataset.py` ve `production_pretrain.py`, yukarıdaki neural runtime
boşluklarını gizlemeye çalışmaz. HDC yolunda gerçekten uygulanabilir, test
edilebilir işlemsel eksikleri kapatır:

1. local strict corpus için snapshot-temelli atomic dataset publication;
2. RAM'e sığmayan corpus'lar için SQLite-backed exact record dedupe;
3. raw-record digest'ten stable train/validation assignment ve exact leakage
   rejection;
4. shard payload/ledger/data manifest/binary/policy SHA identity chain;
5. `min_frequency=1` continual policy ile exact train unique vocabulary count
   ve `--max-vocab` fail-closed preflight;
6. controller'a immutable manifest digest pin'i; child training sırasında
   ledger replacement'ın kabul edilmemesi;
7. existing candidate-checkpoint/atomic-state exactly-once semantics'in gerçek
   compiled CLI ile CTest integration doğrulaması;
8. operator TSV retrieval için ayrı frozen held-out exact-grounding/abstention
   evaluation, binary/checkpoint/KB digest binding ve release gate;
9. gerçek model backend'i için Unicode/subword tokenizer/model-artifact manifest,
   deterministic long-context packer ve hash-chain conversation ledger;
10. source/license/hash taşıyan Unicode FTS knowledge evidence ve
    content-addressed image/audio/video asset registry;
11. OpenVINO GenAI text, HTTPS OpenAI-compatible ve local multimodal worker
    backend'leri için fail-closed capability/provenance gateway; bounded worker
    pipes ve redirect-rejecting authenticated endpoint transport;
12. retrieved evidence için trusted policy/data role separation ve operator-owned
    strict structured JSON response contract gate'i;
13. model-emitted evidence citation/explicit abstention marker doğrulaması ve
    gerçek gateway child process'leri üzerinde privacy-conscious held-out LMM
    quality/release gate;
14. model manifest + quality report + target-domain evaluation card bağlayan,
    residency/safety/modality/SLO threshold'lu deterministic expert routing ve
    bağımsız panel planı;
15. actual gateway child-process latency ve backend-reported token ölçümü;
16. external algorithm/optimizer config, training contract ve artifact identity
    için repeated held-out bounded observation, conservative all-objective
    baseline comparison, robust Pareto ve no-replace decision gate'i;
17. actual LMM/operator tool planı için request+registry SHA binding, typed
    predecessor result reference, no-shell read-only executor, distinct verifier,
    privacy-conscious report ve gateway-ready evidence publication;
18. immutable governor state/decision artifact'leri için strict-LF canonical
    JSONL ledger, local lock + expected-state CAS, direct transition/batch/
    directive binding, privacy-safe inspect ve negative-gate verification.

Bunlar veri kaybı, silent duplicate, held-out contamination, capacity overflow,
restart operasyon riski, dar grounded-retrieval kalitesinin ölçülmeden publish
edilmesi, tool-result swap/unchecked action ve model input provenance riskini
azaltır. Semantic language quality, generative ability, semantic tool planning
veya multimodal competence kanıtlamaz; bunlar seçilen gerçek model ve ayrı
evaluation ile kanıtlanmalıdır.

## Gerçek LMM hedefi için dürüst ayrı proje sınırı

Bir LMM/LLM hedefi kabul edilirse uygun iş, `hdc_pretraining.f90` içine birkaç
option eklemek değildir. Ayrı bir architecture decision record ve en az aşağıdaki
kanıtlar gerekir:

1. model family/objective/parameterization ve numerical reproducibility
   contract'i;
2. Unicode/subword + modality-specific input data contract'i;
3. tensor/autodiff/optimizer/checkpoint ABI'si;
4. single-device correctness oracle ve gradient/numerical regression tests;
5. multi-device collective/failure/restart semantics;
6. target Intel CPU/iGPU/GPU/NPU veya seçilen accelerator üzerinde gerçek kernel
   execution mask, correctness ve measured throughput evidence;
7. licensed/consented dataset registry, semantic dedupe/contamination policy ve
   held-out evaluation protocol;
8. memory/SSD/network/resource scheduler ve experiment provenance/retention
   planı;
9. safety, privacy, misuse ve deployment threat model'i.

Bu koşullar olmadan "devasa LMM production training" iddiası teknik olarak
savunulabilir değildir. Mevcut Vegai roadmap'i buna alternatif olarak
**deterministik, scalable HDC production operation** sunar.

İşlemsel ayrıntılar için [PRODUCTION_PRETRAINING.md](PRODUCTION_PRETRAINING.md),
HDC data/checkpoint sözleşmesi için [PRETRAINING.md](PRETRAINING.md), continual
transaction için [CONTINUOUS_PRETRAINING.md](CONTINUOUS_PRETRAINING.md),
held-out retrieval gate/audit için [QUALITY_ASSURANCE.md](QUALITY_ASSURANCE.md),
gerçek model gateway/Unicode/multimodal boundary için [LMM_RUNTIME.md](LMM_RUNTIME.md),
measured domain expert routing için [LMM_PORTFOLIO_ROUTING.md](LMM_PORTFOLIO_ROUTING.md),
immutable shadow/canary/rollback release handoff için
[LMM_RELEASE_GOVERNANCE.md](LMM_RELEASE_GOVERNANCE.md), target-domain'de ölçülebilir LMM üstünlüğü roadmap'i için
[LMM_EXCELLENCE_ARCHITECTURE.md](LMM_EXCELLENCE_ARCHITECTURE.md) ve Fortran core
invariantları için [ARCHITECTURE.md](ARCHITECTURE.md) referanstır.
