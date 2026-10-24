# LMM Excellence Architecture: ölçülebilir üstünlük için mimari

## Dürüst hedef

“Her alanda bütün LLM'lerden daha iyi” bugün kanıtlanabilir bir ürün özelliği
veya tek bir kod değişikliği değildir. Farklı modeller; dil, reasoning,
factuality, latency, maliyet, güvenlik, multimodal görevler ve donanım üzerinde
farklı Pareto noktalarındadır. Ağırlıklar, lisanslı veri, hedef donanım ve
held-out benchmark olmadan evrensel üstünlük iddiası yanlış olur.

Vegai için doğru, denetlenebilir hedef şudur:

> Her tanımlı deployment domain'i için seçilen baseline'lara karşı kalite,
> güvenilirlik, latency, maliyet ve yönetişim ölçümleriyle daha iyi bir Pareto
> noktası kanıtlamak; ölçülmeyen alanlarda abstain etmek.

Bu belge, bu hedefe ulaşmak için gereken uçtan uca mimariyi tanımlar. Durum
alanları özellikle **uygulandı**, **arayüz hazır**, veya **harici gerçek model/
operasyon gerekir** diye ayrılmıştır.

## 1. Katmanlar ve mevcut durum

```text
licensed data / feedback / benchmarks
             │
             ▼
 data governance ──► training + alignment ──► signed model registry
             │                                       │
             │                                       ▼
             └────► held-out contamination gates ─► model portfolio/router
                                                        │
 durable dialogue ─► context + evidence + media ───────┤
                                                        ▼
                                             real LMM/VLM/AV backend
                                                        │
                   policy / tools / safety ────────────┤
                                                        ▼
                         quality evaluator + release gate + observability
```

| Katman | Vegai durumu | Gerçek üstünlük için eksik kanıt |
|---|---|---|
| Deterministic HDC/retrieval | **Uygulandı**; Fortran HDC, Resonator, source-owned retrieval, release gate | Yalnız kendi dar domain'indeki kalite ölçülür; generative LMM değildir |
| Model package/provenance | **Uygulandı**; artifact hash, tokenizer hash, symlink/TOCTOU kontrolleri | Onaylı ağırlık, lisans, revision ve supply-chain attestation |
| Real generation gateway | **Uygulandı**; OpenVINO GenAI text, HTTPS OpenAI-compatible, trusted worker | Gerçek seçilmiş modelin task-quality benchmark'ı |
| Long dialogue/evidence/media | **Uygulandı**; hash-chain ledger, token budget, FTS evidence, image/audio/video asset contract | Semantic memory, embedding/reranker ve gerçek encoder quality evaluation |
| Citation/abstention quality gate | **Uygulandı**; held-out real gateway runner ve release threshold | Factual entailment, human preference, safety/red-team ve calibration benchmark'ları |
| Model portfolio/router | **Kısmi uygulandı**; provenance-bound target-domain expert routing, measured threshold ve independent-panel planı | Persistent policy service, shadow/counterfactual evaluation, canary/rollback ve live queue/SLO validation |
| Verified tool-use/reasoning boundary | **Kısmi uygulandı**; request+registry-bound typed plan, sequential tool-result dataflow, no-shell executor, distinct verifier ve evidence artifact | Real OS sandbox/network policy, side effects+human confirmation, verifier independence audit, benchmarked planning/tool-use quality ve semantic adjudication |
| Algorithm/optimizer evidence governor | **Uygulandı**; immutable algorithm/optimizer/training/artifact identities, repeated held-out bounded metrics, Bonferroni-Hoeffding confidence ve robust Pareto fail-closed selection | Gerçek bağımsız replication, benchmark contamination audit, energy/throughput measurement ve selected optimizer'ın gerçek training run kanıtı |
| Neural pretraining/alignment | **Harici gerçek model gerekir**; EPO yalnız hangi kanıtlanmış external config'in seçilebileceğini govern eder | Tensor runtime, distributed optimizer, data mixture, loss/eval curves |
| Multimodal reasoning | **Arayüz hazır**; assets + worker protocol | Licensed paired data, VLM/ASR/video encoder, modality task scores |
| Accelerator scheduler | **Arayüz hazır**; requested device/provenance, resident service benchmark methodology | Gerçek CPU/iGPU/GPU/NPU placement, throughput/energy evidence |
| Safety/governance | **Kısmi**; immutable inputs, source/license fields, privacy-conscious reports | PII/rights pipeline, policy classifier, red-team, incident response |

## 2. Uygulanan release-quality control plane

`tools/lmm_quality_evaluation.py` bir modelin yanıtını düzeltmez, template
üretmez veya LLM-as-a-judge kullanmaz. Operatorün sahip olduğu held-out JSONL
case'leri aşağıdaki şekilde çalışır:

1. Case, gerçek `vegai-lmm-request/v1` request'i ve backend'e verilecek gerçek
   `vegai-lmm-evidence/v1` bundle'ı taşır.
2. Immutable argv JSON'u içindeki `{request}` / `{evidence}` placeholder'ları
   private temporary dosyalara değiştirilir; shell kullanılmaz.
3. Aynı `lmm_gateway.py --print-json` komutu gerçek model backend'ini çağırır.
4. Modelin kendi yanıtındaki `[[vegai-source:<id>]]` ve
   `[[vegai-abstain]]` marker'ları deterministic olarak denetlenir.
5. Report; ham prompt/yanıt yerine hash, byte count, citation ID ve generation
   provenance hash'i taşır; case'ler boyunca model/gateway/tokenizer/worker
   runtime identity'sinin sabit kaldığını fail-closed denetler.
6. Case-pass, required-citation recall, invalid-citation rate ve abstention
   expectation accuracy threshold'ları release'i fail-closed durdurabilir.

Bu, kaynak ilişkisini ölçer. “Model marker yazdı” ile “iddia doğru” aynı şey
değildir; aşağıdaki bağımsız eval katmanları zorunludur.

## 3. Model portfolio ve routing tasarımı

Tek model her domain için optimal değildir. Production router aşağıdaki immutable
model manifest alanlarını kullanmalıdır: context window, modal capability,
tokenizer, artifact revision, license ve gerçek eval kartı.

`tools/lmm_portfolio.py`, bunun repository içindeki fail-closed control-plane
uygulamasıdır. Candidate'i yalnız model manifest SHA'sına bağlı, geçmişte başarılı
citation/abstention quality report'u ve target-domain evaluation card'ı varsa
seçer. Domain/residency/safety/modality/context/latency/cost threshold'ları
geçmeyen candidate için “daha zayıf modelle devam et” cevabı üretmez. `single-best`
gerçek seçilmiş gateway'i çağırabilir; `independent-panel` farklı provider/family
group'ları için provenance-bound plan üretir, fakat semantik adjudication veya
çoğunluk oylamasını kalite kanıtı diye uydurmaz. Ayrıntılı sözleşme:
[LMM_PORTFOLIO_ROUTING.md](LMM_PORTFOLIO_ROUTING.md).

### 3.1 Router kararı

Her request için policy, ölçülmüş değerlerle şu sırayı izlemelidir:

1. **Hard gate:** veri residency, license, modality, context, safety tier ve
   tool permission uygun mu?
2. **Evidence gate:** kaynak gerekiyorsa retrieval/citation-capable model ve
   `evidence-or-abstain` policy seçilir.
3. **SLO gate:** queue, deadline, token budget ve cost ceiling'e uyan model
   seçilir.
4. **Quality policy:** yalnız target-domain held-out score'u ve calibration
   raporu yeterli olan model/reranker route edilir.
5. **Fallback:** quality/SLO/safety gate başarısızsa daha küçük model üretmez;
   güvenilir fallback veya explicit abstention döner.

Router asla model adına, vendor pazarlama iddiasına veya yalnız benchmark
ortalamasına göre route etmemelidir. Her policy revision hash'lenmeli, shadow
traffic üzerinde counterfactual olarak ölçülmeli ve canary rollback desteklemelidir.

### 3.2 Gereken yeni production bileşenleri

Bunlar bu depoda sahte olarak uygulanmamıştır; gerçek deployment'ta gerekli
harici servislerdir:

- signed model registry + SBOM/license/weight attestation;
- persistent model servers (OpenVINO/vLLM/benzeri) ve queue-aware scheduler;
- per-model tokenizer/chat-template conformance testleri;
- model/reranker/embedding version registry;
- signed/persistent router policy service, shadow/counterfactual telemetry collector ve traffic-enforcing deployment controller;
- tenant isolation, rate limit, quota ve audit log retention policy.

Bu repo artık `lmm_release_governor.py` ile frozen portfolio/task/model/gateway/
held-out identity'lerini aggregate-only shadow/canary kararına, ayrıca
`lmm_release_registry.py` ile verified artifact'leri single-host authoritative
expected-head CAS/hash-chain ledger'ına bağlayan bir control-plane contract'i
sağlar. Persistent signer/remote registry, live traffic enforcement, telemetry
integrity ve distributed single-writer transaction yine external production
servisidir.

### 3.3 Algorithm/optimizer evidence governor

`tools/evidential_optimizer.py`, model router'ın yerine geçmeyen fakat onun
altındaki algorithm/optimizer training candidate'larını yöneten ayrı bir
control plane'dir. Candidate manifest gerçek architecture/optimizer config,
training contract ve artifact SHA-256 identity'lerini bağlar. Aynı immutable
held-out suite/protocol altında repeated bounded measurement yoksa ya da
conservative all-objective baseline improvement sertifikalanmazsa karar
yayınlamaz. Bu, “dünyanın en iyi optimizer'ı” marketing iddiası değil,
operator-declared target domain için falsifiable selection gate'tir. Matematik,
provenance, no-replace publication ve bağımsız verifier:
[EVIDENTIAL_OPTIMIZER.md](EVIDENTIAL_OPTIMIZER.md).

### 3.4 Shadow/canary release governor ve local authoritative head

`tools/lmm_release_governor.py`, router'ın offline eligible candidate kararını
user-visible deployment'a otomatik sıçrama yaptırmadan önce denetler. Immutable
policy baseline/candidate portfolio'larını, selected routing task'ı, model
manifest'i, gateway command'i, evaluation card'ı ve passed held-out quality
report'unu bağlar. Policy quality, reliability, safety, cost ve latency için
ayrı bounded live metric zorunluluğu getirir. Consentli shadow aggregate'leri
pass etmeden canary başlatmaz; canary'de time-uniform conservative confidence
bound ile ancak tüm metric'ler non-regression/promotion şartını sağlarsa
`promote`, herhangi bir confident failure varsa `rollback`, diğer durumda
fail-closed `pending` üretir. Raw prompt/response veya telemetry identifier
state/decision artifact'ine yazılmaz.

Decision yalnız `shadow-mirror-only`, `bounded-canary`, `candidate-promoted` ya
da `baseline-only-rollback` external traffic directive'idir.
`tools/lmm_release_registry.py`, decision/state pair'ini policy/governor/source
identity, direct predecessor, consumed observation-batch ve directive binding
ile local strict-LF ledger'a kaydedebilir; stale veya competing child local CAS
ile reject edilir. Model server'a route yapmaz, provider weights'i attest etmez,
telemetry'nin dürüstlüğünü veya quality/safety label'larını doğrulamaz. Signed
append-only remote registry, distributed controller CAS, tenant/network
isolation, alerting ve emergency kill switch production owner tarafından
uygulanmalıdır. Tam schema, istatistik, privacy ve non-claim sınırları:
[LMM_RELEASE_GOVERNANCE.md](LMM_RELEASE_GOVERNANCE.md).

## 4. Eğitim ve data factory mimarisi

Modelin gerçek kalite sınırı ağırlıklar ve veridir. Vegai'nin mevcut streaming
schema/provenance katmanı bu factory'ye veri sınırı sağlar; aşağıdakiler ayrıca
zorunludur:

| Aşama | Zorunlu contract | Kabul kanıtı |
|---|---|---|
| Kaynak alma | consent, license, geography, retention, PII class | immutable dataset manifest + review |
| Temizleme | language ID, parser/decoder, PII removal, malware/poisoning scan | sampled audit + false-positive/negative çalışma |
| Dedupe | exact + semantic near-duplicate + benchmark contamination | train/validation/test leakage report |
| Data mixture | domain/language/modality weights ve drift policy | reproducible mixture revision |
| Tokenizer | Unicode normalization, script coverage, fertility/round-trip | target dil tokenizer test kartı |
| Pretraining | causal/seq2seq/multimodal objective, deterministic resume | loss, divergence, checkpoint recovery evidence |
| Post-training | instruction, preference, tool/citation, abstention, safety data | held-out preference/reliability scores |
| Continual learning | replay, temporal split, catastrophic-forgetting gate | old/new benchmark Pareto report |

“Pretraining hazır” demek için, gerçek tensor/autodiff/optimizer stack'i,
distributed checkpoint recovery ve en az bir lisanslı model/data run'ının
reproducible kanıtı gerekir. Fortran HDC pretraining bu neural objective'in
yerine geçmez.

## 5. Retrieval, memory, tool ve reasoning plane

Üstün domain güvenilirliği latent parametreye bırakılmamalıdır.

1. **Retrieval:** lexical FTS yanında gerçek embedding index + cross-encoder
   reranker; her source için ACL, temporal validity, license ve freshness.
2. **Evidence packing:** source budget, diversity, contradiction detector ve
   prompt-injection isolation. Mevcut FTS evidence/marker gate başlangıç
   sözleşmesidir; semantic relevance ölçülmelidir.
3. **Memory:** hash-chained raw dialogue, user-consented durable facts, scoped
   summaries, memory expiry ve deletion audit. Model summary'si source history
   yerine geçmemelidir.
4. **Tools:** `lmm_tool_orchestrator.py` typed schema, request/registry
   binding, idempotency key, no-shell timeout, result provenance ve declared
   distinct verifier ile **read-only** tool plane sağlar. OS sandbox, network
   egress policy, retry semantics, mutating side-effect confirmation ve human
   authorization ayrıca gerekir.
5. **Reasoning:** Plan/execute/verify ayrımı ve previous-result dataflow
   uygulandı; calculator/search/code tool'unun gerçek semantic verifier'ı,
   confidence calibration, plan quality ve unsupported-claim abstention
   evaluation'ı ayrı target-domain benchmark gerektirir.

Her tool/retrieval bağlantısı için attack simulation, stale-data testleri ve
source-to-answer trace gerekir. Contract ve negative-gate test kapsamı:
[VERIFIED_TOOL_ORCHESTRATION.md](VERIFIED_TOOL_ORCHESTRATION.md).

## 6. Multimodal plane

Mevcut asset registry bir decoder veya semantic encoder değildir. Üretim VLM/AV
katmanı için şunlar gerekir:

- görüntü OCR/layout, vision encoder ve image benchmark'ları;
- audio ASR/diarization, timestamp/token alignment ve audio safety;
- video sampling, temporal encoder, shot/event grounding ve long-video budget;
- modality fusion/cross-attention, modality-missing fallback;
- paired-data rights/consent, adversarial media/red-team ve provenance watermark
  politikası.

Her modality için model manifest'indeki capability, gerçek encoder output'u,
latency/memory profile'ı ve held-out task score ile doğrulanmalıdır. Asset kabul
edildi diye “model anladı” iddiası yapılmaz.

## 7. Performans ve donanım mimarisi

Hedef Intel Core Ultra sınıfı CPU/iGPU/NPU dahil geniş uyumluluk için:

- portable CPU baseline ve host-specific opt-in profile;
- dynamic batching, KV-cache budget, prefix cache ve admission control;
- resident model server ile cold-load/warm-throughput ölçümünün ayrılması;
- model quantization accuracy/latency/memory Pareto kartı;
- CPU/GPU/NPU placement policy, transfer cost, thermal/power telemetry;
- RAM/dual-channel/NUMA/SSD bandwidth preflight ve spill policy;
- device failure fallback'ın aynı quality/safety gate'lerden geçmesi gerekir.

`lmm_gateway` benchmarkı yalnız configured child-process ölçüsünü verir. Gerçek
accelerator gain için provider execution trace, target-host benchmark ve quality
non-regression raporu gerekir.

## 8. Güvenlik, safety ve yönetişim

Kaliteli bir model sistemi şu sınırlara da uymalıdır:

- prompt injection ve data exfiltration threat model;
- PII/secret detection, tenant isolation, encryption/key rotation;
- source/license/copyright enforcement ve deletion/retention workflows;
- policy classifier, jailbreak/misuse red-team, human escalation;
- model artifact signing, dependency/SBOM vulnerability response;
- audit log minimization, response privacy, incident/revocation playbook.

Mevcut hash/provenance/citation mekanizmaları bu katmanın yalnız bir parçasıdır.

## 9. Üstünlük iddiası için release standardı

Bir domain'de “daha iyi” denebilmesi için release raporu en az şunları aynı model
revision, aynı temporal data cut ve aynı budget ile göstermelidir:

| Boyut | Ölçüm |
|---|---|
| Görev kalitesi | preregistered held-out benchmark, confidence interval, baseline karşılaştırması |
| Güvenilirlik | citation entailment, abstention calibration, contradiction/hallucination audit |
| Multilingual | hedef script/dil başına ayrı quality ve tokenizer metrics |
| Multimodal | image/audio/video task-specific held-out score ve failure taxonomy |
| Güvenlik | adversarial/red-team pass rate, leakage/misuse severity |
| Operasyon | p50/p95/p99 latency, token/s, cost, RAM/VRAM, energy ve error rate |
| Dayanıklılık | failure/restart, provider fallback, load/shedding ve data drift testleri |
| Yönetişim | license/consent/PII/retention audit pass |

Bir metrik kötüleştiğinde tek bir ortalama skor “daha iyi” iddiasını kurtarmaz.
Release, hedef deployment için açık Pareto policy'sini, benchmark sınırlarını ve
kalan riskleri yayınlamalıdır.

## 10. Uygulama sırası

1. Her target domain için frozen held-out citation/abstention suite'i oluşturup
   `lmm_quality_evaluation.py --fail-on-gate` ile CI/release'e bağlayın.
2. Her gerçek model için immutable target-domain evaluation card üretin ve
   `lmm_portfolio.py route` ile domain/residency/safety/SLO eşiklerini geçen
   expert'i veya bağımsız paneli seçin.
3. Frozen baseline/candidate route kararını `lmm_release_governor.py` policy'sine
   bağlayın; consented shadow → bounded canary → promote/rollback state chain'ini
   her transition'da `lmm_release_registry.py record` ile local expected-head
   CAS'e kaydedin. HA/multi-host ortamda buna ek olarak signed remote controller
   CAS'i kullanın. Her policy quality/reliability/safety/cost/latency live
   metrics'ini ve aggregate privacy protocol'ünü preregister etmelidir.
4. Gerçek lisanslı model package'i `lmm_model_manifest.py create/verify` ile
   bağlayın; tokenizer/chat-template conformance testini ekleyin.
5. Retrieval relevance + factual-entailment benchmark'ı, calibration ve human
   preference eval'i ekleyin.
6. Persistent server/router/accelerator telemetry'yi shadow ve canary traffic ile
   ölçün; telemetry integrity, provider/weight attestation ve emergency rollback
   ayrıca test edilmelidir.
7. Training/data factory ve safety/red-team katmanlarını gerçek verilerle
   devreye alın; her phase'de old benchmark non-regression gate uygulayın.
8. Ancak bu raporlar target domain'de gerekli Pareto üstünlüğünü gösterirse
   deployment claim'i yapın; diğer alanlarda kapsamı açıkça sınırlayın.

İlgili operasyon contract'i için [LMM_RUNTIME.md](LMM_RUNTIME.md), measured
expert routing/panel sözleşmesi için [LMM_PORTFOLIO_ROUTING.md](LMM_PORTFOLIO_ROUTING.md),
mevcut HDC/LMM ayrımı için [LMM_SCOPE_AND_GAP_ANALYSIS.md](LMM_SCOPE_AND_GAP_ANALYSIS.md)
ve genel release güvence tablosu için [QUALITY_ASSURANCE.md](QUALITY_ASSURANCE.md)
bakın.
