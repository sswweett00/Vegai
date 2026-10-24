# Eğitim sistemi denetimi — 2026-10

Bu denetim, “bütün zayıflıkları düzelt” hedefini kanıtsız mutlaklık olarak değil,
repo içindeki native HDC pretraining/continuous-pretraining yolunun sistematik,
testlenebilir fail-closed incelemesi olarak ele alır. Aşağıdaki tablo bulunan
riskleri, uygulanan kontrolü ve doğrulama kanıtını ayırır.

> Bu denetim bir Transformer/LLM/LMM eğitim sisteminin var olduğunu veya Vegai
> HDC modelinin genel kalite üstünlüğünü iddia etmez. Lisanslı domain corpus,
> bağımsız held-out annotation, hedef benchmark ve uygun compute olmadan gerçek
> domain/model kalitesi sonucu çıkarılamaz.

## Kapsam, bulgu ve kontrol matrisi

| Alan | Denetimde görülen zayıflık | Uygulanan fail-closed kontrol | Kanıt/test |
|---|---|---|---|
| Shard integrity | Mutable source path'ın child okuması ledger SHA'sından farklı byte görebilirdi | SHA/preflight öncesi ve sonrası doğrulanan private immutable shard snapshot | `vegai_continuous_pretraining` |
| Exactly-once state | Candidate checkpoint durable state'ten önce görünürse yanlışlıkla release sanılabilirdi | Candidate private çalışır; `state.json` tek commit marker; recovery orphan'ı replay eder | existing native continuous verifier |
| Native binary drift | Aynı state farklı executable ile devam edebilir, checkpoint ABI/semantiği karışabilirdi | Her yeni state exact executable SHA-256 + byte count bağlar; resume'da farklı binary child başlamadan reddedilir | `verify_continuous_quality_regression.py` binary-mutation subcase |
| Child availability / output exhaustion | Controller `subprocess.run` ile unbounded output ve timeout'suz native child başlatıyordu | Process-group termination, bounded stdout/stderr, explicit `--train-timeout-seconds`; output hash/byte evidence | new controller journal contract + 900MB contract verifier |
| Throughput attribution | Controller/evaluator/CI süresi “training speed” diye yanlış raporlanabilirdi | Timer yalnız native `train`/`continue` child spawn→exit sınırında; explicit exclusions, raw/active token/s | state `training_measurement`, large-run report |
| RAM/storage | Large shard/checkpoint için controller-level preflight yoktu | Her private shard/candidate için RAM/checkpoint free-space resource plan; not-ready fail | controller resource preflight; compact journal evidence |
| Quality scratch | Evaluator private checkpoint snapshot'ı scratch volume'u doldurabilirdi | Candidate checkpoint/binary/case-bound için scratch capacity gate | continuous quality controller path |
| Absolute quality | Correctly trained checkpoint'in target semantic behaviour'i bilinmiyordu | Optional strict held-out evaluator/policy, HMAC-redacted report and report provenance | existing `vegai_pretraining_quality_evaluation` |
| Catastrophic forgetting | Continuous shard candidate'i only data/checkpoint validation ile publish olabiliyordu | Generation-1 frozen anchor; every candidate absolute gate + aggregate/per-case regression policy; failure before checkpoint rename | `vegai_continuous_quality_regression` native + intentional forgetting subcase |
| Regression evidence | Failed candidate silinince operator quality düşüşünü denetleyemeyebilirdi | Private candidate silinir; raw-label-free quality/comparison/rejection evidence `quality-rejections/` altında kalır | intentional forgetting subcase |
| Report tampering | State yalnız report path taşısaydı completed checkpoint qualified gibi gösterilebilirdi | Report size/SHA, strict duplicate-key JSON snapshot, provenance/status/pass decision audit; status fail-closed | tampered report status subcase |
| Policy laundering | Forgetting görüldükten sonra tolerans/policy gevşetilerek retry yapılabilirdi | Cases, absolute policy, regression policy, evaluator and comparator identities immutable state binding | policy-swap subcase |
| Large-workload units | “900mb” binary/decimal ambiguity yaratır | Fixed documented `900MB=900,000,000` and `90MB=90,000,000` physical bytes; fixture manifest binds exact bytes | small-byte orchestration contract + manual workflow |
| Synthetic-data overclaim | Fixture result real language/domain competence diye sunulabilirdi | Explicit provenance/limitations in tool, reports, workflow summary and docs | docs and report contracts |

## Uygulama sınırları / açık operatör kararları

Aşağıdakiler kodla dürüstçe “çözülmüş” sayılmaz; bunlar deployment veya başka bir
model/runtime gerektirir:

1. **Gerçek corpus/evaluation governance:** Kullanıcı lisanslı 900 MB corpus,
   data-retention policy, independent held-out annotations veya target KPI
   vermedi. Repository run'ı yalnız deterministic synthetic topology kullanır.
2. **Model sınıfı:** Mevcut native learner HDC/VSA semantic-prototype modelidir;
   cross-entropy Transformer optimizer, distributed data-parallel gradient
   training, multimodal foundation weights veya genel LLM capability yoktur.
3. **GPU/NPU assertion:** Resource plan adapter inventory yapar. CPU binary,
   yalnız validated OpenVINO context-provider invocation gerçek kanıtı olmadan
   GPU/NPU training dispatch iddia etmez.
4. **Host-level durability and security:** Atomic-rename/fsync protocol local
   trustworthy filesystem, single writer and operator access control varsayar.
   Signed remote attestation, malicious privileged host defense, filesystem
   replication semantics ve power-loss certification ayrı deployment çalışmasıdır.
5. **Performance portability:** OMP thread count ve placement explicit'tir ama
   dual-channel RAM/SSD/NUMA/CPU mikro-mimarisi için evrensel speedup sözü
   değildir; actual artifact'taki child-only timing hedef hostta ölçülmelidir.

## Required evidence for a production claim

Bir gerçek domain continuation release'i için en az şunlar gerekir:

- rights/license/provenance bağlanmış train/continuation corpus;
- target-domain dışı ve model-selection dışı tutulmuş human-reviewed annotation
  seti; case/policy threshold calibration kaydı;
- native binary/toolchain/runtime/device provenance;
- resource, timing, checkpoint and regression report artifacts;
- independent reproduction/review and an explicit rollback/promotion policy.

`CONTINUOUS_QUALITY_REGRESSION.md`, controller CLI contract'i ve exact decimal
900 MB → 90 MB controlled workflow için ayrıntılı uygulanabilir sözleşmedir.
