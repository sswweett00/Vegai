# LMM release governance: shadow, canary ve fail-closed rollback

`tools/lmm_release_governor.py`, gerçek bir LMM deployment platformunun yerine
geçmeyen, **artifact identity'lerine bağlı karar düzlemi**dir. Amacı bir
portfolio router tarafından zaten target-domain için uygun bulunmuş baseline ve
candidate'i önce consented shadow traffic'te, sonra sınırlı canary'de ölçmek;
sonuç yeterince güçlü değilse candidate'i otomatik olarak genişletmemektir.

Bu katman model eğitmez, ağırlık indirmez, model server çalıştırmaz, kullanıcı
trafiğini bölmez ve telemetry uydurmaz. External deployment controller'ın
ürettiği immutable decision artifact'i uygulaması gerekir. Bu nedenle bir
`promote` sonucu Gemini-sınıfı kapasite, genel factuality, güvenlik veya
herhangi bir benchmark üstünlüğü kanıtı değildir.

İlgili sınırlar:

- candidate seçimi ve frozen held-out card binding:
  [LMM_PORTFOLIO_ROUTING.md](LMM_PORTFOLIO_ROUTING.md);
- actual model/gateway/runtime contract'i: [LMM_RUNTIME.md](LMM_RUNTIME.md);
- production roadmap ve dışarıda kalması gereken neural/data/compute sistemi:
  [LMM_EXCELLENCE_ARCHITECTURE.md](LMM_EXCELLENCE_ARCHITECTURE.md).

## 1. Güvenlik ve karar özeti

Araç şu zinciri kurar:

```text
immutable policy
  ├─ immutable routing task
  ├─ baseline portfolio → model manifest + gateway command + evaluation card + held-out quality report
  └─ candidate portfolio → model manifest + gateway command + evaluation card + held-out quality report
          │
          ▼
  immutable release state 0 (shadow veya canary)
          │  strict LF aggregate observations; no raw prompt/response
          ▼
  immutable state N + immutable aggregate-only decision
          │
          ├─ shadow_pass → bounded canary directive
          ├─ canary_insufficient_evidence / canary_pending → canary korunur
          ├─ promote → %100 candidate directive
          └─ rollback → %0 candidate directive
```

Her policy şunları zorunlu kılar:

1. `single-best`, bir candidate'li immutable routing task;
2. baseline ve candidate için yalnız label değil, farklı immutable
   model/gateway/runtime deployment identity;
3. target-domain held-out quality/reliability/safety, residency, modality,
   context, cost ve latency gate'lerini geçmiş portfolio candidate;
4. en az birer `quality`, `reliability`, `safety`, `cost`, `latency` live
   metric policy'si;
5. bounded metric değerleri, observation sayısı ve confidence budget;
6. optional fakat policy ile açıkça etkinleştirilmiş shadow sample ve ardından
   bounded (`<= 5,000` bp) canary.

Candidate label'i tek başına deployment farkı sayılmaz: governor model manifest,
gateway command ve held-out runtime identity'sinden ayrıca physical deployment
digest üretir; baseline ile aynı digest'e sahip alias candidate reddedilir.

Policy/portfolio/model/gateway/evaluation/quality artifact'i işlem sırasında
değişirse veya telemetry/state şeması, stage, identity, replay koruması ya da
confidence şartı bozulursa işlem exit `2` ile durur; yeni decision/state
oluşturmaz. Inconclusive kanıt **promote değildir**.

## 2. Artifact sözleşmeleri

### 2.1 `vegai-lmm-release-policy/v1`

Policy tek JSON object'tir ve tam olarak aşağıdaki top-level alanları taşır:

```json
{
  "schema_version": "vegai-lmm-release-policy/v1",
  "id": "coding-r2026-10",
  "task": {"path": "task.json", "sha256": "...64 lowercase hex..."},
  "baseline": {
    "portfolio": {"path": "baseline-portfolio.json", "sha256": "..."},
    "candidate_id": "baseline-r17"
  },
  "candidate": {
    "portfolio": {"path": "candidate-portfolio.json", "sha256": "..."},
    "candidate_id": "candidate-r18"
  },
  "shadow_basis_points": 250,
  "shadow_minimum_observations_per_arm": 400,
  "canary_basis_points": 500,
  "minimum_observations_per_arm": 800,
  "confidence_delta": 0.05,
  "metrics": []
}
```

`task` ve iki `portfolio` file reference'i hash'iyle bire bir eşleşmelidir.
Relative path policy'nin dizinine göre çözülür; declared dependency symlink
olamaz. Portfolio yükleyicisi model manifest, gateway command, evaluation card
ve held-out quality report binding'ini yeniden kontrol eder. Policy yalnız
static runtime identity'nin held-out report'ta kaydedildiğini bağlar; gerçek
serving process'in aynı weights/runtime'ı kullandığını external attestation ve
telemetry collector doğrulamalıdır.

Shadow kapatmak için **hem** `shadow_basis_points` hem
`shadow_minimum_observations_per_arm` sıfır olmalıdır. Aksi halde ikisi de
pozitif olmalıdır. Shadow ve canary için gereken iki arm observation toplamı,
control plane'in 50.000 kayıttan oluşan immutable state bütçesine sığmalıdır.
`canary_basis_points` 1–5.000 arasındadır; baseline her zaman kalan user
traffic'i taşır. Shadow sample 0–10.000 bp olabilir, ancak shadow output hiçbir
kullanıcıya dönmemelidir.

### 2.2 Zorunlu live metric policy

Her `metrics[]` kaydı exact olarak şu alanları kullanır:

```json
{
  "id": "quality",
  "kind": "quality",
  "direction": "maximize",
  "lower_bound": 0.0,
  "upper_bound": 1.0,
  "promotion_limit": 0.91,
  "rollback_limit": 0.84,
  "maximum_regression_vs_baseline": 0.03
}
```

`kind`, `quality`, `reliability`, `safety`, `cost`, `latency` veya `custom`
olabilir. İlk beş kind'in her biri policy'de **tam bir kez** bulunmalıdır:

| Kind | Zorunlu yön | Örnek ölçüm ve üretim ön koşulu |
|---|---|---|
| `quality` | `maximize` | frozen sampled human/held-out judge veya preregistered task success; online proxy tek başına yeterli değildir |
| `reliability` | `maximize` | valid response / successful completion / verified tool success oranı; bağımsız replication tanımı gerekir |
| `safety` | `maximize` | policy/red-team classifier veya human safety review pass oranı; classifier'ın calibration'ı ayrıca ölçülmelidir |
| `cost` | `minimize` | operator-declared bounded cost unit (ör. microunit/request); pricing revision ve cache policy kayda alınmalıdır |
| `latency` | `minimize` | declared window'ın p95 veya başka önceden kayıtlı bounded latency statistic'i; queue/drop policy sabit olmalıdır |

`custom` metric eklenebilir. Tüm değerler finite olmalı ve declared
`[lower_bound, upper_bound]` içine girmelidir. Quality/reliability/safety
`maximize`, cost/latency `minimize` olmak zorundadır. Sınırların gerçek bir
ölçüm protocol'üne uyduğu tool tarafından kanıtlanamaz; bu policy owner'ın
sorumluluğudur.

Promotion için her metric candidate confidence bound'u promotion limit'ini
geçmeli ve baseline'a karşı maximum regression sınırını sağlamalıdır. Herhangi
bir metric için conservative rollback kanıtı varsa rollback, bütün-metric
promotion'dan önceliklidir. Diğer tüm durumlar `pending`dir.

### 2.3 `vegai-lmm-release-observation/v1`

Telemetry batch strict UTF-8, strict LF (`\n`) JSONL'dir. Boş satır, CRLF,
duplicate JSON key, 256 KiB üstü line, 10.000 üstü batch kaydı, out-of-range
metric veya extra/missing field reddedilir. Her satırın biçimi:

```json
{
  "schema_version": "vegai-lmm-release-observation/v1",
  "id": "opaque-observation-id",
  "replication_id": "opaque-replication-id",
  "policy_sha256": "...",
  "stage": "shadow",
  "arm": "candidate",
  "arm_identity_sha256": "...",
  "metrics": {
    "quality": 0.94,
    "reliability": 0.99,
    "safety": 1.0,
    "cost": 42.0,
    "p95-latency": 311.0
  }
}
```

`stage`, active state ile tam eşleşmelidir: shadow tamamlanmadan canary telemetry
kabul edilmez. `policy_sha256` ve arm deployment identity policy'den yeniden
hesaplanır. `id` release chain boyunca; `arm + replication_id` ise chain
boyunca unique olmalıdır. State ham ID'yi değil SHA-256 digest'ini saklar.

**Privacy kuralı:** `id` ve `replication_id` prompt, response, user ID,
tenant ID, URL, exception text veya reversible business identifier olmamalıdır.
Production collector random opaque ID ya da ayrı anahtarlı HMAC identifier
üretmelidir. Governor raw prompt/response, telemetry body, ID veya replication
ID'yi state/decision'a yazmaz. Tahmin edilebilir kısa identifier'in SHA-256'sı
brute-force edilebilir; bu nedenle synthetic verifier fixture'ındaki açık ID'ler
production örneği değildir.

### 2.4 State ve decision

`vegai-lmm-release-state/v1` immutable, no-replace JSON'dur. `sequence`,
`previous_state_sha256`, policy/task/baseline/candidate SHA'larıyla birlikte
`governor_sha256`, stage bazlı bounded sums/counts, hashed replay setleri ve
consumed-batch digest'i taşır. State `shadow`, `canary`, `promoted` veya
`rolled_back` phase'indedir. Şema şu semantic tutarlılığı da tekrar doğrular:

- sequence 0 yalnız `GENESIS`, empty aggregate ve policy'nin initial stage'iyle
  mümkündür;
- sequence, consumed immutable batch sayısına eşittir;
- shadow geçildiyse canary/terminal state içindeki shadow gate gerçekten pass
  etmiş olmalıdır;
- terminal `promoted` yalnız passed canary evidence ile, `rolled_back` yalnız
  active-stage rollback evidence ile mümkündür;
- terminal state yeniden observation kabul etmez.

`governor_sha256`, state'i yayımlayan control-plane implementation'a bağlar.
Governor source'u değişirse eski state yeni source ile inspect/advance edilmez;
operator historical tool artifact'ini signed registry'de saklayıp o sürümle
audit etmeli veya açık, yeni policy/release migration yapmalıdır. Bu fail-closed
uyumluluk kırılması bilinçlidir.

`vegai-lmm-release-decision/v1` raw telemetry yerine stage başına count, mean,
time-uniform confidence bounds, verdict booleans ve redacted dependency path
ile birlikte şu external directive'i taşır:

```json
{
  "traffic_directive": {
    "mode": "shadow-mirror-only",
    "candidate_basis_points": 0,
    "baseline_basis_points": 10000,
    "shadow_candidate_basis_points": 250,
    "enforcement": "external-deployment-platform-required"
  }
}
```

- `shadow-mirror-only`: user-visible candidate traffic 0; authorized input
  mirror oranı `shadow_candidate_basis_points`.
- `bounded-canary`: candidate user traffic policy'deki canary bp'dir.
- `candidate-promoted`: candidate 10.000 bp'dir.
- `baseline-only-rollback`: candidate ve shadow traffic 0'dır.

State ve decision O_EXCL/no-replace ile yayımlanır, file `fsync` edilir ve host
izin verirse parent directory de `fsync` edilir. Paired state/decision write
hatasında tool best-effort olarak yeni state'i temizler. Güç kaybı, NFS/object
store consistency veya iki dosya arasında gerçek distributed transaction
sağlamaz; production controller signed append-only registry/transaction ile
orphan artifact recovery yapmalıdır.

## 3. İstatistiksel karar kuralı

Her arm/metric/sample size için iki taraflı time-uniform Hoeffding radius:

```text
radius(n) = range * sqrt(log((2 * arms * metrics * n * (n + 1)) / delta) / (2 * n))
```

kullanılır. `n(n+1)` budget allocation'ı üzerinden tüm sequential inspection
zamanlarında union bound verir; fixed-n interval'i tekrar tekrar bakarak
optional-stopping ile promotion üretmez. Bu oldukça konservatiftir ve küçük
canary'lerde özellikle `pending` kalması normaldir.

Bu güvence yalnız şunlar doğruysa anlamlıdır:

- her observation declared arm/stage için bağımsız veya protocol tarafından
  geçerli biçimde bounded bir replication'dır;
- metric aralığı, aggregation window, missing-data, timeout, retry, outlier ve
  p95/cost hesaplama kuralı policy yayımlanmadan önce sabitlenmiştir;
- telemetry collector arm identity/runtime revision/tenant isolation'ı doğru
  bağlamış ve manipülasyona karşı korunmuştur;
- shadow ve canary population'ları policy'nin target deployment population'ını
  temsil eder.

Araç bunları gözlemleyemez. Correlated session, sampling bias, benchmark
contamination, poisoned telemetry veya incorrect safety classifier için bu
bound semantic doğruluk garantisi değildir.

## 4. Kullanım akışı

Önce real held-out evaluation ve routing artifact'leri hazırlanır. Bunların
synthetic verifier fixture'larıyla değiştirilmesi release kanıtı değildir.

```bash
# 1) Immutable initial shadow/canary state VE controller'ın tüketeceği directive.
python3 tools/lmm_release_governor.py init \
  --policy releases/r2026-10/policy.json \
  --state-output releases/r2026-10/state-000.json \
  --decision-output releases/r2026-10/decision-000.json

# 2) Controller decision-000'ı doğrular; shadow phase ise yalnız consented
#    mirror telemetry toplar. Batch raw prompt/response içermez.
python3 tools/lmm_release_governor.py advance \
  --policy releases/r2026-10/policy.json \
  --state releases/r2026-10/state-000.json \
  --observations releases/r2026-10/shadow-001.aggregate.jsonl \
  --state-output releases/r2026-10/state-001.json \
  --decision-output releases/r2026-10/decision-001.json

# 3) Shadow pass sonrasında controller canary directive'i uygular; canary
#    observation batch'iyle aynı immutable transition çağrılır.
python3 tools/lmm_release_governor.py inspect \
  --policy releases/r2026-10/policy.json \
  --state releases/r2026-10/state-001.json \
  --decision-output releases/r2026-10/decision-001-audit.json
```

Her output path yeni olmalıdır; input veya var olan artifact overwrite edilmez.
Controller şunları kendi trust boundary'sinde yapmalıdır:

1. policy/state/decision SHA-256 zincirini ve authorized signer/registry kaydını
   doğrulamak;
2. directive phase'i atomik/idempotent biçimde route config'e uygulamak;
3. model server revision, tokenizer/template, tool policy, provider endpoint,
   weight/SBOM attestation ve tenant/network policy'yi policy-bound deployment
   identity ile doğrulamak;
4. raw request/response'u governor'a göndermeden privacy-preserving aggregate
   telemetry üretmek;
5. emergency kill switch, manual safety rollback, alerting, audit retention,
   incident response ve stale-decision expiry uygulamak.

Aynı state'ten iki ayrı advance branch'i teknik olarak farklı immutable
artifacts oluşturabilir. Governor single-writer distributed lock/registry
olmadığı için controller compare-and-swap expected head veya append-only signed
release ledger kullanmalı ve yalnız bir child state'i authoritative kabul
etmelidir.

## 5. Test ve kapsam sınırı

```bash
python3 tools/verify_lmm_release_governor.py
make release-check
make lmm-check
```

Synthetic verifier; immutable governor/portfolio/model/gateway/evaluation/held-out
quality binding'ini, initial shadow directive'ini, shadow-disabled direct-canary
policy'sini, shadow pass → bounded canary geçişini, conservative promotion'u,
confident canary rollback'i,
aggregate-only privacy'yi, duplicate/replay/forged-identity/malformed JSON,
state semantic, tampered artifact, policy-completeness ve no-replace gates'i
denetler. Fixture worker ve metrics gerçek model ölçümü değildir.

Bu modül şunları **garanti etmez**:

- foundation model weights, neural tensor/autodiff/distributed training veya
  Gemini ölçeğinde compute/data;
- external telemetry'nin dürüstlüğü, online quality/safety label'ının gerçekliği
  veya provider'ın endpoint arkasında revision değiştirmemesi;
- traffic enforcement, network/OS sandbox, tenant isolation, key management,
  PII deletion/retention veya regulator compliance;
- shadow request'inin candidate tarafından safe işlendiği, response'un factual
  olduğu, p95 metric'in doğru hesaplandığı ya da promotion'ın universal model
  superiority gösterdiği.
