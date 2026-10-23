# Evidential Pareto Optimizer — kanıta bağlı algoritma ve optimizer seçimi

## Dürüst amaç

`tools/evidential_optimizer.py`, Vegai'nin native HDC çekirdeğine sahte bir
AdamW/Transformer implementasyonu eklemez. Bunun yerine gerçek bir AI sisteminin
**algoritma + optimizer + training contract + model artifact** adaylarını,
ham held-out ölçüm kanıtından fail-closed olarak seçen bir meta-optimizer
uygular.

Bu tasarımın adı **Evidential Pareto Optimizer (EPO)**'dur. “Dünyadaki en iyi
optimizer” veya “her modelden daha zeki” iddiası değildir; böyle bir iddia,
lisanslı data, aynı benchmark, hedef donanım, bağımsız tekrarlar ve rakip
sistemler olmadan doğrulanamaz. EPO'nun dar ve doğrulanabilir iddiası şudur:

> Bir operatorün sabitlediği held-out benchmark üzerinde, bounded bağımsız
> tekrarların Bonferroni-düzeltilmiş Hoeffding güven sınırları ile her ilan
> edilmiş objective'te baseline'a göre iyileşme sertifikalanmadıkça hiçbir
> algoritma/optimizer deployment kararı yayınlanmaz.

EPO bir model eğitmez, prompt/cevap üretmez, metric uydurmaz veya high score
bulmak için test setini kullanmaz. Gerçek neural optimizer (AdamW, Lion,
Shampoo, Muon vb.), HDC update policy'si veya başka bir training programı dış
sistemde kalır; onun immutable config/artifact hash'leri bu control plane'e
bağlanır.

## Neden meta-optimizer?

Tek bir optimizer, model ailesi, data mixture, batch topology, accelerator veya
domain için evrensel optimum olamaz. Örneğin quality, reliability, safety,
latency, maliyet ve enerji farklı Pareto noktaları oluşturur. Bu nedenle EPO:

1. `algorithm` ve `optimizer` family/config SHA-256 identity'sini ayrı taşır;
2. pinned training contract ve model artifact identity'sini de zorunlu kılar;
3. aynı immutable `heldout` suite revision + protocol altında tekrarlı ölçüm
   ister;
4. her metric için operatorün ilan ettiği alt/üst fiziksel veya sayısal bound'u
   kullanır;
5. confidence-bound ile baseline iyileşmesini kanıtlayamazsa abstain eder;
6. birden fazla kanıtlanmış aday varsa ancak policy'de açıkça belirtilmiş
   lexicographic priority ile robust Pareto frontier'den seçer.

Bu, marketing score'larına veya model adına göre route etmekten daha güvenli bir
algoritma/optimizer selection mekanizmasıdır; gerçek optimizer'ın matematiksel
veya semantic doğruluğunun yerine geçmez.

## Girdi sözleşmeleri

Tüm şemalar strict JSON/JSONL'dir; unknown veya missing field fail-closed olur.
Her ID ASCII identifier, her identity 64 karakterlik lowercase SHA-256'dir.
Ham eğitim verisi, user prompt'u, model output'u veya hyperparameter payload'u
karar raporuna yazılmaz.

### 1. Policy

`vegai-evidential-optimizer-policy/v1` şunları sabitler:

```json
{
  "schema_version": "vegai-evidential-optimizer-policy/v1",
  "id": "target-domain-policy",
  "benchmark": {
    "suite_id": "target-heldout",
    "suite_revision_sha256": "<sha256>",
    "protocol_sha256": "<sha256>",
    "split": "heldout"
  },
  "baseline_candidate_id": "approved-baseline",
  "confidence_delta": 0.05,
  "minimum_observations_per_candidate": 128,
  "selection_strategy": "robust-pareto-lexicographic/v1",
  "metrics": [
    {
      "id": "quality",
      "direction": "maximize",
      "lower_bound": 0.0,
      "upper_bound": 1.0,
      "acceptance_limit": 0.80,
      "minimum_improvement_vs_baseline": 0.02,
      "dominance_margin": 0.0
    }
  ],
  "priority_metrics": ["quality"]
}
```

`split` yalnız `heldout` olabilir. Policy'deki metric lower/upper bound'ları
Hoeffding eşitsizliğinin koşuludur; bunlar ölçümün gerçek, doğru veya bağımsız
olduğunu tek başına kanıtlamaz. `acceptance_limit` maximize metric için
conservative lower bound, minimize metric için conservative upper bound ile
kontrol edilir.

### 2. Candidate manifest

`vegai-evidential-optimizer-candidates/v1` en az baseline ve bir challenger
taşır. Her candidate:

```json
{
  "id": "candidate-a",
  "algorithm": {
    "family": "operator-declared-algorithm",
    "config_sha256": "<sha256>"
  },
  "optimizer": {
    "family": "operator-declared-optimizer",
    "config_sha256": "<sha256>"
  },
  "training_contract_sha256": "<sha256>",
  "model_artifact_sha256": "<sha256>"
}
```

`config_sha256`, gerçek hyperparameter/configuration paketinin hash'idir.
EPO config içeriğini yorumlamaz veya değiştirmez. Aynı benchmark altında farklı
architecture/optimizer kombinasyonlarının karşılaştırılabilmesini sağlar.

### 3. Observation JSONL

Her `vegai-evidential-optimizer-observation/v1` satırı bir gerçek ayrı
replication'ın aggregate metric'lerini taşır. Satırdaki candidate identity,
manifest ile bit-bit eşleşmek zorundadır; `run_id` global, `replication_id`
candidate başına unique olmalıdır. Her observation aynı benchmark identity'yi
taşır ve policy'deki metric setinin tamamını, başka hiçbir metric olmadan
sağlar.

Bir candidate'ın observation'ı eksikse selector o candidate'ı görmezden
gelmez; **tüm declared candidate'lar** minimum sample count'a ulaşana kadar
gate fail olur. Bu, zayıf veya pahalı competitor'ı sessizce dışarıda bırakma
riskini azaltır.

## Matematiksel selection kuralı

Metric `m`, bounded range `[a_m,b_m]`, `n` bağımsız observation, `C` candidate
ve `M` metric için EPO şu iki taraflı, union-bound düzeltilmiş yarıçapı kullanır:

```text
r_m(n) = (b_m - a_m) × sqrt(log((2 × C × M) / δ) / (2 × n))
LCB_m  = max(a_m, mean_m - r_m)
UCB_m  = min(b_m, mean_m + r_m)
```

Bu hesap deterministic olarak report'a yazılır. Bir non-baseline candidate
ancak aşağıdaki tüm koşullarda eligible olur:

- bütün metric'lerde required sample count sağlanır;
- maximize metric için `LCB ≥ acceptance_limit`, minimize metric için
  `UCB ≤ acceptance_limit`;
- maximize metric için
  `LCB(candidate) ≥ UCB(baseline) + minimum_improvement_vs_baseline`;
- minimize metric için
  `UCB(candidate) ≤ LCB(baseline) - minimum_improvement_vs_baseline`.

Yani bir candidate yalnız weighted average veya tek score sayesinde seçilmez;
ilan edilmiş **her objective** üzerinde baseline'a göre conservative iyileşme
zorunludur. Eligible candidate'lar arasında robust dominance yine LCB/UCB ve
`dominance_margin` ile hesaplanır. Nondominated frontier birden fazla üyeye
sahipse policy'nin açık `priority_metrics` sırası kullanılır; gizli ağırlık,
vendor adı veya candidate sırası kullanılmaz.

Bu yalnız aşağıdaki varsayımlar altında istatistiksel anlam taşır:

1. replication'lar gerçekten bağımsızdır;
2. declared metric bounds doğrudur;
3. held-out suite/protocol tüm candidate'larda aynıdır;
4. test seti model/data/optimizer tuning için kullanılmamıştır;
5. analysis pre-registered, frozen ve one-shot'tır; aynı held-out sonucu görerek
   candidate/metric/sample count değiştiren adaptive peeking için ayrı bir
   sequential alpha-spending policy gerekir.

Bu varsayımlar bozulursa Hoeffding sayılarını hesaplamak mümkündür ama resulting
selection'in güvenilir olduğu iddia edilemez. EPO bu dış gerçekleri tek başına
denetleyemez; suite provenance, data governance, contamination audit ve
independent review gerekir.

## Fail-closed yayınlama ve provenance

`select` önce policy, candidate manifest ve observations dosyalarının bounded
UTF-8 identity'sini alır. Karar öncesinde her dosyayı yeniden hash'leyerek
TOCTOU değişimini reddeder. Karar JSON'u:

- input path/byte-count/SHA-256 identity'lerini;
- aggregate mean/LCB/UCB değerlerini;
- algorithm/optimizer family ve config hash'lerini;
- confidence formülünü ve açık varsayımlarını;
- robust frontier ile seçilen candidate'i

taşır. Raw observation, run/replication ID, prompt, answer, training data ve
hyperparameter içeriğini taşımaz.

Yayın `O_CREAT|O_EXCL` ile no-replace yapılır. Output mevcutsa veya immutable
input ile aynı path ise işlem hata verir. Gate fail (`exit 3`) durumunda
herhangi bir deployment decision dosyası yayınlanmaz.

## Kullanım

```bash
python3 tools/evidential_optimizer.py select \
  --policy governed/optimizer-policy.json \
  --candidates governed/algorithm-optimizer-candidates.json \
  --observations runs/heldout-observations.jsonl \
  --output releases/optimizer-decision.json

# Dependency-free synthetic contract verification
make optimizer-check
```

Exit kodları:

| Kod | Anlam |
|---:|---|
| `0` | Bir candidate tüm gate'leri geçti ve no-replace karar yayınlandı |
| `2` | Schema, provenance, identity, input veya publication hatası |
| `3` | Geçerli evidence mevcut ama hiç candidate tüm confidence-bound gate'leri geçmedi |

## Test kapsamı

`tools/verify_evidential_optimizer.py` temporary synthetic protocol fixture ile
şunları sınar:

- algorithm + optimizer identity hash binding;
- held-out benchmark/protocol equality;
- 128 tekrarlı multi-objective Hoeffding/robust Pareto selection;
- deterministic byte-identical decision report;
- output no-replace davranışı;
- forged candidate identity ve duplicate JSON key reddi;
- incomplete veya non-dominating candidate için exit `3` ve karar yayınlanmaması;
- raw run/replication ID'lerinin karar report'unda görünmemesi.

Bu test bir optimizer'ın gerçek model quality'sini ölçmez. Gerçek deployment
öncesinde target-domain suite, independent replication, hardware/maliyet ölçümü,
contamination/safety evaluation ve operator-owned model training evidence
gereklidir.
