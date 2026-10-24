# Doğrulama, istatistik ve kabul kriterleri

Vegai'nin doğrulaması üç birbirini tamamlayan seviyededir:

1. **native Fortran 2023 testleri** (`test/test_all.f90`): derlenen gerçek
   API'nin contract/property testleri;
2. **bağımsız Python referans modeli** (`tools/verify_reference.py`): bit
   cebiri, standart SplitMix64 oracle'ı ve demo senaryosunun farklı bir dilde
   yeniden hesaplanması;
3. **CI checked build** (`.github/workflows/ci.yml`): GNU Fortran,
   `-fcheck=all`, floating-point trap, OpenMP ve CTest.

Bu katmanlar aynı bug sınıfını farklı şekillerde yakalamaya yöneliktir. Python
modeli Fortran derleyicisinin yerine geçmez; native testler de referans modelin
yerine geçmez.

## Çalıştırma matrisi

| Komut | Kapsam | Beklenen sonuç |
|---|---|---|
| `make check` | bağımsız reference/property/demo + stdlib control-plane contract'leri | tüm çağrılar `PASS` |
| `make data-check` | optional PyArrow kurulumu; streaming JSONL/JSON/TXT/Parquet/Arrow round-trip, corruption, production adapter ve curriculum compiler'ın optional Parquet/Arrow replay'i | `training data I/O verification: PASS` + `capability curriculum verification: PASS (... Parquet/Arrow=checked)` |
| `make capability-check` | stdlib-only policy-driven curriculum build/replay, strict text-only projection, held-out/source coverage, split/annotation conflict, no-output failure, controller-ledger ve tamper rejection | `capability curriculum verification: PASS` |
| `make lmm-check` | optional Rust Unicode/BPE tokenizer; long context, session ledger, knowledge evidence, image/audio/video asset registry, real worker protocol, held-out citation/abstention quality release gate, measured domain-expert routing/panel, typed verified read-only tool plan, immutable shadow/canary governor + single-host authoritative registry ve capability gate | `lmm runtime verification: PASS` + `lmm portfolio verification: PASS` + `LMM tool orchestrator verification: PASS` + `LMM release governor verification: PASS` + `LMM release registry verification: PASS` |
| `make release-check` | frozen portfolio/task/model/gateway/evaluation/held-out-quality identity, required five live metric policy, aggregate-only shadow/canary transition, conservative promotion/rollback, privacy/replay/tamper + local authoritative expected-head CAS/hash-chain gates | `LMM release governor verification: PASS (...)` + `LMM release registry verification: PASS (...)` |
| `make release-registry-check` | stdlib-only canonical strict-LF local ledger: `GENESIS`/state-SHA CAS, direct transition serialization, decision batch/directive binding, lock, hash-chain, policy identity ve privacy gates | `LMM release registry verification: PASS (...)` |
| `make optimizer-check` | algorithm/optimizer identity binding, repeated held-out bounded observations, Hoeffding confidence, robust Pareto, no-replace and negative gate | `evidential optimizer verification: PASS` |
| `make tool-check` | request/registry-bound typed read-only tool plan, predecessor output reference, no-shell executor, distinct verifier, evidence provenance/privacy and negative gates | `LMM tool orchestrator verification: PASS` |
| `python3 tools/lmm_tool_orchestrator.py execute --registry ... --request ... --plan ... --evidence-output ... --report-output ...` | gerçek configured tool'ları pass eden typed result'tan gateway-ready evidence'e dönüştürür | başarıda exit `0`; shell/tool/verifier/provenance/schema failure için exit `2`; raw request/argument/result report'a girmez |
| `python3 tools/evidential_optimizer.py select --policy ... --candidates ... --observations ... --output ...` | gerçek external algorithm/optimizer candidate'ları için all-objective baseline-improvement selection gate | seçimde exit `0`; geçerli ama yetersiz evidence için exit `3`; raw run/prompt/data report'a girmez |
| `python3 tools/lmm_quality_evaluation.py --cases ... --gateway-command-json ... --report ... --fail-on-gate` | gerçek configured model gateway ile held-out citation/abstention provenance ve release threshold | report yayınlanır; başarısız quality gate exit `3` |
| `python3 tools/lmm_portfolio.py route --portfolio ... --task ... --request ... --evidence ... --output ...` | immutable model/gateway/quality/evaluation-card binding ile target-domain candidate seçimi veya panel planı | eligible candidate yoksa exit `2`; raw request/response rapora girmez |
| `python3 tools/lmm_release_governor.py init/advance/inspect --policy ... --state... --decision-output ...` | frozen router candidate/baseline identity üzerinde aggregate-only shadow/canary state ve external traffic directive | malformed/tampered/replayed telemetry/state veya inconclusive kanıt exit `2`; raw prompt/response/telemetry ID decision'a girmez |
| `python3 tools/lmm_release_registry.py record/inspect --policy ... --ledger ...` | verified governor state/decision'larından privacy-safe single-host authoritative release head; `record` için `GENESIS` veya current state SHA expected-head token'ı | stale/competing/terminal child, mismatched batch/directive/policy/governor veya corrupt ledger exit `2`; signed remote registry/traffic enforcement değildir |
| `make dialogue-check` | real compiled chat ile held-out grounded/abstention metrikleri, report privacy/provenance, strict schema ve negative release gate | `dialogue evaluation verification: PASS` |
| `make quality-check` | real compiled HDC checkpoint inspect ABI; strict held-out semantic-neighbour case/policy, HMAC privacy, provenance, TOCTOU ve intentional failed gate | `pretraining quality evaluation verification: PASS` |
| `make test` | native cebir, error path, memory, reasoning | `Checks: ..., failures: 0` |
| `make debug` | bounds/runtime/FPE denetimli native test | başarısızlık yok |
| `make run` | 512+ sembollü demo | frame, analogy, 4 causal adım ve 3-factor Resonator PASS |
| `make pretrain` | HDC corpus CLI derlemesi | portable `build/vegai_pretrain` oluşur |
| `make chat` | TSV-grounded retrieval CLI derlemesi | portable `build/vegai_chat` oluşur; cevap generator değildir |
| `make native` | yalnız build-host CPU'suna tuned ikili | `build-native/vegai_pretrain` oluşur |
| `python3 tools/verify_hardware_compatibility.py` | CPU/OS/RAM/GPU/NPU inventory ve CPU-backend truth contract | `hardware compatibility verification: PASS` |
| `python3 tools/verify_training_resource_plan.py` | bounded RAM/checkpoint storage/OpenMP placement planı ve accelerator fail-closed contract | `training resource plan verification: PASS` |
| `python3 tools/verify_openvino_provider.py` | optional Intel OpenVINO provider preflight/truth contract | `OpenVINO provider verification: PASS` |
| `python3 tools/verify_openvino_context_worker.py` | uint32-lane worker protocol, neutral tie/complement padding, resident graph-cache reuse; runtime yoksa fail-closed | `OpenVINO context worker verification: PASS` |
| `python3 tools/verify_openvino_context_provider.py --pretrain-bin build/vegai_pretrain` | native-CPU bypass timing/equality guard + strict resident OpenVINO graph checkpoint equality (runtime yoksa SKIP) | `OpenVINO context provider verification: PASS` |
| `ctest --test-dir build` | CMake entegrasyonu | `vegai_fortran_properties`, `vegai_tokenless_signal_field`, `vegai_evidential_optimizer`, `vegai_lmm_tool_orchestrator`, `vegai_lmm_release_governor`, `vegai_lmm_release_registry`, donanım/OpenVINO contract testleri, `vegai_example_pretraining`, `vegai_dialogue_retrieval`, `vegai_dialogue_evaluation`, `vegai_lmm_runtime`, `vegai_lmm_portfolio`, `vegai_continuous_pretraining`, `vegai_production_pretraining`, `vegai_pretraining_quality_evaluation`, `vegai_capability_curriculum`, `vegai_example_production_dataset_preflight` PASS |

CI bunların checked CMake/OpenMP varyantını ve ayrıca optimize Make hedefini
çalıştırır. Rich-format CI adımı `requirements-dataio.txt` ile optional PyArrow
ve `requirements-lmm.txt` ile Rust `tokenizers` kurar; ardından
`tools/verify_training_data_io.py` ile JSONL/JSON/TXT, Parquet zstd/snappy ve
Arrow zstd round-trip; schema/missing-field/corrupt input rejection; bounded
physical-line/JSON parsing; SQLite exact dedupe/label conflict; parallel file
stream; strict HDC projection ve structured-source production foundry
adaptasyonunu çalıştırır. Bu doğrulama format-I/O contract'ıdır; Fortran HDC
semantic/throughput benchmarkı değildir.

## Native test kapsamı

### `hyper_types`

- standart SplitMix64 seed-0 ilk kelime oracle'ı (`E220A8397B1DCDAF`), aynı
  seed → bit-bit aynı hiper-vektör ve signed-ID seed ayrışması;
- farklı seed'ler için Hamming mesafesinin geniş ama anlamlı near-orthogonal
  aralığı (`7000 < H < 9400`);
- XOR identity ve self-inverse unbinding;
- forward/inverse permutation ve dimension wrap;
- three-way accumulator sonucu = boolean majority-3;
- bounded bit-sliced reset sonrası lazy carry-plane initialization'ın signed
  accumulator ile her cardinality'de bit-tam eşitliği;
- `int16` counter overflow'unda status ve **atomic state preservation**;
- tam 4096 bit noise, determinism ve invalid argument davranışı;
- Hebbian mask determinism ve değişen bit sayısının alt sınırı.

### `symbolic_memory`

- init/destroy lifecycle ve footprint;
- custom vector kayıtları;
- `-1` negatif ID lookup;
- duplicate, missing ID, empty cleanup ve capacity overflow hata yolları;
- full exact cleanup ve LSH-only cleanup metadata'sı;
- 3000-bit gürültü altında cleanup;
- Hebbian prototype değişiminden sonra LSH rebuild ve exact recall.

### `reasoning_core`

- 3-role frame encode/unbind;
- boyut uyumsuzluğu ve empty sequence hata yolları;
- n-gram sıra duyarlılığı;
- 3-pair aligned analogy;
- 4-rule causal bank üzerinden dört ardışık cleanup adımı;
- 50 distractor ile cleanup marjı.

### `resonator_network`

- üç slotlu, her slotta dört candidate bulunan exact factorization;
- dönen factor'ların yeniden bind edilip target ile bit-bit eşitliği;
- network best residual ile final certified residual'ın ayrı metadata'sı;
- invalid exact-verifier limiti ve uninitialized codebook hata yolları;
- duplicate hypervector kaynaklı iki exact tuple'ın `RN_WARN_AMBIGUOUS` ile
  görünür hale gelmesi;
- result allocatable workspace lifecycle.

Resonator iteratif fazı heuristic olduğundan test, yalnızca pozitif similarity
ile başarı kabul etmez; final `exact=.true.` ve `hamming_distance=0` şartını
arar. Bounded verifier birden fazla tuple bulduğunda count değerini ikiye
saturate ederek ambiguity'yi doğrular.

### `hdc_pretraining`

- fixed fixture corpus üzerinde exact unique/active vocabulary, raw/active/OOV
  token sayaçları ve ordered-window context pair sayısı;
- aynı input/config ile iki bağımsız default `hierarchical` run'ın base plane,
  semantic plane ve experience counter bakımından bit-tam eşitliği;
- repeated center token içeren record üzerinde serial ve OpenMP context-build
  run'larının base plane, semantic plane ve experience counter bakımından
  bit-tam eşitliği; adaptive default'un kısa record'larda serial path'i,
  `--parallel-min-contexts=1` override'ının parallel scheduler telemetry'sini
  seçmesi;
- `legacy` signed-counter ve hızlı `positional` bit-sliced encoder'ın
  bit-tam semantic-plane eşitliği;
- semantic prototype lookup ve self-excluded nearest-neighbour diagnostic;
- `hierarchical` ve `coherent` long-record encoder'larının ayrı semantic
  plane üretmesi; coherent fused/reference local-kernel eşitliği ve short-record
  `hybrid` eşitliği;
- binary checkpoint save/load, rebuilt hash index ve payload plane round-trip;
  raw header regression'ı ile fresh model writer'ın v7 yazdığının ve synthetic
  v5/coherent-v6 lineage'lerinin historical exact-mask sampler identity'sini
  koruduğunun doğrulanması;
- kesintisiz ikinci epoch ile checkpoint'ten resume edilen ikinci epoch'un
  bit-tam eşitliği;
- resume semantic config mismatch, strict long-token, strict non-ASCII byte ve
  strict >8.192-byte record hata yolları;
- iki shard üzerinde vocabulary extension: yeni token'ın active/queryable olması,
  mevcut semantic prototype'ın resetlenmemesi ve `OOV=0` training invariantı;
- exact-rate Hebbian update için complement input altında %25 = 4.096 Hamming
  flip property.

CI, küçük `toy_hdc_corpus.txt` ile `train → inspect → resume` smoke zincirini
ve `mini_turkish_ascii_pretrain.txt` ile manifest, dengeli-token cardinality,
bit-tam çift checkpoint, checksum-bit mutation rejection ve `train → inspect → resume`
integration zincirini çalıştırır. CTest'in `vegai_continuous_pretraining` entegrasyonu gerçek CLI ile
SHA-256 shard mutation rejection, immutable manifest digest pin'i, append-only
manifest prefix, OOV=0, stale private-candidate cleanup, idempotent restart ve
read-only status aracının shard/checkpoint/lock integrity gate'ini ayrıca sınar.
`vegai_production_pretraining`, gerçek compiled CLI ile deterministic foundry
byte equality, disk-backed exact record dedupe, `min_frequency=1` exact
vocabulary-capacity preflight, immutable shard/path rejection, record-level
train/validation leakage rejection, exactly-once generation, completed
state/payload health audit'i ve idempotent production resume'ı sınar. `vegai_example_production_dataset_preflight`,
source-controlled sentetik package'ın ledger/content-address/leakage contract'ını
ayrıca doğrular.

`vegai_dialogue_retrieval`, gerçek derlenmis `vegai_pretrain` ve `vegai_chat`
ikilileriyle runtime TSV response degisikliginin user-visible response'u da
degistirdigini, grounded olmayan query'nin abstain ettigini, strict unknown
prompt/malformed TSV failure'ini ve bounded REPL yolunu kontrol eder.
`vegai_dialogue_evaluation`, ayrı held-out JSONL seti ile exact expected-prompt
grounding ve expected abstention metriklerini, checkpoint/KB/binary digest
binding'ini, raw-prompt taşımayan report'u, strict schema rejection'ını ve
bilinçli failed release gate exit policy'sini test eder. Native Fortran property
testi ayni katmanda exact retrieval tie/margin, duplicate embedding, fixed
capacity, strict ASCII ingest ve session ring reset invariantlarini kapsar. Bu,
LLM kalite degerlendirmesi degil, veri-kaynakli retrieval contract'idir.

`vegai_lmm_runtime`, gerçek model ağırlığı üretmeden/indirmeden LMM integration
boundary'sini denetler: Unicode ve optional Rust BPE tokenizer, Turkish/Arabic/
CJK FTS evidence retrieval, hash-chain session, deterministic token-budget long
context, content-addressed image/audio/video asset, backend-owned worker output,
provenance report, model-emitted citation/abstention gate, privacy-conscious
held-out quality release gate, capability yoksa fail-closed rejection ve actual
gateway child-process benchmark sınırı. `vegai_lmm_portfolio`, ayrıca en yüksek
measured target-domain candidate seçimini, independent-group panelini, raw-text
privacy'yi, quality threshold fail-closed davranışını, wrong-model veya
unmeasured gateway command ve runtime-drift rejection'ını test eder.
`vegai_lmm_release_governor`, ayrı synthetic fixture ile frozen portfolio
binding, shadow→canary transition, conservative promotion/rollback,
aggregate-only privacy, replay/forged identity/state semantic/tamper ve required
quality/reliability/safety/cost/latency policy gates'ini doğrular.
`vegai_lmm_release_registry`, aynı governor fixture'ları üzerinde `GENESIS` ve
state-head CAS, competing child rejection, direct phase serialization,
consumed-batch/directive binding, strict-LF hash-chain/tamper, stale lock,
policy mismatch ve raw identifier/path privacy gates'ini doğrular. Bu testler
factuality, fluency veya multimodal reasoning benchmark'ı değildir;
[LMM_RUNTIME.md](LMM_RUNTIME.md), [LMM_PORTFOLIO_ROUTING.md](LMM_PORTFOLIO_ROUTING.md)
ve [LMM_RELEASE_GOVERNANCE.md](LMM_RELEASE_GOVERNANCE.md) contract'lerinin
integration testleridir.

CI önce `tools/hardware_compatibility.py` ile CPU/OS/mimari/RAM/GPU/NPU
inventory metadata'sını, sonra `tools/training_resource_plan.py` ile bounded
RAM/checkpoint-storage/OpenMP placement planını kaydeder. Generic plan GPU/NPU
compute kullanmadığını açıkça raporlar ve provider/device execution bilgisi
olmaksızın accelerator talebini fail-closed reddeder. Ayrı OpenVINO bridge
adımı optional runtime'ı kurar; önce explicit CPU fallback'ın native fused
kernel'e worker başlatmadan bypass ettiğini, ardından strict resident CPU
OpenVINO context graph ile Fortran CPU reference checkpoint byte equality'yi
test eder. `coherent` cross-fold encoder bu ABI tarafından desteklenmez ve
OpenVINO provider seçildiğinde validation aşamasında fail-closed reddedilir;
worker'a sessiz bir farklı context semantics gönderilmez. Worker verifier ayrıca
dynamic-batch compiled-model cache ve neutral tie/complement padding exactness'ini
denetler. Ardından portable
varsayılan Make binary ile aynı host için opt-in `native` binary'yi
`tools/compare_pretraining_build_profiles.py` üzerinden
ölçer. İki profile'ın checkpoint byte eşitliği zorunludur; süre oranı yalnız o
host için anlamlıdır.

Native testler fused raw-word/permuted/pair accumulator plane bit-eşitliğini,
fused/reference hybrid, hierarchical ve coherent replay eşitliğini,
repeated-center serial/OpenMP context-build eşitliğini; tüm fresh context
mode'ları için v7 checkpoint round-trip'ini ve v5 legacy sampler resume'ini
denetler. `tools/compare_pretraining_modes.py`
portable binary üzerinde yalnız her bir gerçek `vegai_pretrain train`
child-process wall-clock'ını raporlar: compiler, CI setup, fixture construction,
inspect ve harness süresi ölçüye girmez. Workflow summary; portable/native
profil oranını, `legacy → positional → fused hybrid` temel karşılaştırmasını,
fused/reference hybrid checkpoint byte eşitliği + hız oranını ve ayrı
16-token-record controlled `hybrid → hierarchical → coherent` topical-cohesion
diagnostic'ini yazar. Coherent guard yalnız hybrid same-cluster Top-1 sonucunun
gerilememesidir; bu HDC diagnostic'i doğal-dil kalite iddiası değildir.

`tools/compare_pretraining_revisions.py`, CI'da sabit `e34fd22` baseline
binary'sini aynı portable Make profile ile build eder ve short-record ile
dense repeated-center fixture'larında aday binary'yle gerçek `train` child
process timing'i yapar. Warm-up/alternating order uygulanır; her binary'nin
self-determinism'i zorunludur ve varsayılan contract baseline/candidate
checkpoint byte equality'dir. v5/v6 → v7 sampler migration'ı gibi review
edilmiş bir algorithm lineage değişikliği yalnız açık
`--checkpoint-contract versioned-algorithm-transition` ile kabul edilir ve
candidate gerçek baseline checkpoint'ini byte-tam resume etmelidir; o
cross-lineage runtime ratio hot-path semantic-equivalence ölçümü veya genel
corpus throughput vaadi değildir.

`tools/compare_pretraining_threads.py`, ayrıca uzun-record fixture'da
`OMP_NUM_THREADS=1` ile 2 arasında gerçek train timing'i ölçer ve checkpoint
byte equality'yi zorunlu kılar. Hız oranı kayıt altına alınır, ama hiçbir
minimum speedup veya lineer core-scaling kabul kriteri değildir. Bu fixture'lar
yalnız pipeline doğrulaması ve sınırlı retrieval diagnostic'idir; semantic
kalite, doğal dil yeteneği veya production throughput benchmarkı değildir.

## Olasılık modeli ve sayısal marjlar

Rastgele bağımsız binary hiper-vektörler için:

```text
H(A,B) ~ Binomial(D, 1/2), D = 16,384
E[H]   = 8,192
σ_H    = sqrt(D/4) = 64
```

Hoeffding/Chernoff sınırı:

```text
P(|H − D/2| >= t) <= 2 exp(-2t² / D)
```

Similarity için `|sim| >= ε`, `t = εD/2` olduğundan:

```text
P(|sim| >= ε) <= 2 exp(-ε²D/2)
```

Örneğin `ε=0.10` için üst sınır yaklaşık `5.3×10^-36`dır. Bu, ideal bağımsız
rastgele model altındaki teorik üst sınırdır; deterministik PRNG'nin tüm
seed/derleyici/işlemci kombinasyonlarına dair kriptografik kanıt değildir.

### Demo 1: 3-item frame + %25 exact noise

Üçlü majority bundle'da doğru bound pair'in her bitinin korunma olasılığı
`3/4`'tür. Sonra bağımsız olarak %25 flip uygularsak:

```text
P(correct after noise) = 0.75×0.75 + 0.25×0.25 = 0.625
Expected similarity    = 2×0.625 − 1 = 0.25
Expected distance      = 6,144
```

Rastgele sözlük vektörü 8.192±64 uzaklıktadır. Hedef ile rastgele merkezler
arasındaki yaklaşık 2.048-bit fark, ideal modelde 32σ düzeyindedir. Bu nedenle
512 sembollü demo sözlüğünde robust retrieval beklenir. Native ve Python
testleri belirlenmiş seed'lerde beklenen ID'yi kontrol eder.

### Demo 2: 3-aligned-pair analogy

`Mapping = majority(Source_i XOR Target_i)` üç item içerir. Sorgu source
vektörü ile unbind edildiğinde hedef pair bitinin beklenen doğruluğu `3/4`,
beklenen similarity'si `0.5` olur. Cleanup marjı frame testinden daha yüksektir.

### Demo 3: 4-rule causal bank

Bir hedef rule + üç crosstalk rule için, deterministic random tie-breaker
modelinde doğru bit olasılığı `11/16 = 0.6875`; beklenen similarity `0.375`'tir.
Her adımda cleanup yapılması bu gürültünün sonraki adıma taşınmasını engeller.

Bu rakamlar **kapasite garantisi değildir**. Bundle item sayısı arttıkça hedef
sinyali zayıflar; sözlük büyüdükçe rastgele en yakın komşu riski artar. Üretim
kullanımında kendi sözlük/bundle/gürültü dağılımınızla ROC, top-k accuracy ve
latency benchmarkı ölçün.

## Negatif testler ve fault containment

| Risk | Davranış |
|---|---|
| `max_symbols <= 0` | `MEM_ERR_INVALID_ARGUMENT` |
| allocation failure | `MEM_ERR_ALLOCATION`, kısmi alanlar serbest bırakılır |
| duplicate ID | `MEM_ERR_DUPLICATE_ID`, mevcut entry korunur |
| full memory | `MEM_ERR_FULL`, yeni entry eklenmez |
| missing ID | `MEM_ERR_NOT_FOUND`, output zero + `found=.false.` |
| empty cleanup | `MEM_ERR_EMPTY` |
| boş/uyumsuz reasoning dizileri | `REASON_ERR_INVALID_SHAPE` |
| `int16` accumulator overflow | `HV_ERR_ACCUMULATOR_OVERFLOW`, hiçbir count mutasyona uğramaz |
| invalid noise count/percent | `HV_ERR_INVALID_ARGUMENT`, output input'a eşit kalır |
| LSH bucket boş | `MEM_ERR_NO_LSH_CANDIDATE` yalnız approximate API'de |
| continuous shard SHA/preflight uyuşmazlığı | controller child CLI çağrısından önce shard'ı reddeder; committed state değişmez |
| continuous manifest committed prefix rewrite | controller reddeder; önceki generation replay edilmez |
| pinned immutable manifest byte replacement | `--expect-manifest-sha256` child/shard işleminden önce reddeder |
| production dataset shard/path/ledger replacement | package SHA/content-address contract ve runner preflight'i reddeder |
| production train/validation raw-record overlap | exact SQLite leakage gate child training başlamadan reddeder |
| production `max-vocab` exact unique train token sayısından küçük | runner training başlamadan capacity hatası verir |
| candidate checkpoint sonrası crash | state journal eski generation'da kalır; orphan candidate durable commit sayılmaz ve normal restart private artifact'ı temizler |
| committed journal coverage metric veya historical checkpoint byte mutation | `continuous_pretrain_status.py` ve controller contract/digest mismatch ile fail-closed olur; resume-safe rapor üretmez |
| health gate sırasında controller lock | `--require-unlocked` fail-closed olur; status aracı lock'u hiçbir zaman silmez |

## Reproducibility kaydı

Bir sonuç raporlanırken en az şu bilgiler saklanmalıdır:

- compiler ve version (`gfortran --version`);
- Fortran 2023 conformance flag'i (`-std=f2023` veya compiler eşdeğeri);
- `FFLAGS`, `OMP_NUM_THREADS`, CPU/OS;
- symbol ID/seed seti;
- bundle tie seedleri;
- rule/frame cardinalitesi;
- cleanup modunun `exact_search` değeri;
- Hamming distance, similarity, sözlük büyüklüğü;
- çalışma zamanı için warm-up ve örnek sayısı.

Bu metadata olmadan yalnız “mikrosaniye” değeri veya “%X accuracy” değeri
tekrar üretilebilir performans/doğruluk kanıtı değildir.

## Sürüm kabul checklist'i

Bir değişiklik merge edilmeden önce:

- [ ] `python3 tools/verify_reference.py`
- [ ] `make debug`
- [ ] `make run` (tüm executable assertion'ları PASS)
- [ ] `cmake` + `ctest` checked build
- [ ] yeni public API için normal ve hata yolu testi
- [ ] kaynakta signed-overflow'a dayanan yeni integer aritmetiği yok
- [ ] Fortran derlemesi `-std=f2023` (veya compiler eşdeğeri) ile geçti
- [ ] exact ve approximate cleanup sözleşmeleri karıştırılmadı
- [ ] Resonator sonucunda heuristic network residual ile certified final residual ayrıldı
- [ ] pretraining corpus manifesti, lisans/PII/retention değerlendirmesi ve digest'i kaydedildi
- [ ] tokenizer/overlong-record telemetry'si incelendi; strict veya kabul politikası açıklandı
- [ ] checkpoint save/load + resume sonucu hedef compiler/ABI üzerinde doğrulandı
- [ ] continual iş varsa immutable SHA-256 manifest, `min_frequency=1`, state journal, restart/fault senaryosu ve `continuous_pretrain_status.py --verify-shards --require-complete --require-unlocked` health gate'i doğrulandı
- [ ] production package kullanılıyorsa exact dedupe, train/validation leakage audit, content-addressed ledger pin'i ve exact `max-vocab` capacity preflight geçti
- [ ] pretraining semantic kalite iddiası held-out görev ve baseline ile ölçüldü
- [ ] `--max-vocab` / checkpoint path için resource plan ve gerçek RAM/storage headroom incelendi
- [ ] multi-core iddiasında serial/parallel checkpoint byte equality doğrulandı
- [ ] GPU/NPU adapter inventory ile actual compute dispatch birbirinden ayrıldı; provider yoksa accelerator iddiası yapılmadı
- [ ] benchmark iddiası hedef platformda ölçüm çıktısıyla desteklendi
