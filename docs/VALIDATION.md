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
| `make check` | bağımsız reference property + demo | `Python reference verification: PASS` |
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
| `ctest --test-dir build` | CMake entegrasyonu | `vegai_fortran_properties`, donanım/OpenVINO contract testleri, `vegai_example_pretraining`, `vegai_dialogue_retrieval`, `vegai_continuous_pretraining`, `vegai_production_pretraining`, `vegai_example_production_dataset_preflight` PASS |

CI bunların checked CMake/OpenMP varyantını ve ayrıca optimize Make hedefini
çalıştırır.

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
- binary checkpoint save/load, rebuilt hash index ve payload plane round-trip;
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
prompt/malformed TSV failure'ini ve bounded REPL yolunu kontrol eder. Native
Fortran property testi ayni katmanda exact retrieval tie/margin, duplicate
embedding, fixed capacity, strict ASCII ingest ve session ring reset
invariantlarini kapsar. Bu, LLM kalite degerlendirmesi degil, veri-kaynakli
retrieval contract'idir.

CI önce `tools/hardware_compatibility.py` ile CPU/OS/mimari/RAM/GPU/NPU
inventory metadata'sını, sonra `tools/training_resource_plan.py` ile bounded
RAM/checkpoint-storage/OpenMP placement planını kaydeder. Generic plan GPU/NPU
compute kullanmadığını açıkça raporlar ve provider/device execution bilgisi
olmaksızın accelerator talebini fail-closed reddeder. Ayrı OpenVINO bridge
adımı optional runtime'ı kurar; önce explicit CPU fallback'ın native fused
kernel'e worker başlatmadan bypass ettiğini, ardından strict resident CPU
OpenVINO context graph ile Fortran CPU reference checkpoint byte equality'yi
test eder. Worker verifier ayrıca dynamic-batch compiled-model cache ve
neutral tie/complement padding exactness'ini denetler. Ardından portable
varsayılan Make binary ile aynı host için opt-in `native` binary'yi
`tools/compare_pretraining_build_profiles.py` üzerinden
ölçer. İki profile'ın checkpoint byte eşitliği zorunludur; süre oranı yalnız o
host için anlamlıdır.

Native testler fused raw-word/permuted/pair accumulator plane bit-eşitliğini,
fused/reference hybrid ile hierarchical replay eşitliğini, repeated-center
serial/OpenMP context-build eşitliğini ve hierarchical checkpoint round-trip'ini
denetler. `tools/compare_pretraining_modes.py` portable binary üzerinde yalnız
her bir gerçek `vegai_pretrain train` child-process wall-clock'ını raporlar:
compiler, CI setup, fixture construction, inspect ve harness süresi ölçüye
girmez. Workflow summary; portable/native profil oranını,
`legacy → positional → fused hybrid` temel karşılaştırmasını, fused/reference
hybrid checkpoint byte eşitliği + hız oranını ve ayrı 16-token-record controlled
`hybrid → hierarchical` topical-cohesion diagnostic'ini yazar.

`tools/compare_pretraining_revisions.py`, CI'da sabit `e34fd22` baseline
binary'sini aynı portable Make profile ile build eder ve short-record ile
dense repeated-center fixture'larında aday binary'yle gerçek `train` child
process timing'i yapar. Warm-up/alternating order uygulanır; her binary'nin
self-determinism'i ve baseline/candidate checkpoint byte equality zorunludur.
Bu, hot-path mimari değişikliğinin ölçümüdür; genel corpus throughput vaadi
değildir.

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
