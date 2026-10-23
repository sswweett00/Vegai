# Üretim veri foundry'si ve denetlenebilir continual HDC pretraining

Bu belge, Vegai'nin mevcut **deterministik HDC/VSA** pretraining yolunu büyük
ve yönetişim altında tutulan corpus'larda işletmek içindir. İki yeni kontrol
aracı vardır:

- `tools/production_dataset.py`: strict corpus'u exact-deduplicate eder,
  içerik-adresli immutable shard'lara böler ve deterministik held-out split
  yayınlar.
- `tools/production_pretrain.py`: paketlenmiş veriyi, binary'yi ve immutable
  eğitim politikasını kimliklendirir; exact leakage audit/resource preflight
  yapar ve gerçek `continuous_pretrain.py` exactly-once controller'ını çağırır.

> ## Mimari sınır — önemli
>
> Bu, Transformer/LLM/LMM ya da multimodal gradient-training runtime'ı değildir.
> Vegai, packed hypervector/prototype güncellemeleri yapan bounded HDC/VSA
> motorudur. Bu katman veri yönetişimi, reproducibility, crash recovery ve
> operasyonel ölçek sağlar; tensor/autodiff, optimizer, attention, tokenizer
> eğitimi, model parallelism veya CPU+GPU+NPU ortak neural training eklemez.
> Dolayısıyla "devasa LLM/LMM eğitimi" iddiası yapılmaz. Böyle bir hedef, bu
> repodan ayrı bir model/runtime/distributed-systems mimarisi gerektirir.

## Sağlanan üretim sözleşmesi

### 1. Fail-closed veri foundry

`--input-format txt` (varsayılan) yolunda **uncompressed** strict corpus için
builder kaynak corpus'u indirmez, normalize etmez ve UTF-8'i sessizce ASCII'ye
çevirmez. Her fiziksel kaydın şu Vegai tokenizer sözleşmesine uymasını ister:

- LF sonlandırmalı kayıt; CRLF kabul edilmez;
- ASCII byte; token karakteri `A-Z a-z 0-9 _ '`;
- en çok 8.192 payload byte/kayıt, 48 byte/token ve 1.024 token/kayıt;
- tokenizer tarafından dropped/truncated token yok.

JSONL/JSON/Parquet/Arrow için `--input-format` açıkça seçilebilir. `.txt.gz`
de aynı adapter yolundan decompress + revalidate edilir. Bu yol önce
`vegai-training-record/v1` schema-valid record stream'ini okur, sonra yalnız
`record.text` alanını aynı strict ASCII/token contract'iyle private canonical
TXT snapshot'a projection eder. `label`, `split`, `source` ve `metadata` native
HDC text'e eklenmez. `dataset.json.input_adapter`, raw structured source'un
SHA-256/byte count'unu ve canonical TXT snapshot'ın SHA-256/byte count'unu
birlikte bağlar; runner bu bağ varsa doğrular. Ayrıntılı rich-data schema,
compression ve conversion sınırları için
[TRAINING_DATA_FORMAT.md](TRAINING_DATA_FORMAT.md) belgesine bakın.

Unicode, PII temizleme, lisans filtresi, dil/kalite filtresi veya dedupe öncesi
normalizasyon gerekiyorsa bunlar **ayrı, versioned ve izlenebilir** bir upstream
pipeline'da yapılmalıdır. Kullanılan komut/sürüm `--preprocessing` ve source
registry retrieval/version olayı `--retrieved-at` ile immutable provenance'a
yazılır. Tool veriyi kendiliğinden dönüştürmediği için "Unicode
öğrenildi" gibi yanlış bir iddia üretmez.

Her retained raw-record identity'si `SHA-256(record-without-LF)` ile SQLite
üzerinde disk-backed olarak tutulur. Bu nedenle corpus boyutu RAM'e sığmak
zorunda değildir. `--deduplicate exact` varsayılandır; production runner yalnız
bu modu kabul eder. Boş veya punctuation-only kayıtlar modele token
vermediğinden açıkça sayılıp filtrelenir. Continual policy `min_frequency=1`
olduğu için builder ayrıca train split'in ASCII-lowercase exact unique-token
sayısını aynı disk-backed index'te hesaplar. `dataset.json` bunu
`unique_tokens_min_frequency_1` olarak bağlar; runner `--max-vocab` bu sayıdan
küçükse gerçek training başlamadan fail-closed olur.

Split seçimi kayıt kimliğinden türetilir:

```text
big_endian_u64(SHA256(record)[:8]) mod 10000 < validation_basis_points
```

Aynı kaydın kaynak sırası değişse bile split'i değişmez. Exact duplicate,
split'ten önce tek kez retained edilir; bir raw identity train ve validation'a
aynı anda yazılamaz. Builder önce private source snapshot alır, snapshot digest'ini
kayda geçirir ve publish öncesi caller source'unun değişmediğini tekrar kontrol
eder; build sırasında source'u mutable bırakmayın. Builder yalnız tamamlanmış
shard, ledger ve `dataset.json` yazıldıktan sonra staging dizinini atomik rename
ile yayınlar. Hata/interrupt halinde hedef dataset dizini yayınlanmaz.

### 2. İçerik adresli immutable shard'lar

Her split `ledger.json` içinde aşağıdaki bilgileri taşır:

- shard payload SHA-256 ve byte count;
- strict ingest preflight counters;
- relative, parent altında kalması zorunlu path;
- declared source/license/retrieval/preprocessing provenance.

`dataset.json` hem train hem validation ledger digest'ini, source digest'ini,
record/token/shard sayılarını, train unique-token capacity requirement'ını ve
split/dedup algoritmasını bağlar. Run başlamadan
önce production controller her shard'ın gerçek byte'ını tekrar hash'ler ve
preflight'ını tekrar tarar. Değiştirilmiş payload, resource planner veya
pretraining executable'a ulaşmadan reddedilir. Wrapper, gerçek controller'a
`--expect-manifest-sha256` geçirir; pinlenmiş training ledger değiştirilirse
child pretraining başlamadan fail-closed olur.

### 3. Held-out leakage gate

Validation varsa varsayılan olarak `production_pretrain.py`, private SQLite
index ile **tüm train ve validation raw-record SHA-256 kimliklerini** karşılaştırır.
Bir tek eşleşme dahi fail-closed hatadır. Bu filename, directory veya shard
SHA'sına dayanan zayıf bir kontrol değildir; record seviyesinde exact kontroldür.

`--skip-exact-leakage-audit` yalnız bilinçli bir operatör kaçışı içindir ve
result manifest'e açıkça `skipped-by-operator` diye kaydedilir. Üretim/evaluation
sonucu için kullanılmamalıdır. Bu gate yalnız exact same-record leakage'i
engeller; semantic near-duplicate, source-level overlap veya benchmark leakage
için ek bir upstream policy gerekir.

### 4. Exactly-once eğitim ve run provenance

Production wrapper, zaten doğrulanmış `continuous_pretrain.py` mekanizmasını
duplikasyon yapmadan kullanır:

1. source shard SHA/preflight doğrulanır ve private snapshot alınır;
2. gerçek `vegai_pretrain train` veya `continue`, yalnız candidate checkpoint
   üzerinde çalışır;
3. candidate fsync + immutable generation adıyla atomik publish edilir;
4. **yalnız atomik `state.json` replacement** shard commit marker'ıdır.

Ara kesinti candidate/final orphan bırakabilir fakat state journal'a girmediği
için sonraki invocation ilgili shard'ı önceki committed modelden yeniden dener;
double-apply yapmaz. `run.json`, dataset/ledger SHA'ları, binary SHA-256, resolved
policy, selected state directory, resource plan, donanım inventory ve repository
revision taşır. `result.json` ya da `failure.json` da atomik yazılır; failure
anında `result.json` da latest-attempt `status=failed` ile overwrite edilir,
böylece eski başarılı resume sonucu yeni hata sanılmaz.

`--resume-run`, dataset/binary/policy/state-directory fingerprint'i bit-bit
aynı değilse reddeder. `--max-shards` ile immutable package içindeki kontrollü
batch processing yapılabilir. Bu wrapper bilerek `--follow` sunmaz: immutable
`dataset.json` ledger digest'i altında sonradan append kabul etmek provenance
zincirini kırar. Append-only live feed için mevcut
`continuous_pretrain.py --follow` yolunu, ayrı/versioned dataset ve run
provenance policy'siyle kullanın.

## Hızlı örnek: source-controlled sentetik dataset

`examples/corpus/mini_turkish_ascii_pretrain.txt` gerçek kişi/veri içermeyen,
ASCII transliterasyonlu MIT fixture'ıdır. Repoda onun foundry output'u da
source-controlled olarak bulunur:

```text
examples/production_dataset/mini_turkish_ascii_hdc/
  dataset.json
  train/ledger.json + 4 immutable shard
  validation/ledger.json + 1 immutable shard
```

Bu örnek **128 kayıt / 1.024 token** içerir. Deterministik %10 held-out kuralı
ile train **114 kayıt / 912 token / 64 exact unique token**, validation
**14 kayıt / 112 token** üretir. Bu sayılar semantic quality, Türkçe kapsamı veya gerçek-world model
başarısı değildir; data/control-plane integration fixture'ıdır.

Aynı paketi yeniden üretmek için (mevcut hedefi silmeyin; builder overwrite
etmeyi bilerek reddeder):

```bash
python3 tools/production_dataset.py \
  --input examples/corpus/mini_turkish_ascii_pretrain.txt \
  --output-dir /tmp/mini_turkish_ascii_hdc \
  --source 'Vegai repository-authored synthetic fixture' \
  --license MIT --retrieved-at 'source-controlled fixture' \
  --preprocessing 'ASCII-safe Turkish transliteration; deterministic cluster rotations' \
  --validation-basis-points 1000 \
  --max-shard-tokens 256 --max-shard-bytes 65536
```

CI `tools/verify_production_pretraining.py` daha küçük bir generated fixture
ile iki bağımsız foundry run'ın byte-identical output'unu, exact dedupe'i,
tampered shard rejection'ını, intentional train/validation overlap rejection'ını,
gerçek compiled HDC binary ile checkpoint generation'larını, read-only completed
state/payload health audit'ini ve idempotent resume'ı doğrular.

## Kendi production dataset'inizi yayınlama

Yalnız local, upstream'te zaten lisans/PII/kalite yönetişimi uygulanmış bir
ASCII-LF corpus verin. Output için git-ignored `/production-datasets/` gibi bir
konum kullanın:

```bash
python3 tools/production_dataset.py \
  --input /governed/input/corpus.txt \
  --output-dir production-datasets/news-v2026-10 \
  --source 'organization dataset registry: news-v2026-10' \
  --license 'approved-internal-license-id' \
  --retrieved-at 'registry-event-2026-10-23T00:00:00Z' \
  --preprocessing 'governed-cleaner 4.2.1; policy-2026-10; ASCII export v1' \
  --validation-basis-points 100 \
  --max-shard-bytes $((64 * 1024 * 1024)) \
  --max-shard-tokens 1000000
```

`--validation-basis-points 100` = deterministic %1. Küçük bir corpus'ta sıfır
validation kaydı olabileceğinden builder bunu fail-closed reddeder; daha büyük
oran seçin veya evaluation policy'nizi açıkça yeniden tasarlayın. Shard limitleri
record'u bölmez; tek bir valid record limitten büyükse publish edilmez.

Dataset'ler immutable kabul edilir. Yeni veri için eski shard'ı değiştirmeyin;
yeni dataset version ya da append-only continual ledger kullanın. Shard'ın
manifest içindeki SHA'sı değiştirilmeden byte'ı değişirse run başlamaz. Builder
staging dizinini output parent altında ve exact record/token SQLite index'ini bu
staging alanında tutar; production output volume'unu hem final shard payload'ı
hem de geçici index/snapshot alanı için yeterli ve güvenilir local storage ile
planlayın.

## Gerçek pretraining run

Önce portable Fortran 2023 binary'yi derleyin, sonra paketin üretim gate'lerini
ve gerçek continuous controller'ı çalıştırın:

```bash
cmake -S . -B build -DVEGAI_CPU_TARGET=portable -DVEGAI_ENABLE_OPENMP=ON
cmake --build build --parallel

python3 tools/production_pretrain.py \
  --pretrain-bin ./build/vegai_pretrain \
  --dataset production-datasets/news-v2026-10/dataset.json \
  --run-dir production-runs/news-v2026-10-hdc \
  --max-vocab 20000 --min-frequency 1 \
  --window 4 --position-shift 73 --context-mode hierarchical --epochs 1 \
  --initial-plasticity 25 --minimum-plasticity 1 --decay-events 32 \
  --omp-threads 8
```

Başlatmadan önce yalnız validation/data/resource kontrollerini görmek için:

```bash
python3 tools/production_pretrain.py \
  --pretrain-bin ./build/vegai_pretrain \
  --dataset production-datasets/news-v2026-10/dataset.json \
  --run-dir production-runs/news-v2026-10-hdc \
  --max-vocab 20000 --preflight-only
```

Bir commit'ten sonra normal resume tamamen idempotenttir:

```bash
python3 tools/production_pretrain.py \
  --pretrain-bin ./build/vegai_pretrain \
  --dataset production-datasets/news-v2026-10/dataset.json \
  --run-dir production-runs/news-v2026-10-hdc --resume-run \
  --max-vocab 20000 --min-frequency 1 --context-mode hierarchical --epochs 1
```

İlk run ile aynı immutable policy verilmelidir. `--resume-run` yeni data,
yeni binary veya başka state directory ile eski run'a yanlışlıkla bağlanamaz.
Harici state konumu gerekiyorsa ilk çağrıda `--state-dir PATH` kullanın; daha
sonra aynı resolved path kullanılmalıdır. SIGKILL/power-loss sonrası wrapper
veya inner controller stale lock bırakmışsa, owner'ın gerçekten çalışmadığını
inceledikten sonra yalnız explicit `--recover-locks` ile resume edin. Bu flag
outer production lock'u ve `continuous_pretrain.py --recover-lock` yolunu
birlikte açar; canlı bir writer varken kullanmak güvenli değildir.

### Read-only post-run health gate

Release, evaluation, backup veya retention işleminden önce inner exactly-once
journal'ı ayrıca read-only doğrulayın. `TRAIN_LEDGER_SHA256`, immutable
`dataset.json` içindeki `train.ledger_sha256` değeri olmalıdır:

```bash
python3 tools/continuous_pretrain_status.py \
  --manifest production-datasets/news-v2026-10/train/ledger.json \
  --state-dir production-runs/news-v2026-10-hdc/state \
  --expect-manifest-sha256 "$TRAIN_LEDGER_SHA256" \
  --verify-shards --require-complete --require-unlocked --format json
```

Araç tüm committed generation checkpoint digest'lerini, manifest prefix'ini ve
isteğe bağlı olarak her source shard'ın SHA/preflight'ını denetler; checkpoint,
state, lock veya result manifest'i yazmaz/değiştirmez. Lock görürse yalnız
raporlar; stale-lock silme kararı operatördedir. Bu gate transactional HDC
state'in resume-safe olduğunu denetler, held-out semantic kalite veya neural
LLM/LMM sağlığı ölçümü değildir. Ayrıntılı flags/race davranışı
[CONTINUOUS_PRETRAINING.md](CONTINUOUS_PRETRAINING.md)'dedir.

## Kaynaklar, dual RAM, CPU/GPU/NPU ve ölçüm sınırı

Wrapper başlangıçtan önce `training_resource_plan.py` üzerinden model/checkpoint
alanı, RAM reserve ve OpenMP yerleşim önerisini manifest'e kaydeder. Bu mevcut
Vegai'nin bounded RAM/corpus-streaming sözleşmesidir; RAM channel topology
software ile güvenilir saptanamadığından dual-channel/NUMA yalnız OS/firmware
ve matched DIMM'lerle ayrıca doğrulanmalıdır.

`--omp-threads N`, gerçek child pretraining invocation'a `OMP_NUM_THREADS=N`
(ve unset ise `OMP_PROC_BIND=close`, `OMP_PLACES=cores`) verir. HDC context
construction paralel olabilir; source-order prototype commit checkpoint
belirlenimini korur. Hızlanma workload ve hosta bağlıdır, garanti değildir.

Bu production control layer GPU/iGPU/NPU training dispatch eklemez. Hardware
inventory veya OpenVINO device görünmesi, ortak CPU/GPU/NPU compute kanıtı
değildir. Intel/OpenVINO context-provider'ın ayrı, verified immutable-context
sınırları için [PRETRAINING.md](PRETRAINING.md) ve
[HARDWARE_COMPATIBILITY.md](HARDWARE_COMPATIBILITY.md) geçerlidir.

`result.json` içindeki `elapsed_train_controller_seconds`, yalnız bu wrapper'ın
**continuous training controller** child çağrısının wall time'ıdır; derleme, CI
setup, dataset construction, exact leakage audit ve resource preflight buna
dahil değildir. Ancak birden çok shard process başlatma/transaction overhead'ini
içerdiği için saf model throughput metriği değildir. Throughput karşılaştırması
sunulacaksa derleme/CI/fixture/harness hariç tutularak yalnız aynı
`vegai_pretrain train` komutlarının ölçülmüş süreleri kullanılmalıdır; mevcut
kontrollü yöntem [PRETRAINING_MODE_COMPARISON.md](PRETRAINING_MODE_COMPARISON.md)
ve `tools/measure_example_pretraining.py` içindedir.

## Bilinen kapsam dışıları

Bu katman şunları yapmaz ve yaptığını iddia etmez:

- web crawling, licensing/PII enforcement veya semantic near-duplicate
  detection;
- Unicode/BPE/SentencePiece tokenizer, multimodal image/audio packing;
- neural loss/evaluation, held-out perplexity, gradient checkpointing,
  mixed precision, optimizer, data/model/pipeline parallelism;
- distributed scheduler, remote object store, fleet telemetry veya automatic
  checkpoint retention policy;
- GPU/NPU acceleration ya da CPU+GPU+NPU birlikte training kanıtı.

Bu ihtiyaçlar gerçek LLM/LMM hedefi için ayrı threat model, dataset governance,
neural runtime, distributed collective ve target-hardware validation planı ile
tasarlanmalıdır.
