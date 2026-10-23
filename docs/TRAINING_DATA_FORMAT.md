# Vegai eğitim verisi formatı ve streaming I/O sözleşmesi

Bu belge, Vegai'nin mevcut Fortran 2023 HDC/VSA eğitim mimarisi için eklenen
çok-formatlı veri sınırını tanımlar. Amaç JSONL/JSON/TXT/Parquet/Arrow'dan
gelen **yapısal kayıtları** doğrulamak, kalite telemetrisi üretmek ve ancak
bilinçli bir son adımda native HDC eğitiminin tüketebileceği strict corpus'a
dönüştürmektir.

> Bu katman Transformer/LLM/LMM tokenizer veya neural data loader değildir.
> Native `vegai_pretrain` yalnız ASCII, LF-delimited, fiziksel satır başına tek
> record tüketir. Bu katman Unicode'u sessizce transliterate etmez, label'ı
> training text'e eklemez ve HDC'nin dağıtımsal prototype öğrenmesini generative
> language-model eğitimi gibi göstermez.

## 1. Kısa mimari özeti ve veri akışı

| Katman | Dosyalar | Girdi/çıktı | Sorumluluk |
|---|---|---|---|
| Native eğitim | `app/pretrain.f90`, `src/hdc_pretraining.f90` | strict ASCII/LF TXT → `.vgai` | iki-pass vocabulary, streaming context/Hebbian update, bounded RAM, checkpoint/resume |
| Native ingest contract | `src/text_tokenizer.f90`, `tools/corpus_manifest.py` | bir fiziksel satır = bir context boundary | 8.192 byte/record, 1.024 token/record, 48 byte/token, ASCII tokenization; silent truncation yok |
| Continual/production control | `tools/continuous_pretrain.py`, `tools/production_dataset.py`, `tools/production_pretrain.py` | immutable TXT shard/ledger | exact SHA-256 dedupe, deterministic held-out split, resource/leakage gate, exactly-once checkpoint transaction |
| **Yeni rich-data boundary** | `tools/training_data_io.py`, `tools/training_dataset.py` | JSONL/JSON/TXT/Parquet/Arrow ↔ logical record | schema validation, lazy streaming, batch I/O, conversion, quality telemetry, strict HDC text export |

Akış aşağıdaki gibidir:

```text
JSONL / JSON / TXT / Parquet / Arrow
        │  (streaming schema validation)
        ▼
vegai-training-record/v1 logical records
        │  (explicit cleaning + exact text dedupe + deterministic split audit)
        ▼
rich prepared dataset  ──► JSONL / JSON / Parquet / Arrow
        │
        └── write_hdc_corpus / --hdc-strict
                   │ (text only; strict ASCII/LF/token limits)
                   ▼
           production_dataset.py → immutable TXT shards → vegai_pretrain
```

`label`, `source`, `split` ve `metadata` rich kayıt formatında korunur. Native
HDC text export yalnız `text` alanını yazar. Bilinen topic label'larını token
olarak enjekte etmek gerekiyorsa bu, otomatik yapılmaz; ayrı, versioned,
provenance'a yazılmış **explicit weak-supervision** upstream adımı olmalıdır.

## 2. Logical veri şeması

Şema sürümü: `vegai-training-record/v1`

| Alan | Tip | Zorunlu | Açıklama |
|---|---|---:|---|
| `schema_version` | string | hayır (varsayılan v1) | Verilirse tam olarak `vegai-training-record/v1` olmalıdır. |
| `id` | string/null | hayır | Upstream record kimliği; boş string kabul edilmez. HDC dedupe identity'si değildir. |
| `text` | string | **evet** | Tek logical record. `CR`/`LF` içeremez. Native export sırasında ayrıca ASCII/token sınırları zorunludur. |
| `label` | string/null | hayır | Kalite/imbalance/stratification audit etiketi; otomatik HDC token'ı olmaz. |
| `split` | `train`/`validation`/`test`/null | hayır | Önceden atanmış audit bilgisi. `prepare` varsayılanı kendi SHA-256 split'ini üretir. |
| `source` | string/null | hayır | Upstream provenance etiketi/URI kimliği. |
| `metadata` | JSON object/null | hayır | Canonical JSON olarak saklanır; azami 64 KiB. Binary veya serileşemeyen Python object kabul edilmez. |

Örnek JSONL kaydı:

```json
{"schema_version":"vegai-training-record/v1","id":"news-42","text":"ekonomi pazar fiyat veri","label":"ekonomi","source":"registry/release-7","metadata":{"language":"tr-Latn","license":"approved"}}
```

Makine tarafından yayınlanan JSON Schema:

```bash
python3 tools/training_dataset.py schema --kind json > training-record.schema.json
```

Parquet ve Arrow IPC için eşdeğer, **tam alan sırası ve tipleri sabit** Arrow
schema kullanılır:

```text
schema_version: string not null
id:             string nullable
text:           string not null
label:          string nullable
split:          string nullable
source:         string nullable
metadata_json:  string nullable     # canonical JSON object
```

Bu sıkı şema, column order'a dayalı yanlış decode'u veya fark edilmeden eksik
alanı reddeder. JSON/JSONL unknown field'ları da fail-closed reddedilir.

### TXT'nin bilinçli dar sözleşmesi

TXT `vegai-training-text/v1` olarak yalnız şunu taşır:

```text
one training text record per UTF-8 LF line
```

Bu nedenle rich `id`, `label`, `split`, `source`, `metadata` alanlarını TXT'ye
yazmak varsayılan olarak hata verir. Bilinçli projection için CLI'da
`--text-only`, native corpus için ise `--hdc-strict` kullanılır. Bu tasarım
metadata/label kaybını sessizce gizlemez ve Fortran'ın line=context-boundary
sözleşmesiyle uyumludur.

## 3. Formatlar, streaming ve hata sözleşmesi

| Format | Okuma/yazma yolu | Şema | Sıkıştırma | Memory davranışı |
|---|---|---|---|---|
| JSONL | satır satır UTF-8 `json.loads` | `TrainingRecord.from_mapping` | deterministic-header `gzip` | O(tek satır) |
| JSON | yalnız top-level array; incremental `JSONDecoder.raw_decode` | aynı | deterministic-header `gzip` | O(tek array elemanı + decoder suffix) |
| TXT | binary satır iterator | yalnız `text` | deterministic-header `gzip` | O(tek satır) |
| Parquet | `ParquetFile.iter_batches` / `ParquetWriter` | strict Arrow schema | `zstd`, `snappy`, `gzip`, `brotli`, `lz4` | O(configured record batch) |
| Arrow IPC | file/stream record-batch iterator | strict Arrow schema | `zstd`, `lz4` | O(record batch; memory-map/file stream) |

`pyarrow` yalnız Parquet/Arrow için lazy import edilir; JSONL/JSON/TXT yolu
stdlib ile çalışır. Reader varsayılanı `error_policy="raise"`dır. Malformed
JSON line, UTF-8, unknown field, wrong Arrow schema veya missing column eğitim
öncesi durur. Bilinçli kayıplı audit için CLI `--drop-invalid` seçeneği vardır;
çıktıdaki issue count ve sınırlı örnekler manifest/report'a yazılır.

Yazıcılar temporary file + `fsync` + atomic replace kullanır. Böylece yarım
JSON closing bracket'i ya da yarım Parquet footer'ı hedef path'te publish
edilmez.

## 4. Kalite, temizlik, dedupe ve split stratejisi

### Exact dedupe

`ExactTextDeduplicator`, `SHA-256(UTF-8 text)` kimliklerini SQLite'da tutar.
Bu seçim, HDC'nin gerçekten öğrendiği alanı hedefler ve büyük corpus için bütün
digest'leri Python RAM'inde tutmaz. Aynı text farklı label ile gelirse duplicate
sayılır ve `label_conflicts` telemetrisi yükselir; `prepare
--reject-label-conflicts` ile fail-closed policy seçilebilir.

Bu exact dedupe'dir; semantic near-duplicate, license/PII veya source-level
overlap çözmez. Bunlar upstream governed pipeline sorumluluğundadır.

### Temizlik

Varsayılan davranış veri değiştirmez. Aşağıdaki transform'lar ancak açık CLI
flag'iyle yapılır ve manifestte sayılır:

- `--strip-outer-whitespace`
- `--drop-empty`
- `--hdc-strict` (ASCII, 8.192-byte record, 48-byte token, 1.024 token limit)

Unicode'u ASCII'ye dönüştürmek, CRLF'yi sessizce düzeltmek veya label'ı text'e
katmak bu tool'un işi değildir. Bunlar reversible/versioned upstream preprocessing
olarak `--preprocessing` provenance alanında kaydedilmelidir.

### Imbalance ve split

`validate`/`prepare`, label count, minimum/maksimum class count, max/min ratio
ve Shannon entropy (bit) raporlar. Resampling yapmaz; class balance değişimi
training semantics değiştirir ve ayrı deney policy'si olmalıdır.

`prepare` split'i source sıra numarasından değil aşağıdaki stable kimlikten
üretir:

```text
bucket = big_endian_u64(SHA256(UTF-8 text)[:8]) mod 10000
bucket < test_basis_points                         → test
bucket < test_basis_points + validation_basis_points → validation
aksi                                                → train
```

Bu, shuffling sonrası aynı text'in aynı split'e gitmesini ve duplicate'in
split leakage üretmemesini sağlar. Bu streaming-friendly hash split **yaklaşık
stratified** davranır; label ile hash arasında correlation varsa tam stratified
split garantisi vermez. Label imbalance raporunu split sonrası inceleyin;
tam stratified regrouping gerekiyorsa disk-backed, ayrı versioned plan kullanın.

## 5. Kurulum

Temel JSONL/JSON/TXT araçları Python standard library kullanır. Parquet/Arrow
ve formatlar arası tam round-trip testleri için yeni, açık bağımlılık:

```bash
# PEP 668 ile global pip'i kilitleyen Linux dağıtımlarında ayrı environment kullanın.
python3 -m venv .venv-dataio
. .venv-dataio/bin/activate                 # Windows: .venv-dataio\Scripts\activate
python -m pip install -r requirements-dataio.txt
python tools/verify_training_data_io.py

# POSIX/GNU Make kısayolu aynı isolated .venv-dataio environment'ını oluşturur.
make data-check
```

Bu `pyarrow>=17,<24` ekler. Portable Fortran runtime'ın, `vegai_pretrain`in
veya CPU/OpenMP training yolunun zorunlu bağımlılığı değildir. CI bu bağımlılığı
ayrı bir adımda kurar ve `tools/verify_training_data_io.py` ile doğrular.

## 6. Kullanım örnekleri

### Validation ve schema audit

```bash
python3 tools/training_dataset.py validate data/train.jsonl \
  --input-format jsonl --hdc-strict

# Malformed satırları açıkça sayıp incelemek için; default fail-closed'tur.
python3 tools/training_dataset.py validate data/train.jsonl \
  --drop-invalid --max-issues 50
```

### Streaming format conversion

```bash
# Rich metadata korunur; Parquet zstd columnar output üretir.
python3 tools/training_dataset.py convert data/train.jsonl data/train.parquet \
  --output-format parquet --output-compression zstd --batch-size 4096

# Rich metadata'yı bilinçli olarak düşürüp strict native HDC corpus üretir.
python3 tools/training_dataset.py convert data/train.parquet data/train-hdc.txt \
  --input-format parquet --hdc-strict

./build/vegai_pretrain train data/train-hdc.txt model.vgai \
  --max-vocab 50000 --min-frequency 2 --context-mode hierarchical --strict
```

### Dedupe + split + quality manifest

```bash
python3 tools/training_dataset.py prepare data/train.jsonl \
  --output-dir prepared/news-v1 --output-format parquet --output-compression zstd \
  --strip-outer-whitespace --drop-empty \
  --validation-basis-points 1000 --test-basis-points 1000 \
  --reject-label-conflicts
```

Bu `prepared/news-v1/{train,validation,test}.parquet` ve `dataset.json`
üretir. JSONL/TXT gibi text formatlarda `--output-compression gzip` seçilebilir.

### Rich source ile mevcut production foundry

`production_dataset.py` artık rich input'u explicit adapter ile kabul eder;
çıktı yine immutable strict TXT shard'dır:

```bash
python3 tools/production_dataset.py \
  --input governed/news.jsonl --input-format jsonl \
  --output-dir production-datasets/news-v1 \
  --source 'registry/news-v1' --license 'approved-license' \
  --retrieved-at 'registry-event-2026-10-23' \
  --preprocessing 'governed cleaner 3.1; ASCII export policy v1' \
  --validation-basis-points 100
```

`dataset.json.input_adapter` raw source SHA/byte count'u ile canonical strict
HDC text snapshot SHA/byte count'unu birlikte bağlar. Label/metadata native
text'e gizlice sızmaz. `production_pretrain.py` bu adapter metadata'sını
validate ederken eski TXT-only v1 dataset'lerini de destekler.

## 7. Performans yaklaşımı ve ölçülebilir hedefler

Bu katman için "yüksek performans" iddiası sabit bir MB/s sayısı değildir;
disk, compression codec, Arrow sürümü, record boyutu, CPU, OS page cache ve
network filesystem sonucu değiştirir. Uygulanan ölçülebilir tasarım hedefleri:

1. JSONL/TXT/JSON valid input'ta **O(batch + en büyük record)** Python memory
   kullanır; tüm corpus list'e yüklenmez.
2. Parquet/Arrow reader/writer configured `--batch-size` kadar record batch ile
   çalışır (default 4.096); Parquet row-group ve Arrow native buffers lazy
   tüketilir.
3. Exact dedupe identity'leri SQLite'dadır; corpus büyüklüğü kadar Python
   `set` allocation yapılmaz.
4. Parallel I/O yalnız bağımsız file listeleri için `iter_records_many` ile
   bounded queue/backpressure kullanır. Cross-file arrival order deterministik
   değildir; source-order training için `workers=1` kullanılır.
5. Atomic output + schema rejection correctness hedefidir; speed için data
   kaybı veya field drop kullanılmaz.

Aynı host ve aynı input üzerinde gerçek reader benchmark'ı:

```bash
python3 tools/benchmark_training_data_io.py data/train.parquet \
  --format parquet --batch-size 4096 --warmup 1 --repetitions 5 \
  --output parquet-reader-benchmark.json
```

Rapor, default bir tam warm-up traversal'ını ölçüm dışı bırakır ve instrumentation
olmayan median `records_per_second` / physical `MiB/s` verir. Python
`tracemalloc` peak'i ayrı, **untimed** bir full traversal'dan gelir; böylece
allocation tracing throughput sayısını bozmaz. Cold-cache sonucu gerekiyorsa
temiz/ayrı host koşulunda `--warmup 0` kullanın ve bunu rapora yazın.
`tracemalloc` Arrow'ın native allocator/RSS'ini içermez; production memory
limit'i için cgroup/RSS metriğini ayrıca kaydedin.

Beklenen **nitel** farklar (garanti değil): Parquet/Arrow, tekrar eden
string/metadata ve column projection workload'larında JSON parse CPU'sunu ve
on-disk byte sayısını azaltabilir; `zstd` I/O'yu düşürür ama CPU harcar;
`snappy` daha düşük compression ile daha hızlı decode tercihidir; JSONL/TXT
sıralı streaming ve debug/provenance için en kolay inspect edilebilir yoldur.
Kabul eşiğini hedef hostta ölçerek yazın; örneğin release manifesti için
"batch=4096, 5 median run, ≥X records/s, ≤Y MiB RSS" gibi somut SLO tanımlayın.
Repository bu sayıları ölçmeden evrensel throughput vaadi vermez.

## 8. Kod entegrasyon haritası

| Dosya | Eklenen/değişen fonksiyon | Entegrasyon etkisi |
|---|---|---|
| `tools/training_data_io.py` | `TrainingRecord`, `iter_records`, `RecordWriter`, `write_records`, `write_hdc_corpus`, `ExactTextDeduplicator`, `iter_records_many` | Tek logical schema, tüm format reader/writer, streaming/batch/atomic/error policy |
| `tools/training_dataset.py` | `schema`, `validate`, `convert`, `prepare` subcommand'ları | Operator-facing schema export, conversion, quality/dedupe/split manifest |
| `tools/production_dataset.py` | `--input-format`, `materialize_structured_input_as_hdc_text` | JSONL/JSON/TXT/Parquet/Arrow source'u strict native TXT shard foundry'ye explicit projection |
| `tools/production_pretrain.py` | optional `input_adapter` validation | Raw rich input provenance ile canonical HDC corpus identity bağının kontrolü |
| `tools/verify_training_data_io.py` | round-trip/failure/integration verifier | JSONL/JSON/TXT/Parquet/Arrow, compression, corrupted record, dedupe/split, HDC export regression |
| `tools/benchmark_training_data_io.py` | reader-only benchmark | Host-specific throughput/memory reporting |
| `requirements-dataio.txt`, `.github/workflows/ci.yml`, `Makefile` | optional dependency + `make data-check` | Parquet/Arrow CI contract; native Fortran runtime dependency değildir |

## 9. Varsayımlar ve sınırlar

- JSON input top-level **array** olmalıdır; `{"records": [...]}` wrapper'ı
  streaming sözleşmesinin parçası değildir.
- Text record içinde CR/LF yoktur. Multi-paragraph source önce explicit
  sentence/record segmentation ile line-level record'lara ayrılmalıdır.
- Arrow/Parquet writer schema'sı stricttir. Başka bir producer'ın nested/map
  schema'sı doğrudan kabul edilmez; önce canonical logical record'a dönüştürün.
- `metadata_json` canonical JSON'dur; metadata query/projection için ayrı
  columnar model gerekiyorsa yeni, versioned schema tanımlayın.
- Generic rich schema UTF-8 text kabul eder; native HDC export ise mevcut
  Fortran tokenizer nedeniyle ASCII-onlydir.
- Bu ek katman distributed data lake, object store, automatic PII/license
  classification, semantic near-dedupe veya LLM-scale sharding değildir.
  Bu yetenekler ayrı security/governance/runtime sözleşmesi gerektirir.
