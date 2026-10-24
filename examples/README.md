# Çalıştırılabilir corpus örnekleri

Bu klasördeki iki corpus da Vegai deposu için yazılmış, **tamamen sentetik**,
ASCII-only ve MIT lisanslı fixture'lardır. Gerçek kişi, gerçek kullanıcı verisi,
telifli kaynak veya doğrulanmış dünyasal bilgi içermezler. Semantic kalite
benchmarkı ya da doğal Türkçe kapsamı iddia etmezler.

## 1. Hızlı smoke corpus

`corpus/toy_hdc_corpus.txt`, çok küçük bir smoke-test corpus'udur. İki-pass
akışı, checkpoint ve query komutunun en kısa örneğini verir:

```bash
./build/vegai_pretrain train examples/corpus/toy_hdc_corpus.txt toy.vgai \
  --max-vocab 64 --min-frequency 2 --epochs 2 --query king

./build/vegai_pretrain inspect toy.vgai --query dog
./build/vegai_pretrain resume examples/corpus/toy_hdc_corpus.txt toy.vgai \
  --epochs 1 --query queen
```

## 2. Dengeli Türkçe-ASCII pretraining fixture'ı

`corpus/mini_turkish_ascii_pretrain.txt`, tokenizer'ın mevcut ASCII sözleşmesi
ile uyumlu, Türkçe kelimelerin ASCII transliterasyonundan oluşan daha büyük bir
örnektir. Sekiz ayrı bağlam kümesi bulunur:

- eğitim: `okul`, `ogrenci`, `ogretmen`, ...;
- sağlık: `hastane`, `doktor`, `hemsire`, ...;
- doğa, teknoloji, pazar, ulaşım, tarım ve mahalle/dayanışma.

Her küme 8 token ve 16 döngüsel/sıra-ters çevrilmiş kayıttan oluşur. Sonuçta
fixture tam olarak **128 kayıt**, **1.024 token**, **64 unique token** içerir;
her token **16 kez** geçer. `--min-frequency 8` ile bütün token'lar active
olur. Bu kontrollü yapı, gerçek dil yeteneği kanıtı değildir; repeatable corpus
pretraining ve context-window telemetrisi için tasarlanmıştır.

```bash
./build/vegai_pretrain train examples/corpus/mini_turkish_ascii_pretrain.txt \
  mini-turkish.vgai \
  --max-vocab 128 --min-frequency 8 --window 2 --position-shift 41 \
  --context-mode hierarchical --epochs 2 --initial-plasticity 25 --minimum-plasticity 1 \
  --decay-events 8 --strict --query okul

./build/vegai_pretrain inspect mini-turkish.vgai --query teknoloji
./build/vegai_pretrain resume examples/corpus/mini_turkish_ascii_pretrain.txt \
  mini-turkish.vgai --epochs 1 --strict --query okul
```

İlgili `mini_turkish_ascii_pretrain.manifest.json`, source-controlled corpus
SHA-256'sini, byte/kayıt sayısını ve ingest preflight sonucunu taşır. Manifest
fixture değiştiğinde yeniden üretilmelidir:

```bash
python3 tools/corpus_manifest.py examples/corpus/mini_turkish_ascii_pretrain.txt \
  --output examples/corpus/mini_turkish_ascii_pretrain.manifest.json \
  --source 'Vegai repository-authored synthetic fixture' --license MIT \
  --retrieved-at 'source-controlled fixture' \
  --preprocessing 'ASCII-safe Turkish transliteration; deterministic cluster rotations' \
  --strict-contract
```

### Fixture testi

```bash
# Corpus yapısı, manifest digest'i ve tokenizer sözleşmesi
python3 tools/verify_example_corpus.py

# Yukarıdakilere ek olarak gerçek CLI train/checkpoint/inspect/resume determinism testi
python3 tools/verify_example_corpus.py --pretrain-bin ./build/vegai_pretrain
```

İkinci komut iki özdeş eğitimin bit-tam checkpoint üretmesini, `inspect` ve
`resume` çalışmasını ve beklenen corpus telemetrisini doğrular. Semantic nearest
neighbour etiketi özellikle assert edilmez: HDC context prototype'ı için bu
fixture ancak diagnostic sinyal üretir, anlam doğruluğu sertifikası değildir.
Gerçek executable run'ın counter/schedule/checkpoint ayrıntıları
[`docs/EXAMPLE_PRETRAINING_RUN.md`](../docs/EXAMPLE_PRETRAINING_RUN.md)
belgesindedir. `legacy`, bit-tam hızlı `positional`, explicit local fused
`hybrid` ve varsayılan long-record `hierarchical` controlled diagnostic'i ile
fused/reference checkpoint byte-eşitliği karşılaştırması için:

```bash
python3 tools/compare_pretraining_modes.py \
  --pretrain-bin ./build/vegai_pretrain --repetitions 3 --omp-threads 1
```

Bu yalnız sentetik in-corpus diagnostic'tir; doğal Türkçe anlama benchmarkı
veya production throughput iddiası değildir. Kaydedilmiş CI önce/sonra değerleri
[`docs/PRETRAINING_MODE_COMPARISON.md`](../docs/PRETRAINING_MODE_COMPARISON.md)
belgesindedir.

## 3. Rich training-data schema fixture'ı

`training_data/mini_turkish_ascii_records.jsonl`, yeni
`vegai-training-record/v1` sözleşmesinin **16 kayıt / 4 label** içeren tamamen
sentetik örneğidir. `id`, `label`, `source` ve JSON `metadata` alanlarını
korur; `text` alanlarının tümü strict ASCII HDC export ile uyumludur. Bu
fixture gerçek Türkçe anlama, sınıf dengesi veya production corpus iddiası
değildir.

```bash
# Parquet/Arrow round-trip dahil data-I/O verifier için isolated dependency environment
make data-check
. .venv-dataio/bin/activate

# Schema + strict native-text contract + label imbalance audit
python tools/training_dataset.py validate \
  examples/training_data/mini_turkish_ascii_records.jsonl --hdc-strict

# Metadata korunarak deterministic exact-dedupe + train/validation/test split
python tools/training_dataset.py prepare \
  examples/training_data/mini_turkish_ascii_records.jsonl \
  --output-dir /tmp/vegai-rich-fixture-prepared --output-format parquet \
  --output-compression zstd

# Mevcut immutable production foundry yalnız text'i explicit projection eder.
python tools/production_dataset.py \
  --input examples/training_data/mini_turkish_ascii_records.jsonl --input-format jsonl \
  --output-dir /tmp/vegai-rich-fixture-production \
  --source 'Vegai synthetic rich fixture/v1' --license MIT \
  --retrieved-at 'source-controlled fixture' \
  --preprocessing 'ASCII-safe fixture; no label-to-text injection' \
  --validation-basis-points 2500
```

`tools/verify_training_data_io.py` fixture'ın record count/label setini ve her
text'in strict HDC compatibility'sini CI'da doğrular; ayrıca tüm supported
format round-trip, corruption ve raw-source-to-canonical-text provenance
binding testlerini çalıştırır. Ayrıntılı format sözleşmesi
[`docs/TRAINING_DATA_FORMAT.md`](../docs/TRAINING_DATA_FORMAT.md)'dedir.

## 4. Veri odakli grounded diyalog fixture'i

`dialogue/mini_turkish_hdc_dialogue.tsv`, yukaridaki mini corpus'un active
ASCII token'lariyla yazilmis sentetik `prompt<TAB>response` bilgi bankasidir.
`vegai_chat` cevap metnini source code'dan üretmez; bu TSV'den aynen alir ya da
grounding esigi gecmezse abstain eder. Once mini checkpoint'i egittikten sonra:

```bash
./build/vegai_chat ask mini.vgai examples/dialogue/mini_turkish_hdc_dialogue.tsv \
  --prompt "okul ders" --max-entries 64 --history-turns 0

./build/vegai_chat repl mini.vgai examples/dialogue/mini_turkish_hdc_dialogue.tsv \
  --max-entries 64 --history-turns 3
```

`dialogue/mini_turkish_hdc_evaluation.jsonl`, ayni sentetik domain icin sekiz
exact grounding case'i ve bir expected-abstention case'i tasir. Gercek compiled
chat ile versioned held-out quality/gate smoke testi icin:

```bash
python3 tools/dialogue_evaluation.py \
  --chat-bin ./build/vegai_chat --checkpoint mini.vgai \
  --knowledge examples/dialogue/mini_turkish_hdc_dialogue.tsv \
  --evaluation examples/dialogue/mini_turkish_hdc_evaluation.jsonl \
  --output /tmp/mini-dialogue-quality.json --max-entries 64 \
  --min-similarity 0.90 --min-margin 0.00 \
  --min-grounded-prompt-accuracy 1 --min-grounded-precision 1 \
  --min-grounded-recall 1 --min-abstain-precision 1 --min-abstain-recall 1 \
  --max-false-grounded-cases 0 --max-false-abstain-cases 0
```

Bu sekiz response, dokuz evaluation case'i ve `1.0` smoke eşikleri gerçek bilgi,
doğal Türkçe kalite veya LLM cevabı değildir; data contract ve grounded
retrieval integration fixture'idir. TSV strict ASCII/LF, tek TAB ve
active-checkpoint-prompt sözleşmesine uymalıdır. Gerçek release threshold'u
ayrı calibration setinde seçilmeli, frozen test setinde tuning yapılmamalıdır.
Ayrıntılar için [DIALOGUE_RETRIEVAL.md](../docs/DIALOGUE_RETRIEVAL.md) ve
[QUALITY_ASSURANCE.md](../docs/QUALITY_ASSURANCE.md)'ye bakın.

## 5. Optional LMM integration fixture'ı

`lmm/mini_multilingual_model.json`, `lmm/mini_multilingual_request.json` ve
`lmm/mini_multilingual_knowledge.jsonl`; Türkçe/Japonca/Arapça Unicode,
long-context request, model capability manifest'i ve source-grounded evidence
contract'i için **model-weightsiz sentetik** fixture'lardır. Bunlar bir gerçek
LMM, dünya bilgisi, fluent response veya media reasoning benchmarkı değildir.

```bash
# Rust BPE tokenizer varsa gerçek tokenizer.json path'i de test edilir.
make lmm-check

# Unicode source knowledge -> evidence bundle
python3 tools/lmm_knowledge.py build \
  --input examples/lmm/mini_multilingual_knowledge.jsonl \
  --output /tmp/vegai-lmm-knowledge.sqlite
python3 tools/lmm_knowledge.py query \
  --index /tmp/vegai-lmm-knowledge.sqlite --query '東京' \
  --output /tmp/vegai-lmm-evidence.json
```

Gerçek generation için operator; model weights/artifacts, tokenizer JSON,
source/license bilgisi ve backend seçimini sağlar. `worker` protocol'u
image/audio/video alabilir; OpenVINO GenAI adapter'i bu repositoryde text-only
olarak tanımlıdır. Operatorün gerçek backend'i için held-out citation/abstention
release gate'i `tools/lmm_quality_evaluation.py` ile çalışır; case'ler request +
evidence içerir, report ham prompt/response tutmaz. Ayrıntılı güvenlik,
performance, model package ve quality-gate sözleşmesi
[LMM_RUNTIME.md](../docs/LMM_RUNTIME.md)'dedir.

Target-domain expert seçiminde kullanılacak operator JSON şablonları
[`lmm/portfolio/`](lmm/portfolio/) altında bulunur. Bu şablonlardaki zero SHA
placeholder'ları gerçek artifact değildir; tamamlanmış model/evaluation
provenance'ı olmadan route etmeye çalışmak fail-closed olur. Çalışan sentetik
protocol fixture'ı `tools/verify_lmm_portfolio.py` temporary workspace'te
oluşturur. Ayrıntılı schema ve sınırlar için
[LMM_PORTFOLIO_ROUTING.md](../docs/LMM_PORTFOLIO_ROUTING.md)'ye bakın.

## Kesintisiz shard örneği

Bu fixture tek dosyalı deterministic smoke/benchmark içindir. Production benzeri
append-only akışta corpus'u ayrı strict ASCII/LF shard'lara bölün; önce
`tools/continuous_shard_manifest.py` ile SHA-256 ledger üretin, sonra
`tools/continuous_pretrain.py` ile `min-frequency=1` state journal'ına commit
edin. İkinci araç candidate checkpoint, private snapshot ve atomic state
publish kullanır; doğrudan `continue` komutunu durable checkpoint üzerinde
çalıştırmak aynı crash/restart garantisini vermez. Tam komutlar ve failure
semantics için
[`docs/CONTINUOUS_PRETRAINING.md`](../docs/CONTINUOUS_PRETRAINING.md)'ye bakın.

## 6. Production-data foundry fixture'ı

`production_dataset/mini_turkish_ascii_hdc/`, yukarıdaki 128-record sentetik
fixture'ın source-controlled, immutable production-package örneğidir. Exact
record dedupe, SHA-256 shard ledger ve deterministic %10 held-out split sonucu
**114 train record / 912 token / 64 exact unique token / 4 shard** ve
**14 validation record / 112 token / 1 shard** içerir. Bu bir LLM/LMM dataset'i veya semantic benchmark
değil; Vegai HDC için data governance/run-control integration fixture'ıdır.

```bash
# Tam shard/ledger/data manifest doğrulamasından sonra gerçek exactly-once HDC run
python3 tools/production_pretrain.py \
  --pretrain-bin ./build/vegai_pretrain \
  --dataset examples/production_dataset/mini_turkish_ascii_hdc/dataset.json \
  --run-dir /tmp/vegai-mini-production-run \
  --max-vocab 128 --min-frequency 1 --window 2 --position-shift 41 \
  --context-mode hierarchical --epochs 1 --omp-threads 2
```

Run önce exact train/validation record leakage audit ve resource preflight
yapar. Package'ı yeniden üretme, immutable SHA sözleşmesi, resume/failure
semantiği ve gerçek mimari sınırlar için
[`docs/PRODUCTION_PRETRAINING.md`](../docs/PRODUCTION_PRETRAINING.md)'ye bakın.

## Veri yönetişimi

`*.vgai` çalışma çıktısıdır; gerçek corpus veya checkpoint'i depoya commit
etmeyin. Veri kaynağı, lisans, rıza/kişisel veri analizi, preprocessing komutu
ve corpus digest'i proje dışı bir deney manifestinde kaydedilmelidir. Yerel
corpus digest/preflight için `python3 tools/corpus_manifest.py CORPUS --output
manifest.json ...` kullanılabilir. Ayrıntılı veri yönetişimi ve model semantiği
[`docs/PRETRAINING.md`](../docs/PRETRAINING.md) belgesindedir.
