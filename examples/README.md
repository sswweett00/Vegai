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

## 3. Veri odakli grounded diyalog fixture'i

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

Bu sekiz response gercek bilgi, dogal Turkce kalite veya LLM cevabi degildir;
data contract ve grounded retrieval integration fixture'idir. TSV strict
ASCII/LF, tek TAB ve active-checkpoint-prompt sozlesmesine uymalidir. Ayrintilar
icin [DIALOGUE_RETRIEVAL.md](../docs/DIALOGUE_RETRIEVAL.md)'ye bakın.

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

## 4. Production-data foundry fixture'ı

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
