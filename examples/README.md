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
  --context-mode hybrid --epochs 2 --initial-plasticity 25 --minimum-plasticity 1 \
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
belgesindedir. `legacy`, bit-tam hızlı `positional` ve varsayılan `hybrid`
encoder'ın eğitim hızı ile controlled topical-cohesion karşılaştırması için:

```bash
python3 tools/compare_pretraining_modes.py \
  --pretrain-bin ./build/vegai_pretrain --repetitions 3 --omp-threads 1
```

Bu yalnız sentetik in-corpus diagnostic'tir; doğal Türkçe anlama benchmarkı
veya production throughput iddiası değildir.

## Veri yönetişimi

`*.vgai` çalışma çıktısıdır; gerçek corpus veya checkpoint'i depoya commit
etmeyin. Veri kaynağı, lisans, rıza/kişisel veri analizi, preprocessing komutu
ve corpus digest'i proje dışı bir deney manifestinde kaydedilmelidir. Yerel
corpus digest/preflight için `python3 tools/corpus_manifest.py CORPUS --output
manifest.json ...` kullanılabilir. Ayrıntılı veri yönetişimi ve model semantiği
[`docs/PRETRAINING.md`](../docs/PRETRAINING.md) belgesindedir.
