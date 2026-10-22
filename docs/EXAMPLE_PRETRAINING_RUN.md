# Kontrollü fixture üzerinde gerçek HDC pretraining çalışması

Bu rapor, `examples/corpus/mini_turkish_ascii_pretrain.txt` için
`tools/verify_example_corpus.py --pretrain-bin …` tarafından gerçek
`vegai_pretrain` executable'ının stdout'ından doğrulanan contract
istatistiklerini açıklar. Test iki bağımsız `train` çalıştırır, checkpoint
byte'larını karşılaştırır, `inspect` çağırır ve aynı checkpoint'i bir epoch
`resume` eder.

Bu, gerçek Fortran/HDC pipeline yürütmesidir; ancak input kasıtlı olarak küçük,
dengeli ve sentetik bir fixture'dır. Sonuçlar doğal dil kalite benchmarkı,
factual knowledge veya genel amaçlı Türkçe pretraining sonucu değildir.

## Reproducible komut

```bash
python3 tools/verify_example_corpus.py \
  --pretrain-bin ./build/vegai_pretrain
```

Script'in kullandığı eğitim config'i:

```text
max_vocabulary              = 128
min_frequency               = 8
context_window              = 2
position_shift_bits         = 41
context_mode                = hybrid
epochs                      = 2
initial_plasticity_percent  = 25
minimum_plasticity_percent  = 1
plasticity_decay_events     = 8
strict_tokenization         = true
```

## İzole `train` komutu zamanlaması

`tools/measure_example_pretraining.py`, aynı fixture/config ile **tek**
`vegai_pretrain train` process'ini `perf_counter_ns` ile ölçer. Sayaç process
spawn'ından hemen önce başlar ve process çıktığında durur; compiler configure,
derleme, CTest discovery ve önceki CI adımları buna dahil değildir. Komutun
kendi içindeki vocabulary pass, iki epoch streaming öğrenme ve checkpoint write
ise `train` subcommand'inin zorunlu parçaları olduğu için dahildir.

```bash
python3 tools/measure_example_pretraining.py \
  --pretrain-bin ./build/vegai_pretrain \
  --checkpoint /tmp/mini-turkish-timed.vgai \
  --context-mode hybrid --epochs 2 --omp-threads 1
```

CI derlemeden sonra bu timing kapsamını `tools/compare_pretraining_modes.py`
içinde `legacy`, bit-tam hızlı `positional` ve varsayılan fused `hybrid` için
üçer tekrar ile çalıştırır. Aynı config/corpus ile `--reference-context-kernel`
rotate-then-add yolu da ayrıca ölçülür ve fused checkpoint ile **byte-tam eşit**
olması zorunludur. Araç ayrıca normal sekiz-token fixture'ın eşiğinin altında
kaldığını açıkça koruyarak, ayrı üretilmiş 16-token-record controlled fixture
üzerinde `hybrid` ile `hierarchical` topical-cohesion diagnostic'ini raporlar.
Medyan elapsed süre, training-token/saniye ve dar semantic diagnostic workflow
summary/check-run notice'a yazılır. Süreler runner, derleyici bayrakları ve
CPU'ya bağlıdır; bunlar doğal dil kalitesi veya production throughput iddiası
olmayan reproducible smoke/controlled timing'lerdir.

## Input ve vocabulary pass

| Ölçüm | Doğrulanmış değer |
|---|---:|
| Corpus SHA-256 | `886f754c1cf65676c5d35837cc323b9c14f467a5b8e7f0be7db2097328faa362` |
| Byte | 6.544 |
| LF kayıt | 128 |
| Kabul edilen token | 1.024 |
| Unique token | 64 |
| Active token | 64 |
| Token frekansı | her token için 16 |
| Tokenizer truncation | 0 |
| Non-ASCII input byte | 0 |
| Overlong record | 0 |
| OOV token | 0 |

Her kayıt 8 token içerir. `min_frequency=8` eşiğinin üzerinde kalan 64 token'ın
tamamı active olduğundan, eğitim pass'inde hiçbir token OOV olarak düşmez.

## İki epoch gerçek eğitim sonucu

Aşağıdaki satırlar CLI'ın `phase: streaming HDC context learning` bölümünden
assert edilir:

| Ölçüm | 2-epoch train çağrısı |
|---|---:|
| Tamamlanan epoch | 2 |
| Corpus line pass | 256 |
| Raw token | 2.048 |
| Active token | 2.048 |
| OOV token | 0 |
| Ordered context pair | 6.656 |
| Prototype update | 2.048 |
| Tokenizer truncation | 0 |
| Non-ASCII byte | 0 |
| Overlong record | 0 |

Window `W=2` ve her kayıtta 8 active token olduğundan bir kayıt için ordered
context contribution sayısı:

```text
2 + 3 + 4 + 4 + 4 + 4 + 3 + 2 = 26
128 kayıt × 26 = 3.328 context pair / epoch
2 epoch × 3.328 = 6.656 context pair
```

Her center'ın en az bir active neighbour'ı vardır. Bu nedenle `1.024 token /
epoch` doğrudan `1.024 prototype update / epoch` verir. Dengeli fixture'da her
active token iki epoch sonunda `16 × 2 = 32` experience event biriktirir.

## Exact plasticity schedule hesabı

`plasticity_decay_events=8` için ilk iki epoch'ta token başına schedule şöyledir:

| Experience aralığı | Token başına update | Global update | Plastik oran | Seçilen boyut/update |
|---|---:|---:|---:|---:|
| `e = 0..7` | 8 | 512 | %25 | 4.096 |
| `e = 8..31` | 24 | 1.536 | %13 | 2.130 |

Böylece iki-epoch çağrısında exact-cardinality mask'lerin seçtiği toplam
boyut sayısı `512 × 4096 + 1536 × 2130 = 5.368.832` olur. Bu değer seçilmiş
replacement dimension sayısıdır; yeni context bit'i prototype bit'iyle aynıysa
nihai Hamming değişiminin daha küçük olması beklenir.

## Checkpoint, inspect ve resume

Test aynı config/corpus ile iki bağımsız train çağrısı yapar ve üretilen `.vgai`
checkpoint byte dizilerinin bit-tam eşit olduğunu doğrular. Ardından checkpoint
üzerinde `inspect --query teknoloji` çalışır; active vocabulary, frequency,
window ve positional shift değerleri kontrol edilir. Query'nin nearest-neighbour
etiketi semantic kalite iddiası olarak assert edilmez, yalnız gerçek semantic
retrieval yolunun çalıştığı doğrulanır.

Aynı checkpoint bir epoch daha resume edildiğinde:

| Ölçüm | Resume çağrısı | Checkpoint toplamı |
|---|---:|---:|
| Tamamlanan epoch | 3 | 3 |
| Corpus line | 128 | 384 |
| Raw / active token | 1.024 / 1.024 | 3.072 / 3.072 |
| Ordered context pair | 3.328 | 9.984 |
| Prototype update | 1.024 | 3.072 |
| OOV / truncation / non-ASCII / overlong | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

Resume epoch'unda her token 32'den 48 experience event'e ilerler. `e=32`,
`e < 4D` (`4 × 8 = 32`) yarım-oran fazının ilk dışındaki değerdir; ek 1.024
update bu nedenle %7 exact selection kullanır. %7 için seçilen cardinality
`(16.384 × 7 + 50) / 100 = 1.147` boyuttur. Ek seçilmiş dimension sayısı
`1.024 × 1.147 = 1.174.528`, üç epoch sonunda toplam seçilmiş dimension sayısı
ise `6.543.360`dır.

## Bellek ve sınırlar

`capacity=128` için yalnız HDC data plane'leri:

```text
base_bits      = 256 × 128 × 8 = 262.144 byte
semantic_bits  = 256 × 128 × 8 = 262.144 byte
experience     = 128 × 8       =   1.024 byte
-----------------------------------------------
HDC planes                         525.312 byte
```

CLI'ın `estimated resident bytes` alanı bunlara ek olarak token string,
frequency, active mask ve hash-index array'lerini de içerir. Sonuç compiler
ABI'sine göre `logical` storage size nedeniyle değişebilir; bu yüzden bu belge
portable API garantisi olarak tek bir toplam byte değeri vermez.

Gerçek, lisanslı ve doğal corpus ile production pretraining için data
governance, held-out evaluation, bias/privacy analizi ve task-specific
benchmark gereklidir. İlgili süreç için [PRETRAINING.md](PRETRAINING.md)'ye
bakın.
