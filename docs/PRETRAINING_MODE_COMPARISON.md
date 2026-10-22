# HDC pretraining: önce / sonra karşılaştırması

Bu belge, bit-sliced context accumulator ve varsayılan `hybrid` context encoder
sonrası gerçek executable ile alınan kontrollü karşılaştırmayı kaydeder.
Ölçüm [GitHub Actions run 39823123909](https://github.com/sswweett00/Vegai/actions/runs/39823123909)
üzerinde, commit `922eac5` için başarıyla tamamlandı.

> Bu sonuçlar sentetik, dengeli ve corpus-içi bir diagnostic'tir. Transformer
> performansı, genel Türkçe dil anlama, held-out task accuracy veya production
> throughput garantisi değildir.

## Ölçüm protokolü

```bash
make clean && make all
python3 tools/compare_pretraining_modes.py \
  --pretrain-bin ./build/vegai_pretrain --repetitions 3 --omp-threads 1
```

Her mode için aynı corpus/config ile üç bağımsız `vegai_pretrain train` process'i
çalıştırıldı; tabloda wall-clock medyanı vardır.

```text
Corpus                 mini_turkish_ascii_pretrain.txt
Training token/run     2.048
Epoch/run              2
Ordered context pair   6.656
Vocabulary             64 active token
OMP_NUM_THREADS        1
Compiler target        GNU Make -O3 -march=native -fopenmp
Timer scope            tek train process; compiler/test/harness hariç
Train process scope    vocabulary pass + streaming learning + checkpoint write dahil
```

## Hız karşılaştırması

| Encoder | Median train süresi | Training token/s | Legacy'ye göre |
|---|---:|---:|---:|
| `legacy` | 0,332480 s | 6.159,771 | 1,0000× |
| `positional` | 0,098175 s | 20.860,690 | **3,3866×** |
| `hybrid` | 0,100236 s | 20.431,707 | **3,3170×** |

### Yorum

- `positional`, önceki signed-counter positional öğrenme semantiğini
  bit-tam korur; buna rağmen word-parallel `hv_bitacc_t` sayesinde **3,3866×**
  daha hızlıdır.
- Varsayılan `hybrid`, positional kanala unpermuted topical context view ekler.
  Bu ek semantic kanala rağmen `legacy` yolundan **3,3170×** daha hızlı kaldı.
- `hybrid`, saf hızlı positional encoder'dan yalnızca **0,002061 s** (%2,10)
  daha uzun sürdü; semantic-cohesion kazanımı bu kontrollü fixture'da aşağıda
  görünürdür.

## Controlled semantic-cohesion karşılaştırması

Her checkpoint bir kez yüklendi; `--query-file` ile 64 fixture token'ının
her biri için self-excluded Top-1 semantic neighbour sorgulandı. Her fixture
kümesi sekiz topical token içerir.

| Encoder | Aynı-küme Top-1 | Mean Top-1 bipolar similarity | Positional'a göre gain |
|---|---:|---:|---:|
| `legacy` | 64 / 64 (%100) | 0,073368 | — |
| `positional` | 64 / 64 (%100) | 0,073368 | 0,000000 |
| `hybrid` | 64 / 64 (%100) | **0,137414** | **+0,064046** (%87,29) |

`legacy` ve hızlı `positional` modlarının Top-1 neighbour output'ları testte
tam eşit bulundu; bu, hız optimizasyonunun positional model davranışını
değiştirmediğini gösterir. Fixture zaten üç modda da %100 Top-1 cluster hit
verdiği için kaliteli sinyal, accuracy artışı değil doğru topical komşuya olan
**similarity marjının** %87,29 artmasıdır.

## Yapılan iyileştirmeler

1. **Bit-sliced unit-weight bundling:** `hv_bitacc_t`, 16.384 ayrı bit için
   signed-counter/branch döngüsü yerine 64-bit word başına XOR/AND carry
   planes kullanır. `legacy` ve `positional` equivalence'ı native ve bağımsız
   Python reference testlerinde sınanır.
2. **Üç açık context mode:**
   - `legacy`: eski signed-counter positional regression referansı;
   - `positional`: aynı öğrenme sonucunu üreten hızlı yol;
   - `hybrid` (varsayılan): positional ve order-invariant topical evidence
     birleşimi.
3. **Resume güvenliği:** context mode checkpoint v4 payload/checksum'ına girer;
   farklı mode ile resume `PT_ERR_INCOMPATIBLE_CONFIG` verir.
4. **Batch semantic diagnostic:** `vegai_pretrain inspect --query-file PATH`,
   bir checkpoint'i tek kez yükleyerek her nonblank satırdaki query için
   nearest-neighbour raporu üretir. Bu hem benchmark overhead'ini düşürür hem
   operatör değerlendirmesini hızlandırır.
5. **CI karşılaştırması:** optimized binary üzerinde üç mode, üç repetition ve
   gerçek checkpoint query'leriyle hız/cohesion raporu workflow summary ve
   check-run annotation olarak üretilir.

## Sınırlar ve tekrar üretim

Süreler CPU modeli, compiler sürümü/flags, filesystem cache, termal/frequency
state ve `OMP_NUM_THREADS` değerine duyarlıdır. Farklı makinede mutlak token/s
sayısı değişebilir; karşılaştırmada esas alınması gereken aynı runner üzerindeki
mode-to-mode oranıdır.

Gerçek corpus kullanırken bu fixture sonucunu production semantic kalite iddiası
olarak kullanmayın. Lisanslı corpus, held-out task, leakage/dedup kontrolü,
robustness, bias/privacy ve hedef donanım benchmark'ı ile ayrı bir deney
manifesti oluşturun. Ayrıntılı veri sözleşmesi için
[PRETRAINING.md](PRETRAINING.md)'ye bakın.
