# HDC pretraining: bit-tam hız ve controlled semantic-cohesion karşılaştırması

Bu belge `tools/compare_pretraining_modes.py` ile üretilen sınırlı ölçüm
protokolünü tanımlar. Araç gerçek `vegai_pretrain train` executable'ını çalıştırır
ve ölçümü yalnız o child process'in başlatılması ile bitişi arasında alır.
Compiler/configure, CI setup, fixture üretimi, checkpoint inspect, JSON yazımı
ve Python harness süresi **ölçüye dahil değildir**. Vocabulary pass, streaming
öğrenme ve checkpoint write ise `train` komutunun parçası oldukları için dahildir.

> Bu sonuçlar repository-authored, dengeli, sentetik ve corpus-içi controlled
> diagnostic'tir. Doğal dil zekâsı, held-out task accuracy, genel Türkçe anlama
> veya production throughput garantisi değildir. Gerçek corpus için ayrı
> held-out/de-duplicated evaluation manifesti gerekir.

## V2 protokolü

```bash
make clean && make all
python3 tools/compare_pretraining_modes.py \
  --pretrain-bin ./build/vegai_pretrain --repetitions 3 --omp-threads 1
```

Her varyant üç bağımsız train process'i ile ölçülür; tablo değeri wall-clock
medyandır. Sabit `OMP_NUM_THREADS=1`, `max-vocab=128`, `min-frequency=8`,
window `2`, positional shift `41`, plasticity `25 → 1`, decay `8` kullanılır.

### A. Base encoder diagnostic

`mini_turkish_ascii_pretrain.txt` sekiz disjoint topical kümeden oluşur:

```text
Corpus                    mini_turkish_ascii_pretrain.txt
Record length             8 active token
Training token/run        2,048
Epoch/run                 2
Ordered context pair/run  6,656
Vocabulary                64 active token
```

Burada `legacy`, bit-tam hızlı `positional` ve fused varsayılan `hybrid`
çalıştırılır. Her checkpoint gerçek CLI üzerinden 64 self-excluded Top-1 query
ile okunur. `positional` nearest-neighbour çıktısının `legacy` ile tam eşit
olması gerekir; `hybrid` aynı-küme hit'i geriletmemeli ve mean Top-1 similarity
artışı `> 0.01` olmalıdır.

### B. Fused/reference semantic-preservation hız karşılaştırması

Aynı base corpus/config ile iki `hybrid` varyant çalışır:

- **reference:** `--reference-context-kernel` ile eski allocate/permute/iki-add
  yolu;
- **fused:** varsayılan raw SoA-word + fused positional/topic pair yolu.

İki checkpoint'in tüm byte dizisi ve SHA-256'sı **eşit** olmak zorundadır. Bu
kural Top-1 diagnostic eşitliğinden daha güçlüdür: vocabulary, base plane,
semantic plane, experience sayaçları, totals ve v5 checksum'ın tamamının aynı
olduğunu doğrular. Hız oranı `reference_median / fused_median` olarak verilir.

### C. Long-record hierarchical controlled diagnostic

Normal fixture kasıtlı olarak sekiz-token satırlıdır ve `hierarchical` eşiği
olan 12 active token'ın altında kalır. Araç geçici, ASCII-only bir fixture üretir:
her biri aynı disjoint topical kümenin sekiz token'ını iki kez içeren **16-token**
kayıtlar; küme başına 16 kayıt, toplam 128 kayıt. İki epoch için:

```text
Training token/run        4,096
Ordered context pair/run  14,848
```

`hierarchical`, local `hybrid` representation'ı korur ve yalnız bu uzun
kayıtlarda satır-base-vector majority özetini üç bounded katkı olarak ekler.
Bu nedenle comparison `hybrid` ile `hierarchical` arasında yapılır. Acceptance
signal: same-cluster Top-1 gerilemez ve mean Top-1 similarity gain `> 0.002`
olur. Bu bilerek tasarlanmış in-corpus cohesion testidir; held-out task
başarısı değildir.

## Önceki baseline (fused-pair değişikliğinden önce)

[GitHub Actions run 39826604165](https://github.com/sswweett00/Vegai/actions/runs/39826604165)
üzerindeki son önceki optimized GNU Make run'ı, o dönemin `legacy`, positional
ve pre-fused hybrid implementasyonunu şu şekilde raporladı:

| Encoder | Median train süresi | Training token/s | Legacy'ye göre | Mean Top-1 similarity |
|---|---:|---:|---:|---:|
| `legacy` | 0.466586 s | 4,389.331 | 1.0000× | — |
| `positional` | 0.160154 s | 12,787.702 | 2.9134× | 0.073368 |
| `hybrid` | 0.165548 s | 12,371.067 | 2.8184× | 0.137414 |

Base fixture'ın Top-1 cluster hit'i üç mode için `%100` idi; hybrid'in
positional'a göre similarity gain'i `+0.064046` idi. Bunlar tarihsel baseline
olarak saklanır; CPU/runner/derleyici farklı olduğunda mutlak saniye veya
token/s değerlerini doğrudan karşılaştırmayın.

## Güncel v2 sonucu

Bu kaynak değişikliğinin doğrulanmış GNU Fortran CI sonucu, commit ile birlikte
workflow summary'den bu bölüme kaydedilir. Raporda aşağıdaki üç ayrı sonuç
bulunmalıdır:

1. fused/reference hybrid checkpoint byte equality ve hız oranı;
2. base fixture `legacy → positional → fused hybrid` medyanları;
3. long-record `hybrid → hierarchical` controlled cohesion sonucu.

Her zaman aynı runner içindeki oranları yorumlayın. Cache, CPU frequency/termal
state, filesystem, compiler sürümü/flag'leri ve OpenMP ayarı mutlak ölçümü
değiştirir.

## Uygulama notları

- `hv_bitacc_add_words`, `hv_bitacc_add_permuted_words` ve
  `hv_bitacc_add_permuted_pair_words`, geçici `hv_t` kopyasını kaldırarak aynı
  bit-sliced population-counter arithmetic'ini doğrudan vocabulary'nin SoA
  column'unda uygular.
- Native property testleri negatif/pozitif shift, bütün counter plane'leri,
  sample count ve pair-overflow atomicity'sini kontrol eder.
- `--reference-context-kernel` checkpoint formatına yazılmaz; yalnız benchmark
  ve regression oracle'dır. Aynı semantic config ile fused/reference checkpoint
  byte eşitliği bu nedenle zorunludur.
- Checkpoint writer v5'tir; reader v4/v5 kabul eder ve checksum'ı dosyanın
  kendi version değeriyle doğrular.
- Hierarchical global signal satırda bir kez üretilir, yalnız ≥12 active token
  için etkinleşir ve local context accumulator kapasitesiyle bounded kalır.
