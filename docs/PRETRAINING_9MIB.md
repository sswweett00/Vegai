# Tam 9 MiB sentetik pretraining çalıştırması

Bu depo, tekrarlanabilir bir **9 MiB = 9.437.184 byte** HDC pretraining
çalıştırması için uçtan uca bir operasyon yolu içerir. Bu yolun amacı, büyük
bir metin koleksiyonunu depoya eklemek değil; streaming ingest, vocabulary
sayımı, HDC context öğrenimi, checkpoint yazımı, checkpoint okunabilirliği ve
throughput ölçüm zincirini gerçek bir byte bütçesi üzerinde denetlemektir.

> **Kapsam sınırı:** Varsayılan corpus algoritmik olarak üretilmiş strict-ASCII
> cluster/token kayıtlarıdır. Üçüncü taraf metni, dünya bilgisi, doğal dil
> öğretisi, güvenlik verisi veya domain bilgisi içermez. Başarılı bir 9 MiB koşu
> yalnızca bu yapısal/operasyonel sözleşmenin çalıştığını gösterir; doğal dil
> kalitesi, factuality, muhakeme, kodlama, güvenlik veya genel model kapasitesi
> kanıtı değildir.

## Neler bağlanır?

`tools/synthetic_pretraining_fixture.py` aşağıdaki sabit sözleşmeyi üretir ve
streaming doğrular:

- tam fiziksel byte sayısı: varsayılan `9MiB`, yani `9 * 1024 * 1024`;
- yalnız US-ASCII küçük harf, rakam, `_`, boşluk ve LF; CR/UTF-8/overlong record
  kabul edilmez;
- Vegai'nin `TOKEN_LEN=48`, `MAX_LINE_CHARS=8192` ve
  `MAX_TOKENS_PER_LINE=1024` sınırlarının altında kayıtlar;
- deterministic clustered-cyclic token topology (`32 × 32 = 1.024` sözlük
  öğesi); seyrek komşu-cluster bridge yalnız context/topology egzersizidir;
- corpus SHA-256, manifest SHA-256, line/token/frequency sayıları, probe token
  ve provenance sınırı;
- corpus'u tek string olarak belleğe almadan line-by-line doğrulama.

Her `generate` işlemi, corpus ve manifest'i ayrı atomik replace ile yazar;
sonra aynı streaming verifier ile kendi çıktısını yeniden okur. `verify`,
byte sayısı, digest, LF/ASCII/token biçimi, line/token limitleri ve bütün
manifest aggregate'leri uyuşmazsa fail-closed döner.

`tools/run_9mib_pretraining.py` yalnız doğrulanmış bu fixture'ı kabul eder.
Train öncesi ve sonrası corpus/manifest/binary kimliklerini hash'ler, RAM ve
checkpoint storage için `training_resource_plan.py` preflight'ını çalıştırır,
sınırlı stdout/stderr ve timeout ile bir `vegai_pretrain train` child process'i
başlatır, sayaçları manifest ile karşılaştırır, checkpoint SHA-256 üretir ve
ayrı bir `inspect --query` ile checkpoint'in okunabildiğini doğrular.

## Yerel kullanım

GNU Fortran/OpenMP olan bir hostta en kısa yol:

```bash
# build/, corpora/, checkpoints/ ve pretraining-runs/ Git tarafından izlenmez.
make pretrain-9mib

# Daha sonra fixture kimliğini bağımsız olarak yeniden denetleyin.
make verify-fixture-9mib
```

Bu hedef varsayılan olarak şunları üretir:

```text
corpora/synthetic_hdc_9mib.txt
corpora/synthetic_hdc_9mib.manifest.json
checkpoints/synthetic_hdc_9mib.vgai
pretraining-runs/synthetic_hdc_9mib.json
```

Komutları görünür biçimde ayırmak istenirse:

```bash
python3 tools/synthetic_pretraining_fixture.py generate \
  --output corpora/synthetic_hdc_9mib.txt \
  --manifest corpora/synthetic_hdc_9mib.manifest.json \
  --target-bytes 9MiB

python3 tools/synthetic_pretraining_fixture.py verify \
  --corpus corpora/synthetic_hdc_9mib.txt \
  --manifest corpora/synthetic_hdc_9mib.manifest.json

make pretrain
python3 tools/run_9mib_pretraining.py \
  --pretrain-bin ./build/vegai_pretrain \
  --corpus corpora/synthetic_hdc_9mib.txt \
  --manifest corpora/synthetic_hdc_9mib.manifest.json \
  --checkpoint checkpoints/synthetic_hdc_9mib.vgai \
  --report pretraining-runs/synthetic_hdc_9mib.json \
  --max-vocab 1024 --min-frequency 1 --window 2 \
  --context-mode hierarchical --epochs 1 --omp-threads 2
```

`--target-bytes 9000000` ile tam ondalık 9 MB ayrıca üretilebilir. Kanonik CI
fixture'ı özellikle belirsizliği önlemek için `9MiB` kullanır.

## Ölçüm tanımı

Raporun `measurement.train_process_wall_seconds` alanı **yalnız bir**
`vegai_pretrain train` subprocess çağrısının monotonic wall-clock aralığıdır:

- başlangıç: `Popen` çağrısından hemen önce;
- bitiş: bu train child process'i exit ettikten hemen sonra.

Aşağıdakiler bu değere dahil değildir: fixture üretimi, manifest doğrulaması,
resource preflight, binary/dataset hash'i, compilation, test discovery, CI
orchestration, checkpoint `inspect`, report formatting veya report yazımı.
Corpus'un vocabulary scan'i, streaming training epoch'u ve checkpoint write'ı
`train` komutunun doğal parçaları olduğu için dahildir. Rapor hem raw hem
active token/s sayısını, train stdout/stderr digest/byte sayılarını, OMP
değişkenlerini, binary/corpus/manifest/checkpoint digest'lerini ve inspect
sonucunu taşır.

Timeout, malformed CLI metric'i, OOV/truncation/strict-ASCII ihlali,
input değişimi, boş checkpoint veya inspect hatası başarı raporuna dönüşmez.
Başarısızsa runner en iyi çabayla `status: failed` raporu yazar ve nonzero
çıkar. `--skip-resource-preflight`, sadece izole contract testlerinde uygun
olan, rapora açıkça yazılan bir bypass'tır; normal eğitimde kullanılmamalıdır.

## CI ve gerçek binary doğrulaması

`.github/workflows/pretraining-9mib.yml`, pretraining altyapısı veya Fortran
train kaynakları değiştiğinde pull request'te gerçek GNU Fortran/OpenMP binary
oluşturur; runtime alanında fixture'ı üretir, doğrular ve bu runner ile bir
full 9 MiB/1 epoch koşu yapar. Manifest, run report ve checkpoint kısa süreli
GitHub artifact'i olarak saklanır. Böylece repository'ye 9 MiB corpus ya da
checkpoint commit edilmez; CI sonucu yine de indirilebilir/denetlenebilir.

`tools/verify_9mib_pretraining.py` ise normal CTest ve `make check` içinde
çalışır. Tam 9 MiB üretimi, deterministic corpus/manifest, tamper reddi,
runner'ın output/input alias koruması ve fake executable ile success/failure
rapor sözleşmesini denetler. CTest ayrıca
`tools/verify_9mib_pretraining_real_binary.py` ile aynı schema'nın 256 KiB
bounded instance'ını gerçek derlenmiş Fortran `train`/`inspect` ABI'sine bağlar;
bu küçük test normal native doğrulamayı kısa tutar. Full 9 MiB model koşusu
ayrı CI job'ında kaldığı için bu Python contract testi model eğitimi yerine
geçmez.

## Kaydedilmiş gerçek Fortran koşusu

Bu mekanizma ilk kez aşağıdaki GitHub Actions pull-request job'ında gerçek
GNU Fortran/OpenMP binary ile çalıştırıldı:

- kaynak commit: [`5f48f38`](https://github.com/sswweett00/Vegai/commit/5f48f38fe6ea79d63533ac289ab58ee9f48b5ac7);
- workflow/job: [Synthetic 9 MiB pretraining attestation #40091559976](https://github.com/sswweett00/Vegai/actions/runs/40091559976),
  `Real Fortran 9 MiB fixture run` (GitHub-hosted `ubuntu-latest`);
- fixture: `9,437,184` byte, `15,942` physical record, `255,058` raw token,
  `1,024` unique token;
- corpus SHA-256:
  `979a1e9938599c0d80b7b52082f79ee84a4ae936690e482aed7492600d65e454`;
- manifest SHA-256 (deterministic canonical output):
  `cb44f95b08dc565c9af5dbcddbc0b448a0303359d7dcfe939d97f519fc9a5180`;
- config: `max-vocab=1024`, `min-frequency=1`, `window=2`,
  `position-shift=41`, `context-mode=hierarchical`, `epochs=1`,
  `OMP_NUM_THREADS=2`;
- yalnız train child wall-clock: **`9.123744 s`**;
- raw token throughput: **`27955.411 token/s`**;
- checkpoint SHA-256:
  `a09e5e55e9dcbac99fd65eff030dfe4f1e545b7fb4e86f3867c34711cd3b1ff2`.

Runner'ın success contract'i bu koşuda active token sayısının raw token sayısına
eşit olmasını, OOV/truncation/non-ASCII/overlong record sayaçlarının sıfır
olmasını, checkpoint'in nonempty olmasını ve manifest probe ile ayrı
`inspect` çağrısının başarılı olmasını da zorunlu kıldı. Full report, binary
SHA-256 ve checkpoint boyutu, 14 gün saklanan
`synthetic-9mib-pretraining-40091559976` GitHub Actions artifact'indedir
(artifact ID `12559336717`; koşu anındaki expiry `2026-11-07`).

Bu tek gözlem derleme, fixture generation, verification, resource preflight,
inspection, artifact upload ve CI queue/overhead'i içermeyen train-only
ölçümdür. Farklı runner, CPU, OMP thread sayısı, compiler veya config ile
karşılaştırılabilir bir hız iddiası değildir; production/domain kalite ölçümü
de hiç değildir.

## Gerçek üretim corpus'una geçiş

Lisansı, kaynak URL/kimliği, kullanım izni, hedef-domain tanımı ve held-out
başarı ölçütü olan gerçek veride bu sentetik fixture'ı tekrar ederek bilgi
öğretildiği iddia edilmemelidir. Bunun yerine:

1. `tools/production_dataset.py` ile immutable provenance, exact dedupe ve
   held-out split package oluşturun.
2. `tools/production_pretrain.py` ile binary/dataset/config/checkpoint lineage
   bağlayın; gerekiyorsa `PRETRAINING_QUALITY_GATE.md`'deki operator-owned
   held-out semantic-neighbour gate'i zorunlu kılın.
3. Hedef hostta corpus, model kapasitesi, RAM/SSD ve OMP ayarları için
   `tools/training_resource_plan.py --require-ready` çalıştırın.
4. Gerçek baseline/candidate kalite, maliyet ve latency ölçümlerini ayrı,
   immutable raporlarda karşılaştırın. Fixture throughput'u domain kalitesine
   veya başka modellerle kapasite karşılaştırmasına çevrilmemelidir.

İlgili temel sözleşmeler: [PRETRAINING.md](PRETRAINING.md),
[PRODUCTION_PRETRAINING.md](PRODUCTION_PRETRAINING.md),
[PRETRAINING_QUALITY_GATE.md](PRETRAINING_QUALITY_GATE.md) ve
[RESOURCE_AWARE_TRAINING.md](RESOURCE_AWARE_TRAINING.md).
