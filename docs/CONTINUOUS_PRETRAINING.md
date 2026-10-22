# Kesintisiz pretraining: immutable shard ledger ve exactly-once commit

> **Sınır:** Bu belge HDC semantic prototype'ların tersinir bir corpus arşivi
> olduğunu iddia etmez. `bundling` ve Hebbian update semantik olarak kayıplıdır.
> Buradaki **kayıpsız** sözü, kabul edilmiş corpus byte'larının doğrulanması,
> tokenizer kaynaklı sessiz düşmenin reddi ve bir shard'ın durable model
> durumuna en fazla bir kez commit edilmesi anlamındadır.

`vegai_pretrain continue` ve `tools/continuous_pretrain.py`, tek bir büyük
corpus yerine sıralı immutable shard'lar üzerinden continual HDC/VSA öğrenme
için birlikte çalışır. Bu katman, `train`/`resume` için genel amaçlı bir job
scheduler değildir: düzenli, append-only bir veri ledger'ı ve sabit eğitim
policy'si gerektirir.

## 1. Model katmanı

Fortran API'si:

```fortran
call pretrain_extend_vocabulary(model, shard_path, config, vocabulary_metrics, status)
call pretrain_train_file(model, shard_path, config, training_metrics, status)
```

`pretrain_extend_vocabulary` mevcut checkpoint'i yüklenmiş model üzerinde
shard'ı gözlemler. Sabit `max_vocabulary` kapasitesinin izin verdiği yeni
slotları ekler; yalnız **yeni active** slotların deterministic base identity,
semantic prototype ve experience counter'ını başlatır. Önceden var olan
slotların prototype ve learning schedule durumu resetlenmez.

CLI bunun candidate checkpoint üzerinde çalışan güvenli yüzüdür:

```bash
# Mevcut checkpoint'i yerinde genişletip shard'ı öğrenir.
# Controller doğrudan durable checkpoint üzerinde değil, private candidate
# üzerinde bu komutu çağırır.
./build/vegai_pretrain continue SHARD.txt candidate.vgai --epochs 1 --strict
```

Continual ledger yolu `min_frequency=1` zorunluluğu uygular. Böylece strict bir
shard'daki her kabul edilmiş token, aynı shard'ın training geçişinden önce
active olur. Capacity aşımı `PT_ERR_VOCABULARY` ile durur; approximate top-k
veya sessiz token çıkarma yoktur.

## 2. Kabul edilen shard sözleşmesi

`continuous_shard_manifest.py` her input'u byte olarak tarar, SHA-256 hesaplar
ve aşağıdaki koşullardan herhangi biri varsa ledger'a hiç yazmaz:

- 8.192 byte'tan uzun fiziksel kayıt;
- 48 byte'tan uzun token veya satırda 1.024'ten fazla token nedeniyle drop;
- `>127` byte (Unicode/UTF-8 dahil);
- CRLF kaydı; continuous path yalnız LF kabul eder;
- sıfır kabul edilmiş token;
- aynı SHA-256 içeriğinin manifestte ikinci kez görünmesi.

Bu, normal tokenizer'ın telemetry üretme davranışından daha dar bir
operasyonel contract'tır. Ayrıca `vegai_pretrain --strict`, token-limit ve
non-ASCII byte için `PT_ERR_TOKENIZATION`; overlong record için
`PT_ERR_RECORD_TOO_LONG` ile durur. Controller yine de CLI'a güvenmek yerine
önce ve sonra kendi byte preflight'ını yapar.

Önce source corpus'un reversible/izlenebilir ASCII dönüşümünü üretin. Türkçe
ham UTF-8 metni doğrudan bu path'e koymayın; hangi normalizasyon veya
transliteration'ın kullanıldığını provenance alanında kaydedin.

## 3. Append-only immutable manifest üretimi

Shard sırası eğitim sırasıdır. Manifest aracı dosyayı atomik olarak replace
eder (`fsync` + rename); controller hiçbir zaman yarım JSON okumaz.

```bash
mkdir -p data/shards state/continual
# data/shards/000001.txt ve 000002.txt yalnız LF/ASCII strict shard'lardır.
python3 tools/continuous_shard_manifest.py \
  data/shards/000001.txt data/shards/000002.txt \
  --output data/continual-ledger.json \
  --source 'dataset-name release-2026-10' \
  --license 'dataset-license-id' \
  --preprocessing 'normalizer v1; ASCII reversible export'
```

Her entry `id == sha256`, byte count, LF-record count ve strict preflight
counter'larını taşır. Path manifest dizinine göre relatiftir. Yeni veri
alındığında önceki entry'leri, sıralarını veya hash'lerini değiştirmeden liste
sonuna ekleyin ve manifesti tekrar atomik yayınlayın. Controller committed
prefix'i state journal ile byte-byte kimlik açısından karşılaştırır; silme,
sıra değiştirme veya rewrite reddedilir.

## 4. Controller ile ilk commit ve devamı

İlk çağrı immutable HDC policy'yi state'e yazar. `--min-frequency 1` zorunlu;
`max-vocab` ise tüm zamanlardaki unique token sayısına yetecek biçimde
planlanmalıdır.

```bash
python3 tools/continuous_pretrain.py \
  --pretrain-bin ./build/vegai_pretrain \
  --manifest data/continual-ledger.json \
  --state-dir state/continual \
  --max-vocab 50000 --min-frequency 1 \
  --window 4 --position-shift 73 --context-mode hierarchical \
  --epochs 1 --initial-plasticity 25 --minimum-plasticity 1 --decay-events 32
```

State var olduktan sonra policy option'ları verilirse mevcut immutable değerle
aynı olmak zorundadır; aksi halde çağrı başarısız olur. `context_mode` policy
allowlist'i `legacy`, `positional`, `hybrid` ve `hierarchical` değerlerini
kabul eder; hierarchical mode yalnız ≥12 active-token içeren shard kayıtlarında
bounded whole-record signal ekler. Sıradan restart için aynı üç temel argüman
yeterlidir:

```bash
python3 tools/continuous_pretrain.py \
  --pretrain-bin ./build/vegai_pretrain \
  --manifest data/continual-ledger.json \
  --state-dir state/continual
```

Yeni atomik manifest yayınlarını izlemek için:

```bash
python3 tools/continuous_pretrain.py \
  --pretrain-bin ./build/vegai_pretrain \
  --manifest data/continual-ledger.json \
  --state-dir state/continual --follow --poll-seconds 10
```

`--max-shards N`, recovery/integration işleri için bir çağrıda en fazla `N`
pending shard commit eder. Aynı state directory üzerinde tek controller
çalışabilir. O_EXCL lock kalırsa önce gerçek owner'ı inceleyin; yalnız stale
olduğu operatörce doğrulandıktan sonra `--recover-lock` kullanın.

## 5. Durable transaction protokolü

Her pending shard için controller aşağıdaki sırayı uygular:

1. State journal'daki her checkpoint SHA-256 ile doğrulanır ve manifestin
   committed prefix'i ile karşılaştırılır.
2. Publisher shard'ı manifest hash/preflight ile yeniden taranır.
3. Controller state dizinindeki private snapshot kopyalanır ve aynı SHA-256
   ile tekrar doğrulanır. Child process yalnız bu snapshot'ı okur; source path
   eşzamanlı değiştirilse bile mixed read öğrenilmez.
4. Önceki durable checkpoint (veya ilk `train`) kullanılarak unique candidate
   checkpoint üzerinde gerçek `vegai_pretrain train`/`continue` komutu çalışır.
   CLI çıktısında observed/raw/active token kapsamı, `OOV=0`, zero
   truncation/non-ASCII/overlong telemetry ve beklenen epoch sayısı doğrulanır.
5. Publisher shard yeniden hash'lenir. Candidate `fsync` edilir ve immutable
   generation dosyasına atomik rename edilir.
6. Sadece en son `state.json`, `fsync` edilmiş temporary JSON + atomic rename
   ile yeni generation'ı commit marker olarak yayınlar.

`state.json` içindeki `applied_shards` her generation için shard SHA-256,
byte/token sayıları, checkpoint SHA-256 ve doğrulanan OOV metriklerini taşır.
`generations/` içindeki checkpoint tek başına commit sayılmaz.

| Kesinti noktası | Sonraki çağrının sonucu |
|---|---|
| candidate eğitiminden önce | journal eski generation'da kalır; shard baştan denenir |
| candidate/final checkpoint yazıldıktan sonra, `state.json` öncesi | unjournaled orphan yok sayılır/üstüne yazılır; önceki durable modelden shard yeniden denenir |
| `state.json` atomik rename sonrasında | generation committed'dir; restart shard'ı tekrar eğitmez |

Bu tasarım başarısız bir candidate'ın CPU işini yeniden yapabilir, ama committed
modelde duplicate Hebbian update oluşturmaz. Network/filesystem gerçek
atomiklik garantileri deployment'ın local durable filesystem'ine bağlıdır;
state dizinini güvenilir, tek-writer bir volume üzerinde tutun ve generation
checkpoint'lerini manuel olarak silmeyin.

## 6. Doğrulama

CMake/CTest içindeki `vegai_continuous_pretraining` testi gerçek derlenmiş
CLI ile iki shard çalıştırır. Test şunları denetler:

- ikinci shard'ın `gamma` vocabulary slotunu eklemesi;
- mutation edilmiş pending shard'ın SHA-256 nedeniyle CLI çağrılmadan reddi;
- iki shard için `training_oov_tokens = 0` ve accepted-token accounting;
- completed epoch sayısının iki olması;
- completed state ile tekrar çalıştırmanın state/checkpoint byte'larını
  değiştirmemesi;
- committed manifest prefix'inin reordering ile reddi;
- non-ASCII ve CRLF shard'ların manifest authoring aşamasında reddi ile
  manifest output'unun input shard'ı overwrite etmesine karşı koruma.

Tek başına iki-pass corpus pipeline, telemetri ve checkpoint formatı için
[PRETRAINING.md](PRETRAINING.md)'ye; gerçek speed/semantic benchmark sınırları
 için [PRETRAINING_MODE_COMPARISON.md](PRETRAINING_MODE_COMPARISON.md)'ye
bakın.
