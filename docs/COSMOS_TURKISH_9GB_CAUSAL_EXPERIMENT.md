# 9 GB COSMOS Türkçe Nedensel Ön-eğitim Protokolü

> **Durum: bir gerçek run tamamlandı.** `Real 9 GB COSMOS Turkish causal
> pretraining evidence` workflow'u 25 Ekim 2026'da başarıyla tamamlandı. Exact
> provenance, timing, held-out metric ve düzenlenmemiş prompt/response çıktıları
> [COSMOS_TURKISH_9GB_RESULTS.md](COSMOS_TURKISH_9GB_RESULTS.md)'dedir. Bu
> protokol belgesi tek başına kalite iddiası değildir.

## Amaç ve önceki sınır

Önceki `vegai_pretrain` yolu HDC/VSA bağlamsal ilişki öğrenir; `vegai_chat`
ise ayrı bir TSV'den literal/semantik retrieval yapar. Bunların hiçbiri
nedensel token olasılığı, decoding veya serbest metin üretimi değildir. Bu
nedenle 9 GB'ta sadece o yolu büyütmek, Türkçe serbest sohbet kalitesi veya
"zeka" kanıtı olarak sunulamazdı.

Bu protokolde eklenen `vegai_causal_lm` farklı bir native Fortran 2023
çalıştırılabiliridir. Her eğitim geçişinde:

1. kaynak metnin UTF-8 Türkçe token'larını okur;
2. `P(t_i | t_{i-1})` için bounded heavy-hitter successor tablosunu;
3. `P(t_i | t_{i-2}, t_{i-1})` için bounded set-associative trigram context
   cache'ini

günceller; sonra bir istemin son bir/iki token'ından örnekleme yaparak yeni
bir devam üretir. Böylece retrieval ile karıştırılmayan gerçek bir *causal
next-token generation* yüzeyi vardır.

Bu yine de Transformer, attention tabanlı neural LLM, instruction-tuned
asistan veya genel akıl yürütme sistemi değildir. Sınırlı successor listeleri
nadir token olasılıklarını kaybeder; trigram context cache'i collision/eviction
yapar; örneklenen devamlar yerel corpus istatistikleridir. Bu sebeple aşağıdaki
hiçbir iddia yapılmaz:

- genel zekâ veya insan düzeyi Türkçe;
- doğru bilgi, kaynaklı cevap, güvenli tavsiye veya instruction following;
- tüm kişisel verilerin yokluğu;
- near-duplicate'lerin tamamının yokluğu;
- GPU/NPU hızlandırması ya da bütün donanımlarda aynı throughput.

## Seçilen gerçek kaynak

Controller yalnızca şu kamuya açık veri setini kabul eder:

- Hugging Face dataset: `ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0`
- kataloglanan amaç: Türkçe continual-pretraining corpus;
- kataloglanan büyüklük: 22.8 GB / 9,075,453 Türkçe satır / 15B token;
- gerekli dataset-card lisansı: **CC-BY-4.0**.

Çalışma sırasında `main` etiketi önce API'den hexadecimal immutable revision'a
çözülür. Bu revision daha sonra card ve her parquet shard indirmesinde tekrar
kullanılır. Card'ın YAML front matter alanında `license: cc-by-4.0` bulunmazsa,
parquet toplam advertised size 10 GB'ın altındaysa, onaylı UTF-8 metin kolonu
bulunamazsa veya 9 GB çıktıya ulaşılamazsa işlem **başarısız olur**. Başarı
raporu resolved revision, card SHA-256, shard SHA-256'ları ve shard isimlerini
tutar.

CC-BY attribution yükümlülüğü için yayımlanan artifact card'ı ve source
identity'yi saklar. Ham 9 GB corpus artifact olarak yüklenmez.

## Veri yapımı: tam byte, tekrar yok, split ayrımı

`tools/run_cosmos_turkish_9gb.py` aşağıdaki deterministic sözleşmeyi uygular:

| Kontrol | Uygulama |
|---|---|
| Eğitim bütçesi | Tam **9,000,000,000 decimal byte**, LF ile biten UTF-8 kaynak-türevi kayıtlar |
| Held-out | Tam 50,000,000 byte; eğitimden önce ayrılır |
| Split ayrımı | Normalize edilmiş tam kaynak doküman SHA-256'sı iki split arasında tekil olmalıdır |
| Exact duplicate | Normalize edilmiş yayımlanmış kayıt SHA-256'sı iki split'te de tekil olmalıdır |
| Normalizasyon | Unicode NFC, whitespace → tek ASCII boşluk; transliteration/çeviri yok |
| Kayıt sınırı | En fazla 4096 byte; yalnız whitespace sınırında bölme |
| Son exact sınır | Gerekirse bir **tek** kaynak-contiguous ve UTF-8 geçerli substring; padding, cyclic reuse veya synthetic filler yok |
| Doğrudan PII | Email, muhafazakâr Türkiye telefon ve IPv4 pattern içeren dokümanlar reject edilir |
| Yapısal kalite | En az 96 karakter, alfa/görünür oranı ≥ 0.45, Latin-alfa oranı ≥ 0.70 |

Bu kontroller sadece belgede yazan güçlü garantileri verir. Regex PII filtresi
kapsamlı privacy audit değildir; exact SHA-256 deduplication da fuzzy/semantic
near-duplicate garantisi değildir. Kaynak corpus'un Turkish declaration'ı
provenance'dir; controller bağımsız dil sınıflandırıcısı olduğunu iddia etmez.

## Tam olarak bir epoch ve zaman ölçümü

Workflow'da source hazırlığı bittikten sonra yalnız bir tane çağrı vardır:

```text
vegai_causal_lm train train-9gb.txt cosmos-turkish-causal.vclm ...
```

Bu native program corpus üzerinde iki sıralı işlem yapar:

1. bounded vocabulary heavy-hitter seçimi (model ağırlığı güncellemez);
2. **tam bir** causal transition update geçişi.

Sadece ikinci çağrının hemen çevresindeki native `system_clock` ölçümü
`native_training_epoch_wall_seconds` olur. Throughput'un paydası olarak bu
zaman kullanılır. Vocabulary seçimi, data download/parse/dedup, derleme,
checkpoint serialization, held-out evaluation, generation ve CI orchestration
bu ölçüme dahil değildir. Parent process wall time yalnız audit alanıdır ve
throughput hesabında kullanılmaz.

## Held-out ölçüm ve gerçek cevaplar

Kaynak-doküman-disjoint held-out üzerinde native `evaluate` aşağıdakileri
raporlar:

- eligible known-token transition sayısı;
- Top-1 next-token accuracy;
- bounded successor table içindeki target coverage;
- trigram ve bigram context hit sayıları.

Bunlar n-gram sketch'in local causal continuation ölçüleridir. Perplexity,
factuality, open-ended answer quality veya genel Türkçe yeteneği değildir.

Workflow ayrıca sabit, held-out'tan alınmayan Türkçe istemlerle native
`generate` çağrıları yapar. Artifact `result.json` gerçek cevapları saklar;
workflow summary'de de görür. İnsanlara sunulacak bir sonuç, yalnız o artifact
incelendikten sonra **verbatim** yazılacaktır; önceden örnek üretmek veya
başarılı cevap varsaymak yasaktır.

## Çalıştırma

```bash
# Python source adapter yalnız data/provenance kontrolüdür; model native Fortran'dır.
python3 -m pip install -r requirements-cosmos-9gb.txt
make causal-lm BUILD_DIR=build-cosmos
python3 tools/run_cosmos_turkish_9gb.py \
  --causal-bin ./build-cosmos/vegai_causal_lm \
  --run-dir /ignored/vegai-cosmos-turkish-9gb
```

Gerçek veri egress'i ve 9 GB fiziksel corpus ürettiği için standart PR/CI
çalıştırması değildir. Sadece manuel GitHub Actions workflow'su gerçek evidence
run'ı başlatır. Herhangi bir preflight hatasında `failure.json` yayımlanır;
başarısız run hiçbir şekilde 9 GB eğitim veya model-quality başarısı olarak
sunulmaz.
