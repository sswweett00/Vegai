# Gerçek Türkçe Wikipedia ile 9M native-token RESONANT deneyi

Bu belge, sentetik fixture yerine açık lisanslı gerçek Türkçe veriyle çalışacak
tekrar üretilebilir deney sözleşmesidir. Çalıştırma noktası manuel GitHub Actions
workflow'udur:

```text
.github/workflows/turkish-wikipedia-9m-evidence.yml
```

Workflow tamamlandığında küçük provenance/retrieval artifact'i ve ayrı native
checkpoint artifact'i üretir. Sonuçlar, yalnız o run'ın immutable
`experiment-report.json` dosyasından belgelenmelidir; başarısız veya yarım bir
run kalite sonucu gibi raporlanmaz.

## Kaynaklar ve lisans

| Amaç | Sabit kaynak | Lisans | Bağlanan kimlik |
|---|---|---|---|
| Eğitim corpus'u | [`tdd-ai/trwiki-67`](https://github.com/tdd-ai/trwiki-67), Türkçe Wikipedia'nın 20 Temmuz 2021 dump'ından üretilmiş raw text | `CC-BY-SA-4.0` | workflow, repo `HEAD` revision'ı, repo `LICENSE` SHA-256'sı, Zenodo arşiv SHA-256'sı ve kullanılan raw train split SHA-256'sını kaydeder |
| Arşiv | `https://zenodo.org/record/5213891/files/trwiki-67.tar.bz2?download=1` | yukarıdaki dataset lisansı | indirilen `.tar.bz2` byte sayısı ve SHA-256 |
| Grounded TSV | [`Aliumutcan/Turkish_NLP_FAQ_DATASET`](https://github.com/Aliumutcan/Turkish_NLP_FAQ_DATASET), `dataset_v1.txt` | `MIT` | repo revision'ı, `LICENSE` SHA-256'sı, dataset SHA-256'sı, seçilmiş satır no ve kaynak web URL'si |

`trwiki-67` proje README'si raw train split'i 63.5 milyon sözcük olarak
açıklar. Bu deney bütün split'i kopyalamaz veya sözleşme fixture'ını gerçek veri
olarak göstermez: açık kaynak split üzerinde belirlenmiş bir filtre çalıştırır
ve son çıktıdaki native token sayısını **tam 9,000,000** olarak doğrular.

## Native ASCII sözleşmesi ve dönüştürme

Mevcut native Fortran tokenizer'ı byte-ASCII'dir. Türkçe UTF-8 byte'larını
sessiz ayırıcılar olarak kullanmak yerine `tr-ascii-v1` adlı kayıtlı projection
uygulanır:

1. `ç/Ç → c/C`, `ğ/Ğ → g/G`, `ı/İ → i/I`, `ö/Ö → o/O`, `ş/Ş → s/S`,
   `ü/Ü → u/U`; tipografik apostrophe'lar ASCII apostrophe olur.
2. Kalan metin Unicode `NFKD` decomposition ile ASCII'ye project edilir;
   kalan combining/non-ASCII karakterler atılır ve ASCII harfler lowercase
   edilir.
3. 48 byte'tan uzun token'lar truncate edilmez, atılır.
4. Tüm raw train split üzerinde frekans sayılır; en sık 32,768 native-uyumlu
   token (frekans azalan, lexical tie-break) seçilir.
5. Kaynak sırası korunur. Seçili olmayan her token record boundary'dir;
   dolayısıyla filtrelenmiş kelimelerin iki tarafı yapay biçimde komşu olmaz.
   Her çıktı record'u en çok 128 token ve 8,192 byte'tır.
6. Seçilmiş gerçek kaynak token'larından 9,000,000 tane yazılır. `--max-vocab
   32768 --min-frequency 1` ile her yazılmış token checkpoint'in active
   vocabulary'sine sığar. Native eğitim metrikleri hem `raw tokens` hem
   `active tokens` için `9,000,000 × epoch` ve `out-of-vocabulary tokens = 0`
   doğrulamasını geçmek zorundadır.

Bu projection Türkçe karakter ayrımlarını native ASCII yüzeyinde taşımaz;
özellikle `i/ı` veya `o/ö` ayrımı kaybolabilir. Bu yüzden artifact, kaynak,
dönüştürücü kimliği, çıktı SHA-256 ve strict telemetry'yi birlikte taşır; bu
run Unicode-Türkçe tokenizer benchmarkı veya dil kusursuzluğu iddiası değildir.

## Eğitim protokolü

Native Fortran komutu şuna eşdeğerdir:

```bash
OMP_NUM_THREADS=2 ./vegai_pretrain train TURKISH_WIKIPEDIA_9M_ASCII.txt checkpoint.vgai \
  --max-vocab 32768 --min-frequency 1 --window 2 --position-shift 41 \
  --context-mode resonant --epochs 2 --strict \
  --parallel-context-build --parallel-min-contexts 32
```

İki epoch bilinçlidir: RESONANT'ın ilk epoch'u HYBRID ile aynı başlangıç
semantic düzleminden çalışır; ikinci epoch yalnız önceki epoch sonunda commit
edilmiş frozen semantic snapshot'ı topical feedback olarak kullanabilir.
`train_process_wall_seconds` yalnız bu tek `vegai_pretrain train` child
process'ini kapsar. İndirme, tar açma, normalizasyon, frekans seçimi,
derleme, checkpoint hash'i, inspect, TSV kurma, chat veya CI süreye dahil
değildir.

## On bir checkpoint semantic sonucu

`vegai_pretrain inspect --query-file` gerçek checkpoint üzerinde 11 active
Türkçe-ASCII token seçer. Önce adlandırılmış probe sırası denenir
(`turkiye`, `istanbul`, `ankara`, `tarih`, ...); eksik kalırsa frequency
azalan, stopword olmayan deterministic fallback seçilir. Her satır şu gerçek
çıktıları taşır:

```text
query, nearest_semantic_neighbour, bipolar_similarity
```

Bu, self-excluded nearest-neighbour inspection'dır. Held-out semantic accuracy,
factual doğruluk, serbest yanıt üretimi veya genel dil anlama ölçümü değildir.

## On bir grounded TSV yanıtı

FAQ corpus'u eğitim corpus'undan ayrı bir Git revision/lisans/provenance
zinciridir. Controller, 11 ayrı FAQ row seçer; prompt'taki bütün tokenların
checkpoint vocab'unda olduğunu doğrular ve kaynak soru tamamen kapsanmadıysa
bunu `prompt_projection` alanında açıkça işaretler. Yanıt alanı kaynak cevap
metninin printable ASCII normalizasyonudur.

`vegai_chat ask` her prompt için `--require-all-known`, similarity `0.95` ve
margin `0.01` ile çalışır. Artifact şunları kaydeder:

```text
prompt, matched_prompt, best_bipolar_similarity,
runner_up_bipolar_similarity, grounding_margin, grounded, answer,
source_url, source_row
```

`answer`, TSV'deki field'in aynen geri alınmış biçimidir. `vegai_chat` bir
LLM, next-token veya serbest prose generator değildir: eşik/margin geçmezse
abstain eder. Bu nedenle on bir başarılı exact TSV retrieval, kaynak TSV'nin
factual doğruluğunu ya da geniş sohbet yeteneğini tek başına kanıtlamaz.

## Run kaydı

Bu commit sadece reproducible protocol'ü ekler. Gerçek run tamamlandıktan sonra
şu alanlar artifact'ten buraya eklenir:

- Actions run URL/ID ve commit SHA;
- raw arşiv, raw split, normalized 9M corpus ve checkpoint SHA-256;
- native eğitim metriği ve train-only süre/throughput;
- 11 semantic nearest-neighbour satırı;
- 11 grounded TSV prompt/answer, similarity, runner-up ve margin satırı.

Sonuçlar bir sonraki belge güncellemesinde yalnız kesin artifact değerleriyle
doldurulacaktır; tahmin veya önceden yazılmış başarı değeri kullanılmaz.
