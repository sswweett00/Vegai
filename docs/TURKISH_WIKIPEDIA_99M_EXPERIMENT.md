# Gerçek Türkçe Wikipedia ile 99M native-token eğitim ve inference deneyi

Bu protokol, sentetik fixture veya corpus tekrarını gerçek eğitim sonucu olarak
sunmaz. Amaç, versioned resmi Türkçe Wikipedia dump'ından **tam 99,000,000**
native-tokenlik özgün bir train akışı üretmek, iki epoch native Fortran
RESONANT eğitimi çalıştırmak ve gerçek checkpoint inference çıktısını ölçmektir.

Çalıştırma workflow'u:

```text
.github/workflows/turkish-wikipedia-99m-evidence.yml
```

Başarılı run dışındaki hiçbir sayı sonuç olarak raporlanmaz. Corpus yetersizse
controller başarısız olur; aynı source satırlarını yeniden kullanarak 99M'e
doldurmaz.

## Kaynak, lisans ve provenance

| Amaç | Kaynak | Lisans / kimlik |
|---|---|---|
| Eğitim ve held-out makale havuzu | Resmi [Turkish Wikipedia pages-articles dump](https://dumps.wikimedia.org/trwiki/) | `CC-BY-SA-4.0`; workflow o an tamamlanmış `dumpstatus.json` revision'ını, exact archive URL'sini, archive SHA-256'sını ve dump-status SHA-256'sını kaydeder |
| Metin extraction | [`attardi/wikiextractor`](https://github.com/attardi/wikiextractor) commit `eeaae048eaa3057c9cd5dfb992eb42104a7a5174` | Workflow, Python 3.11+ ile doğrulanmış commit'i kullanır; `--json --no-templates --discard_empty` modunu raporlar |
| Grounded response knowledge base | [`Aliumutcan/Turkish_NLP_FAQ_DATASET`](https://github.com/Aliumutcan/Turkish_NLP_FAQ_DATASET), `dataset_v1.txt` | `MIT`; revision, LICENSE SHA-256, data-file SHA-256, kaynak URL ve row numarası raporda bulunur |

Corpus veya checkpoint repository'ye commit edilmez. Artifact sadece doğrulama
manifestleri, transcriptler, retrieval çıktıları ve checkpoint kimliğini taşır.

## Article-level gerçek held-out bölümü

Extraction sonrası her gerçek Wikipedia article'ı şu kararlı anahtarla tek bir
partition'a girer:

```text
SHA-256("vegai-trwiki-page-partition-v1\0" + page_id + "\0" + title) mod 100
```

Bucket `0` held-out; diğer 99 bucket train partition'dır. Held-out page text'i:

- vocabulary frekans seçimine girmez;
- train corpus'a yazılmaz;
- native prototype update alamaz;
- sadece checkpoint tamamlandıktan sonra inference değerlendirmesinde okunur.

Bu, page kimliği/titlesi düzeyinde ayrık train/test bölümü sağlar. Wikipedia'nın
kendi makaleleri arasında doğal alıntı veya tekrar olabileceği için bunu mutlak
semantic leakage yokluğu diye sunmuyoruz.

## Native tokenizer ve train sözleşmesi

Native tokenizer ASCII olduğundan `tr-ascii-v1` projection uygulanır:

1. `ç/Ç → c/C`, `ğ/Ğ → g/G`, `ı/İ → i/I`, `ö/Ö → o/O`, `ş/Ş → s/S`,
   `ü/Ü → u/U`; tipografik apostrophe'lar ASCII apostrophe olur.
2. Metin NFKD decomposition sonrası ASCII'ye project edilir, küçük harfe iner;
   48 byte üstü token truncate edilmez, atılır.
3. Yalnız train partition üzerinde global frequency ile en sık 32,768 token
   seçilir. Seçili olmayan token ve page sınırları context boundary'dir.
4. Her emitted record 2–128 token ve en çok 8,192 byte olmalıdır. Tek-token
   segmentler atılır; bu sayede `active tokens` yalnız lexical telemetry değil,
   contextual prototype-update sayısıyla aynı olur.
5. Tam 99,000,000 token çıktı doğrulanır. Her native epoch için `raw tokens`,
   `active tokens` ve `prototype updates` tam 99,000,000 olmalı; OOV, non-ASCII,
   CRLF, truncation ve overlong telemetry sıfır kalmalıdır.

Native eğitim komutu şu yapıdadır:

```bash
OMP_NUM_THREADS=4 ./vegai_pretrain train TURKISH_WIKIPEDIA_99M_ASCII.txt checkpoint.vgai \
  --max-vocab 32768 --min-frequency 1 --window 2 --position-shift 41 \
  --context-mode resonant --epochs 2 --strict \
  --parallel-context-build --parallel-min-contexts 32
```

`train_process_wall_seconds` yalnız native `vegai_pretrain train` child
process'ini kapsar. İndirme, extraction, normalization, vocabulary seçimi,
derleme, inference, artifact ve CI süreleri train throughput'e dahil edilmez.

## Gerçek inference ve dar kalite ölçümü

Eğitimden sonra native `vegai_pretrain inspect --query-file` ile active
vocabulary'den 128 deterministic, non-stopword query için gerçek checkpoint
nearest-neighbour inference yapılır. Held-out article partition içinde aynı
`window=2` ile gerçek local co-occurrence sayımları elde edilir.

Her eligible query için held-out expected association aşağıdaki deterministic
skorla seçilir:

```text
heldout_pair_count(query, neighbour) / heldout_occurrences(neighbour)
```

En az 12 held-out query occurrence ve en az 3 pair occurrence gerekir. Rapor:

- held-out association ile exact Top-1 agreement;
- checkpoint prediction'ın held-out context penceresinde gözlenme oranı;
- held-out association rank'ine göre MRR;
- her query'nin gerçek predicted/expected token, count ve rank bilgisi

yazar. Bu ölçüm gerçek model inference'ıdır; ancak **insan anotasyonlu semantic
similarity benchmarkı, LLM evaluation, factuality veya genel Türkçe NLP puanı
değildir**. Sonuçlar bu dar kapsamla yorumlanır.

Ayrıca 11 semantic nearest-neighbour output ve bağımsız FAQ TSV'sinden native
`vegai_chat` ile 11 literal grounded response kaydedilir. FAQ response'ları
serbest metin üretimi değildir.

## Automation sınırı

Model eğitimi, checkpoint inspect ve chat retrieval native Fortran
executable'larıdır. Wikipedia dump retrieval/extraction, provenance ve artifact
orchestration stdlib Python controller ile external WikiExtractor çağrısından
oluşur. Bu nedenle bu deney bütün otomasyonun Fortran olduğu iddiasını taşımaz;
native model/inference ölçümü ile orchestration katmanı açıkça ayrılmıştır.

## Sonuç kaydı

Başarılı immutable artifact oluştuğunda gerçek SHA-256 değerleri, train-only
süre, 128-query held-out metrics, 11 semantic sonuç ve 11 grounded response
ayrı sonuç belgesine eklenecektir. Tahmini veya fixture değeri eklenmez.
