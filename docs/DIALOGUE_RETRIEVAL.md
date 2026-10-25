# Veri odakli grounded HDC diyalog retrieval

`vegai_chat`, bir Vegai HDC checkpoint'i ve operatorun sagladigi TSV bilgi
bankasi ile **retrieval tabanli** diyalog yuzeyi saglar. Her gorunen cevap,
TSV'deki response alanindan byte olarak gelir; kaynak kodda cevap, intent veya
sohbet senaryosu bulunmaz.

> ## Dürüst kapsam siniri
>
> Bu modül Transformer/LLM/LMM, next-token generator veya genel dogal dil
> muhakeme sistemi degildir. HDC semantic prototype'lari ile prompt benzerligi
> bulur, kaynak-veri cevabini döndürür ya da grounding esigi saglanmazsa
> **abstain** eder. Daha zeki/güvenilir cevaplar için TSV bilgi bankasini,
> checkpoint'i ve held-out retrieval degerlendirmesini iyilestirmek gerekir;
> model uydurma prose üretmez.

## Moduler sinirlar

Bagimlilik yonu tek taraflidir:

```text
hyper_types + text_tokenizer + hdc_pretraining checkpoint
                         |
                         v
                 dialogue_retrieval
                    |           |
                    v           v
                vegai_chat   operator TSV knowledge base
```

- `src/turkish_text.f90`: strict UTF-8 kontrolü ve Türkçe Latin harflerini
  checkpoint'in ASCII projeksiyonuna deterministik katlayan native Fortran
  sınırı. Response byte'ları dönüştürülmez.
- `src/dialogue_retrieval.f90`: bounded index, deterministic prompt encoding,
  legacy HDC veya opt-in lexical/HDC hybrid reranking, confidence/margin gate
  ve bounded turn history. Corpus training veya checkpoint semantiğini
  değiştirmez.
- `app/chat.f90`: `ask` ve `repl` CLI adaptörü. Cevap üretmez; yalnızca modülün
  sonucunu raporlar.
- `tools/verify_dialogue_retrieval.py`: gerçek derlenmis `vegai_pretrain` ve
  `vegai_chat` ile runtime TSV cevabi, abstention, strict failure ve REPL
  contract'ini denetler.
- `tools/dialogue_evaluation.py`: versioned held-out JSONL setinde stateless
  grounded-prompt accuracy/precision/recall ve abstention metriklerini ölçer;
  checkpoint/KB/binary/source digest'lerini report'a bağlar ve explicit release
  threshold'larını gate eder.
- `app/dialogue_eval.f90` / `vegai_dialogue_eval`: checkpoint ve knowledge TSV
  index'ini bir kez yükleyen all-Fortran, stateless, strict three-column TSV
  batch evaluator. Raw prompt/response yayınlamadan exact retrieval, rendered
  abstention, false grounding ve OOV safe rejection count'larını JSON'a yazar.
- `tools/calibrate_dialogue_abstention_policy.py`: native batch evaluator'ı
  sadece validation source-disjoint setiyle çalıştırır; önceden ilan edilmiş
  hybrid score floor'lar arasında leakage cap'ini geçen en düşük değeri seçer.
  Frozen test set bu aracın girdisi değildir.

Bu ayrim chat UX'ini training/continual journal/production control-plane'den
ayri tutar. Dialogue index checkpoint'e yazilmaz; bilgi bankasinin versionu,
SHA-256'si, lisansi ve retention policy'si operator tarafindan kayda
alinmalidir.

## Bilgi bankasi sozlesmesi

Her nonblank fiziksel satir tam olarak bir ASCII TAB ayiraciyla yazilir:

```text
prompt<TAB>response
```

Ornek:

```text
okul ders	okul ders kitap ogrenci ogretmen ile ilgili bilgi
hastane doktor	hastane doktor hasta hemsire tedavi ile ilgili bilgi
```

Runtime fail-closed kuralları:

- yalnız LF ve `<= 8.192` byte fiziksel kayıt;
- ayrac **tek ASCII TAB**'dır; CR/CRLF, ikinci TAB, boş alan ve C0/C1 kontrol
  karakterleri reddedilir;
- prompt en çok 1.024, response en çok 2.048 **UTF-8 byte** olabilir; UTF-8
  biçimi doğrulanır, oversized `--prompt` veya REPL physical record'u truncate
  edilmeden reddedilir;
- native prompt sınırı `ÇçĞğİıÖöŞşÜü` ile Turkish circumflex
  `ÂâÎîÛû` harflerini sırasıyla ASCII `c/g/i/o/s/u/a/i/u` projeksiyonuna
  katlar; public source'larda görülen Romanian `Șș` varyantı da `s` olur;
  common typographic apostrophe/dash ve
  NBSP için de açık katlama vardır. Desteklenmeyen Unicode prompt karakteri
  sessizce düşürülmez, fail-closed olur;
- response alanı geçerli printable UTF-8 olarak **verbatim** saklanır ve
  döndürülür; doğal görünen metin yalnız operatorun yayınladığı veridir;
- bilgi bankası prompt'undaki **her** canonical token yüklenmiş checkpoint'in
  active vocabulary'sinde olmalıdır; OOV anahtar sessizce indexlenmez;
- aynı HDC prompt embedding'i (response aynı olsa bile) ikinci kez yayınlanamaz;
- `--max-entries` kapasitesi aşılırsa index büyümez;
- herhangi bir TSV publication/load hatasında partial index destroy edilir ve
  `entries_loaded=0` olur; düzeltilmiş revision için operator yeniden init/load
  etmelidir.

Cevap alani tokenization veya execution icin kullanilmaz; display edilecek
operator verisidir. Bu nedenle bilgi bankasini guvenilir/veri-yonetisiminden
gecmis bir kaynak olarak ele alin. Bir response içindeki instruction benzeri
metin executable degildir, fakat kullaniciya oldugu gibi gösterilir.

Checkpoint'in training tokenizer'ı hâlâ ASCII/transliterasyonludur; UTF-8
sınırı bunu değiştirmez veya yeni model bilgisi icat etmez. Bu yüzden upstream
normalizasyon/transliteration sürümü, source license'i ve bilgi-bankası hash'i
kaydedilmelidir. `examples/dialogue/mini_turkish_hdc_dialogue.tsv` tamamen
sentetik, ASCII transliterasyonlu bir integration fixture'dir; gerçek bilgi veya
kalite iddiası değildir.

## Kullanım

Once uygun bir HDC checkpoint egitin:

```bash
cmake -S . -B build -DVEGAI_ENABLE_OPENMP=ON -DVEGAI_CPU_TARGET=portable
cmake --build build --parallel

./build/vegai_pretrain train examples/corpus/mini_turkish_ascii_pretrain.txt /tmp/mini.vgai \
  --max-vocab 128 --min-frequency 8 --window 2 --position-shift 41 \
  --context-mode hierarchical --epochs 2 --strict
```

Tek bir grounded sorgu:

```bash
./build/vegai_chat ask /tmp/mini.vgai examples/dialogue/mini_turkish_hdc_dialogue.tsv \
  --prompt "okul ders" --max-entries 64 --history-turns 0 \
  --min-similarity 0.05 --min-margin 0.01
```

HDC komşuluklarının zayıf olduğu Türkçe biçimbirimli/parafrazlı sorgular için
**opt-in hybrid** policy, checkpoint association ile şeffaf lexical kanıtı
ayrı ayrı raporlar ve yalnız explicit hybrid gate geçerse response döndürür:

```bash
./build/vegai_chat ask /tmp/mini.vgai /srv/kb/turkish-dialogue.tsv \
  --prompt "Öğrencinin kaydı nasıl yapılabilir?" --max-entries 50000 --history-turns 0 \
  --semantic-weight 0.25 --lexical-weight 0.75 \
  --min-hybrid-score 0.65 --min-hybrid-margin 0.03
```

Bu örnekte TSV'deki prompt ve response UTF-8 olabilir; seçenek, cevap
üretmediği için coverage dışındaki soru için hâlâ abstain etmelidir.

Stateful terminal oturumu:

```bash
./build/vegai_chat repl /tmp/mini.vgai examples/dialogue/mini_turkish_hdc_dialogue.tsv \
  --max-entries 64 --history-turns 3 --min-similarity 0.05 --min-margin 0.01
```

`repl` icinde `/quit` yalnız terminal komutudur. Kullanici sorusu indexte yeterli
bicimde grounded degilse CLI `grounded: false` ve `answer unavailable:` verir;
`answer:` satiri yazmaz. Bu bilincli abstention davranisidir. Her call ayrica
known/ignored token sayisini, exact candidate sayisini, best/runner-up bipolar
similarity'yi ve margin'i; baslangic ozeti ise resident index byte estimate'i
ile aktif grounding policy esiklerini yayinlar.

## Retrieval, baglam ve olcek siniri

1. UTF-8 prompt doğrulanır ve desteklenen Türkçe harfleri checkpoint'in ASCII
   tokenizer projection'ına canonicalize edilir. Original UTF-8 source prompt
   provenance/display alanında korunur.
2. Active checkpoint semantic hypervector'leri deterministic weighted majority
   bundle'a çevrilir.
3. Son `--history-turns` user prompt'u varsa, current prompt weight'i ve
   history weight'i ile ikinci bounded bundle oluşturulur. Varsayılan policy
   current=`2`, her history turn=`1`, history capacity=`3`tür; tümü CLI ile
   değiştirilebilir, `--history-turns 0` state'i kapatır.
4. Index tüm entry'leri exact Hamming mesafesiyle tarar. Varsayılan legacy
   ranking yalnız bu bipolar semantic similarity'dir ve
   `--min-similarity`/`--min-margin` kapılarını kullanır.
5. `--lexical-weight > 0` ise her candidate için canonical token overlap,
   exact adjacent bigram ve en az üç-character ortak-root sonrası bounded edit
   / ortak-prefix biçimbirim kanıtı hesaplanır. `semantic_weight` ile normalize edilen hybrid score,
   semantic score ve lexical score ayrı yayımlanır; ranking/gate yalnız
   `--min-hybrid-score`/`--min-hybrid-margin` ile yapılır. CLI'deki legacy
   `grounding margin` her zaman aktif ranker margin'idir (hybrid modda
   `hybrid grounding margin` ile aynıdır); bipolar runner candidate'ın raw
   semantic skoru daha yüksek olabilir. Eski semantic eşikler hibrit moda
   gizlice uygulanmaz.

Bu prefix heuristic Türkçe morfoloji desteği sağlar; anlam eşdeğerliği, factual
truth veya doğal dil anlayışı kanıtı değildir. Türkçe paraphrase/coverage
kalitesi source-disjoint held-out set ile ölçülmeden iddia edilmez.

Exact tarama deterministik ve recall kayipsizdir, fakat bir turn icin `O(N ×
16.384-bit)` maliyetlidir. Bu modul uzak/dağıtık milyarlarca dokuman retrieval
servisi oldugunu iddia etmez. `--max-entries` ile RAM siniri aciktir; index
yaklaşık olarak entry başına 2.048 byte prompt HV + 1.024 byte original
UTF-8 prompt buffer + 1.024 byte canonical projection + 2.048 byte response
buffer + sayaçlar kullanır. CLI `knowledge index resident bytes` ile
veri-duzlemi estimate'ini yayinlar. Daha buyuk deployment'ta index sharding,
external search backend ve versioned retrieval evaluation ayrı bir adapter
karari olmalidir; HDC exact semantics gizlice approximate result olarak
sunulmaz.

`--require-all-known`, live prompt'ta da OOV token goruldugunde fail-closed
ister. Varsayilan daha esnek mod, en az `--min-known-tokens` active token varsa
bilinmeyen filler token'lari sayar ve raporlar; cevap yine similarity/margin
kapisindan gecmek zorundadir.

## Native batch evidence ve validation-only calibration

`vegai_dialogue_eval` yalnız retrieval davranışını ölçer; decoder çağırmaz,
cevap üretmez ve raw evaluation metnini output'a yazmaz. Input satırı tam üç
kolondur:

```text
case-id<TAB>prompt<TAB>expected-prompt
```

Boş `expected-prompt`, expected abstention'dır. Program `--max-cases` aşılırsa
sessizce prefix ölçmez; nonzero exit ile fail eder. No-active-checkpoint-token
sonucu rendered abstention sayılmaz: expected-null case için ayrı
`oov_rejected_abstain_cases`, expected-grounded case için ayrı
`oov_rejected_expected_grounded_cases` yazılır. Böylece OOV retleri gerçek
rendered abstention ile karıştırılmaz.

```bash
./build/vegai_dialogue_eval evaluate /tmp/model.vgai knowledge.tsv validation.tsv \
  --max-entries 5000 --max-cases 2000 --min-similarity -1 --min-margin 0 \
  --semantic-weight .25 --lexical-weight .75 \
  --min-hybrid-score .80 --min-hybrid-margin .03 > validation-native.json
```

Üretim evidence workflow'u doğrudan sabit `.80` kullanmaz. Önce
`calibrate_dialogue_abstention_policy.py` validation TSV'de önceden ilan edilen
floor listesini çalıştırır, strictest candidate'da train exact literal contract
başarısını doğrular ve validation false-grounded-rate cap'ini geçen **en düşük**
floor'u seçer. Düşük floor seçimi potansiyel retrieval coverage'ı gereksiz
azaltmamak içindir; relevance kanıtı değildir. Calibration JSON'u checkpoint,
KB, train/validation TSV ve native report hash'lerini bağlar. Frozen test TSV
bu seçim aracına verilmez; seçimden sonra, önceden ilan edilmiş baseline ile
tek ölçüm için kullanılır.

## Held-out kalite değerlendirmesi ve release gate

Grounding eşiğini yalnız örnek prompt'a bakarak seçmek güvenilirlik kanıtı
değildir. `tools/dialogue_evaluation.py`, gerçek `vegai_chat ask` binary'sini
versioned UTF-8 JSONL held-out case seti üzerinde history kapalı çalıştırır.
Evaluator semantic ve hybrid CLI policy'nin tamamını report'a bağlar; Her case
ya beklenen KB prompt'unu ya da açık `expected_prompt: null` abstention
beklentisini taşır. Report raw prompt/response yerine SHA-256 prompt kimlikleri
veya `--case-hmac-key-file` verildiğinde keyed HMAC-SHA-256 kimlikleri,
checkpoint/KB/binary/evaluation-set digest'leri ve şu metrikleri yazar:

- exact grounded-prompt accuracy, precision ve recall;
- explicit rendered `grounded: false` için abstain precision/recall;
- false grounding, wrong prompt ve false abstention sayıları;
- source-disjoint negative set için optional `--allow-oov-rejections-for-expected-abstention`
  kullanıldığında ayrı `oov_rejected_abstain_cases` ve
  `nonfabrication_recall_including_oov_rejections`. Bu, rendered abstention
  metriğine eklenmez; checkpoint'in prompt'u hiç encode edemediğini açıkça
  raporlar.
- explicit quality gate sonucu.

Hassas held-out setlerde HMAC key'i repository/report dışında saklanır; report
key path'ini redact eder ve yalnız key-byte SHA-256 kimliğini bağlar.

```bash
python3 tools/dialogue_evaluation.py \
  --chat-bin ./build/vegai_chat --checkpoint /tmp/mini.vgai \
  --knowledge examples/dialogue/mini_turkish_hdc_dialogue.tsv \
  --evaluation examples/dialogue/mini_turkish_hdc_evaluation.jsonl \
  --output /tmp/mini-dialogue-quality.json \
  --max-entries 64 --min-similarity 0.90 --min-margin 0.00 \
  --semantic-weight 0.25 --lexical-weight 0.75 \
  --min-hybrid-score 0.65 --min-hybrid-margin 0.03 \
  --min-grounded-prompt-accuracy 1.0 --min-grounded-precision 1.0 \
  --min-grounded-recall 1.0 --min-abstain-precision 1.0 --min-abstain-recall 1.0 \
  --max-false-grounded-cases 0 --max-false-abstain-cases 0
```

Sentetik fixture için `1.0` eşikleri yalnız contract smoke testidir; gerçek
release'te eşikler domain riski, yeterli case sayısı ve ayrı calibration seti
ile belirlenmelidir. Threshold seçimi validation setinde yapılır; aynı frozen
test setinde tuning yapmak leakage'dir. Exit `3`, teknik evaluation tamamlanıp
kalite gate'inin geçmediğini; exit `2`, evaluation contract'inin çalışmadığını
belirtir. Bu katman LLM benchmarkı, factual truth oracle veya safety alignment
kanıtı değildir. Tam audit ve kalite planı
[QUALITY_ASSURANCE.md](QUALITY_ASSURANCE.md)'dedir.

## Test contract'i

CTest'teki `vegai_dialogue_retrieval` gerçek Fortran binary'leriyle:

- runtime TSV tarafindan belirlenen cevabin aynen döndüğünü;
- ayni prompt icin yalniz TSV response'u degistirmenin user-visible cevabi da
  degistirdigini (source-code cevabi olmadigini);
- known ama uygun olmayan prompt'un abstain etmesini;
- OOV user prompt, malformed TSV ve capacity/duplicate prompt error'larini;
- stateful `repl` yolunu;
- native Fortran property testinde bounded session ring, strict UTF-8 TSV load,
  Türkçe diacritic folding, response byte preservation, deterministic hybrid
  lexical/morphology reranking, exact legacy retrieval ve reset davranışını
  denetler;
- `vegai_dialogue_evaluation`, real binaries ile held-out exact grounding,
  expected abstention, privacy-preserving report, malformed evaluation schema
  ve intentionally failed release-gate davranisini denetler;
- `vegai_native_dialogue_evaluator`, gerçek Fortran train/checkpoint ve strict
  TSV stream ile stateless batch count'larını, OOV'nin ayrı sayılmasını,
  raw-text sızıntısı olmamasını ve `--max-cases` fail-closed davranışını;
- `vegai_dialogue_abstention_calibration`, fake evaluator ABI ile test setini
  girdi kabul etmeyen validation-only lowest-safe-floor seçim kuralını
  denetler.

Bu testler retrieval/data-contract dogrulamasidir. Held-out natural-language
kalitesi, güvenlik veya factual accuracy icin versioned bilgi bankasi, source
review, held-out query seti ve insan/task-specific degerlendirme gerekir.
