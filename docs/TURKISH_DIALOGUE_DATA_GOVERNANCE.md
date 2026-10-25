# Türkçe diyalog verisi ve kalite yönetimi

Bu belge, Vegai'nin Türkçe konuşma kalitesini artırma çalışmasında kullanılacak
verinin nasıl seçileceğini, native Fortran retrieval path'ine nasıl
dönüştürüleceğini ve hangi sonuçların **iddia edilmemesi** gerektiğini tanımlar.

> ## Model sınırı
>
> `vegai_pretrain` HDC/VSA contextual prototype eğitir; causal next-token
> kaybı, logits, decoder veya sampling path'i yoktur. `vegai_chat` ise yalnız
> TSV'de yayınlanmış literal response'u retrieval ile döndürür. Bu dokümandaki
> veri paketi, coverage'i ve grounded retrieval'i iyileştirmek içindir;
> **serbest biçimli doğal Türkçe üretim eğitimi değildir.**
>
> Çok doğal, açık uçlu bir sohbet için gerekli ayrı native causal-model
> sözleşmesi ve gerçek held-out kanıtı
> [LMM_SCOPE_AND_GAP_ANALYSIS.md](LMM_SCOPE_AND_GAP_ANALYSIS.md)'nde belirtilen
> eksikleri kapatmadan “generative chat” iddiası yapılamaz.

## Seçilmiş, lisansı doğrulanmış başlangıç adayı

Aşağıdaki kaynak küçük, denetlenebilir bir **retrieval baseline** olarak
incelenmiştir. Repository'ye corpus kopyası eklenmez; her gerçek run tam
revision ve SHA-256 ile bağlanır.

| Alan | Değer |
|---|---|
| Kaynak | [`3nesdeniz/turkish-daily-dialogues-5k`](https://github.com/3nesdeniz/turkish-daily-dialogues-5k) |
| Pin | `ccd9f05c2f97684bbd9a55d818528da9dfb6bd5a` |
| Kaynağın beyan ettiği lisans | `CC-BY-4.0` |
| Atıf | *Turkish Daily Dialogues 5K*, Enes Deniz, CC BY 4.0; kaynak repository/commit ile birlikte belirtilmelidir |
| Kaynağın beyan ettiği kapsam | 5.000, 4/6/8 dönüşlü Türkçe günlük yaşam diyaloğu; train 3.950, validation 525, test 525 konuşma |
| Kaynağın beyan ettiği köken | sentetik, AI-assisted scenario library + deterministic offline generation; gerçek özel konuşma olmadığı beyan edilir |
| Uygun kullanım | UTF-8/native-boundary, retrieval coverage, split/provenance ve abstention pipeline baseline'ı |
| Uygun olmayan iddia | insan konuşması temsilidir, tek başına yeterli doğal sohbet verisidir, factual/high-stakes güvenlik kanıtıdır |

Bu seçim, kaynak README'sindeki lisans, `MANIFEST.json`, split açıklaması ve
`SHA256SUMS` kontrol edilerek yapılmıştır. Kaynak sahibi beyanı bağımsız hukuki
relisanslama değildir: downstream kullanıcı kendi kullanım koşullarını ve CC BY
atıf yükümlülüğünü doğrulamalıdır.

Başlangıç kaynağı 5.000 sentetik konuşma olduğu için “yeterli veri” hedefini
karşılamaz. Gerçek büyütme için lisansı açık/izinli, provenance'ı kaydedilmiş,
PII/consent politikası olan ve domain çeşitliliği ölçülmüş ek kaynaklar gerekir.
Scrape edilmiş özel konuşma, lisansı belirsiz forum kaydı veya yalnız otomatik
üretilmiş text bu açığı dürüstçe kapatmaz.

## Tekrarlanabilir package builder

`tools/build_turkish_dialogue_retrieval_corpus.py` network kullanmaz ve source
indirmez. Operator önce exact source revision'ı local olarak alır, upstream
manifest/hash kontrolünü yapar, sonra üç split dosyasını explicit verir:

```bash
# Source checkout must already be pinned and its own hashes verified.
git clone https://github.com/3nesdeniz/turkish-daily-dialogues-5k /tmp/trdd5k
cd /tmp/trdd5k && git checkout ccd9f05c2f97684bbd9a55d818528da9dfb6bd5a
sha256sum --check SHA256SUMS

cd /path/to/Vegai
python3 tools/build_turkish_dialogue_retrieval_corpus.py \
  --train-jsonl /tmp/trdd5k/data/train.jsonl \
  --validation-jsonl /tmp/trdd5k/data/validation.jsonl \
  --test-jsonl /tmp/trdd5k/data/test.jsonl \
  --output-dir /tmp/vegai-trdd5k-package \
  --source-repository https://github.com/3nesdeniz/turkish-daily-dialogues-5k \
  --source-revision ccd9f05c2f97684bbd9a55d818528da9dfb6bd5a \
  --source-license CC-BY-4.0 \
  --attribution 'Turkish Daily Dialogues 5K by Enes Deniz, CC BY 4.0' \
  --source-nature synthetic-ai-assisted-deterministic
```

Tool şunları fail-closed doğrular:

- input UTF-8 JSONL/LF, bounded source line, message role sırası,
  control/TAB/CR içermeyen bounded display text;
- train/validation/test arasında `conversation_id` benzersizliği;
- exact source file SHA-256/byte identity ve açık repository/revision/license/
  attribution metadata;
- Türkçe `ÇçĞğİıÖöŞşÜü`, Turkish circumflex `ÂâÎîÛû` ve public-source
  `Șș` varyantı → native ASCII checkpoint projection;
- train canonical prompt için yalnız aynı literal response tekrarlarının
  deterministic dedupe edilmesi; çelişen response fail-closed;
- existing package'ın sessizce değiştirilmemesi.

Output package:

| Dosya | Kullanım |
|---|---|
| `knowledge_train.tsv` | yalnız train split'ten UTF-8 `prompt<TAB>literal response`; native `vegai_chat`'e yüklenir |
| `pretrain_train_ascii.txt` | prompt ve response turn'lerinin strict ASCII projection'ı; native `vegai_pretrain` input'u |
| `train_exact_retrieval_contract.jsonl` | train TSV key'lerinin literal lookup integrity suite'i; held-out/generalization metriği değildir |
| `*_evidence_sample.jsonl` | full setten SHA-256-ranked deterministic bounded subset; evaluator prefix'i sessizce ölçmesin diye CI evidence için kullanılır |
| `validation_source_disjoint_abstention.jsonl` | validation konuşmalarından `expected_prompt: null` seti |
| `test_source_disjoint_abstention.jsonl` | frozen test konuşmalarından `expected_prompt: null` seti |
| `manifest.json` | source/output identity, dedupe sayıları, split isolation ve açık non-generative scope |

Python builder yalnız data/package orchestration'dır. Eğitilen HDC checkpoint,
UTF-8 query normalize/lexical hybrid retrieval ve response rendering
`src/turkish_text.f90`, `src/dialogue_retrieval.f90`, `app/pretrain.f90` ve
`app/chat.f90` içindeki native Fortran path'tedir. Bu ayrım “Python tool var,
dolayısıyla proje generative model oldu” gibi yanlış bir sonucu engeller.

## Split ve evaluation anlamı

Source'un scenario-family split'i source ID düzeyinde korunur. Train TSV'de
validation/test response'u bulunmaz. Bu nedenle generated validation/test
JSONL case'leri `expected_prompt: null` taşır:

- doğru `grounded: false` sonucu **source-disjoint rendered abstention** başarısıdır;
- checkpoint'te hiç active token yoksa native CLI fail-closed OOV rejection
  verir. Evaluator'ın explicit `--allow-oov-rejections-for-expected-abstention`
  modu bunu ayrı `oov_rejected_abstain_cases` olarak raporlar; rendered
  abstention recall'a gizlice katmaz;
- `grounded: true` sonucu aynı topic/kelime yüzünden yanlış grounding olabilir;
- bunlar unseen assistant response'un “doğru cevabı” için label **değildir**.

Bir sistemin gerçekten ilgili cevabı seçebildiğini ölçmek için ayrıca:

1. train KB entry'lerine insan review ile eşlenmiş, source-disjoint Türkçe
   paraphrase sorguları;
2. calibration split ile seçilmiş ve frozen test'e dokunmayan hybrid thresholds;
3. exact source prompt, grounded response relevance ve abstention için ayrı
   sınıflandırma etiketleri;
4. source/diyalog-family overlap, duplicate ve template leakage denetimi;
5. human Turkish naturalness/factuality değerlendirmesi

gereklidir. `tools/dialogue_evaluation.py` UTF-8 held-out cases ile semantic ve
hybrid policy'nin binary/checkpoint/KB/evaluation digest'lerini bağlar. Mevcut
99M Wikipedia semantic score'larının zayıf olduğuna dair gerçek kanıt nedeniyle,
threshold'lar bu frozen değerlendirme olmadan “iyileşme” diye sunulamaz.

## Gerçek run için güvenli başlangıç komutları

Builder output'undaki token türü sayısı, `--max-vocab` seçiminin altında
olmalıdır; aksi halde native TSV loader OOV key yüzünden fail-closed olur.
Örnek baseline (yürütülen süre yalnız train child timing raporundan
alınmalıdır):

```bash
./build/vegai_pretrain train /tmp/vegai-trdd5k-package/pretrain_train_ascii.txt \
  /tmp/trdd5k.vgai --max-vocab 4096 --min-frequency 1 --window 4 \
  --position-shift 73 --context-mode resonant --epochs 2 --strict

./build/vegai_chat ask /tmp/trdd5k.vgai /tmp/vegai-trdd5k-package/knowledge_train.tsv \
  --prompt 'Öğrencinin kaydı nasıl yapılabilir?' --max-entries 5000 --history-turns 0 \
  --semantic-weight 0.25 --lexical-weight 0.75 \
  --min-hybrid-score 0.65 --min-hybrid-margin 0.03
```

İkinci komuttaki response yalnız TSV'den gelir. Çıktıdaki `best bipolar
similarity`, `best lexical similarity`, `hybrid grounding score`, runner-up ve
margin birlikte saklanmalıdır; yalnız seçilmiş güzel örnekler raporlanmaz.

## “Çok doğal sohbet”e geçiş kapısı

Grounded retrieval, yeterli ve iyi yazılmış licensed dialogue rows ile kapsadığı
alanlarda daha doğal görünebilir. Ancak kapsam dışı yeni prose üretemez. Genel,
açık uçlu Türkçe sohbet hedefi için release öncesi aşağıdaki minimum evidence
ayrı ayrı gerekir:

- gerçek native causal/token decoder implementation ve reproducible model
  artifact/tokenizer identity;
- licence/provenance/PII governance ile büyük, çeşitli, source-disjoint Türkçe
  train/validation/test data;
- next-token loss/perplexity yanında dialogue relevance, factual grounding,
  abstention/safety ve Turkish human preference değerlendirmesi;
- contamination/retrieval leakage denetimi ve model/benchmark version binding;
- calibration ve final test ayrımı; final set üzerinde repeated threshold
  tuning yapılmaması.

Bunlar olmadan bu repo, retrieval kalitesini geliştirmiş olabilir ama “çok doğal
free-form Türkçe sohbet eden generative model” olarak sunulamaz.
