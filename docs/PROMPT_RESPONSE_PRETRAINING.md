# 600 MB prompt/response ön-eğitimi ve ölçümü

Bu belge, sabit bütçeli bağlam motorunu **600 MB'lık bir prompt/response
korpusu** üzerinde eğiten ve ardından **prompt → yanıt** davranışını ölçen boru
hattını anlatır. Amaç dil bilgisi iddiası değil; mimarinin veriyle ne yaptığını
ölçülebilir kılmaktır.

İlgili sözleşme: [`FIXED_BUDGET_CONTEXT_ENGINE.md`](FIXED_BUDGET_CONTEXT_ENGINE.md).
Oradaki ufuk bağımsızlığı, üç kanal, gradyan ve kontrol noktası kuralları burada
da geçerlidir.

## 1. Neden sentetik korpus

Depoda 600 MB lisanslı Türkçe metin **yok** ve bu ortamın Wikipedia dökümüne
erişimi de yok. Bu yüzden korpus uydurulmuş bir döküm değil, **tohumdan tam
olarak yeniden üretilebilen** yapılandırılmış bir veri kümesidir:

- `tools/build_prompt_response_corpus.py` korpusu üretir,
- her koşuda manifest yazar: **istenen bayt sayısı, gerçek bayt sayısı, SHA-256,
  belge sayısı, tüm olgu bağları (subject → answer) ve probe değerleri**,
- `verify` adımı dosyayı yeniden okuyup boyutu ve SHA-256'yı manifestle
  karşılaştırır; uyuşmazsa koşu durur.

Üretilen metin Türkçe biçimli sahte sözcüklerden oluşur (açık hecelerden
birleştirilir), yani **dünya bilgisi taşımaz**. Taşıdığı şey ölçülebilir üç
görevdir:

| Görev | Cümle kalıbı | Neyi ölçer |
| --- | --- | --- |
| `fact` | `Soru: X nedir? Cevap: Y.` | Akış boyunca taşınan bağ: X'i görüp Y'yi hatırlamak |
| `copy` | `Tekrar et: W. Sonuç: W.` | Bağlam içi kopyalama (ezber değil) |
| `next` | `Say: abcdef. Sonra: bcdefg.` | Yerel bir kural: bir harf kaydırma |

Her belgede 6 olgu cümlesi, dolgu metni, bir kopyalama ve bir kaydırma görevi
vardır; olgular belge başına deterministik bir akışla karıştırılır, böylece aynı
olgu düzenli aralıklarla değil, düzensiz konumlarda geçer. Bu, "hatırlama"yı
sabit konumlu bir arama olmaktan çıkarır.

Varsayılanlar: **600 000 000 bayt**, tohum **20261028**, **2048** olgu bağı.

```
python3 tools/build_prompt_response_corpus.py generate corpora/prompt_600mb.txt \
  --bytes 600000000 --facts 2048 --probe-out corpora/probe.tsv --overwrite
python3 tools/build_prompt_response_corpus.py verify corpora/prompt_600mb.txt
```

## 2. Tokenizasyon: bayt = token

Motorun sözlük varsayılanı 256'dır, korpus ise UTF-8 metindir. Bu yüzden
**token kimliği = bayt değeri**: tokenizer yok, sözlük dosyası yok, eğitim ile
değerlendirme arasında sözlük kayması **olamaz**. Manifestte yazan bayt sayısı,
modelin gördüğü token sayısıdır.

Türkçe harfler UTF-8'de iki bayttır; `app/prompt_engine.f90` bunu
`transfer`/`iand` ile ham bayt olarak ele alır (`byte_of`, `byte_character`),
`iachar`/`achar`'ın 7 bit varsayımına güvenmez.

## 3. Eğitim: dosyadan akış

`vegai_prompt_engine train` korpusu **belleğe almaz**:

1. `chunk + 1` bayt, dosya içinde açık bir konumdan okunur,
2. bu blok tek bir optimizasyon adımı üretir (`fbc_train_stream`),
3. konum `chunk` bayt ilerler; bloğun son baytı bir sonraki bloğun ilk baytı
   olarak yeniden okunur, böylece **tahmin kayması blok sınırında bozulmaz**,
4. yinelemeli durum (bellek kanalları, yuvalar, yerel halka) bloklar arasında
   **sıfırlanmaz**; bellek dosya boyunca taşınır.

Son blok kısaltılabilir: bir aralığın son baytı hiçbir zaman tahmin hedefi
olamayacağı için aralık `bayt - 1` uzunluğundadır ve bu dürüstçe raporlanır.

Koşuyu durduran her sınır açıktır ve raporlanır:

| Seçenek | Anlamı |
| --- | --- |
| `--bytes N` | Eğitimin **bitiş** baytı (0 = dosya sonu) |
| `--start-byte N` | Bu oturumun başlayacağı bayt (zincirleme için) |
| `--max-updates N` | Güncelleme sayısı sınırı |
| `--max-seconds S` | Duvar saati sınırı (oturum uzunluğu) |
| `--resume FILE` | Kontrol noktasından devam (model + optimizer + sayaçlar) |
| `--progress-every N` | İlerleme satırı sıklığı |

Her oturum `CORPUS_TRAIN_JSON { ... }` bloğu basar; blok **geçerli JSON**'dur
(bu, bir hata sınıfını ortadan kaldırdı: Fortran `L1` ile `T`/`F`, `F0.x` ile
`.000` yazıyor ve virgülleri koymuyordu).

## 4. Kanıt iş akışı

`.github/workflows/prompt-response-600mb.yml`:

1. gfortran kurar, motoru, sözleşme testlerini ve korpus CLI'sini derler,
2. **sözleşme testlerini çalıştırır** (gradyan sondaları, ufuk değişmezliği,
   kontrol noktası gidiş-dönüşü),
3. korpusu üretir ve doğrular (boyut + SHA-256),
4. eğitimi **kontrol noktalı oturumlar** hâlinde yürütür; her oturum
   `--max-seconds` ile sınırlıdır ve bir sonraki oturum kaldığı bayttan devam eder,
5. `tools/summarise_prompt_response_run.py` ile oturum raporlarını doğrular ve
   toplar (eksik alan veya `T`/`F` görürse **hata** verir),
6. `filter-probe` ile yalnızca **eğitilen bayt aralığında gerçekten geçen**
   olguları seçer; görülmemiş olgular ölçümden çıkarılır ve kaç tanesinin
   çıkarıldığı raporlanır,
7. `probe` ve `ask` ile prompt → yanıt ölçümünü yapar,
8. tüm sayıları `tools/report_prompt_response_evidence.py` ile tek blok hâlinde
   yazar ve **annotation** olarak da yayınlar (artifact deposu okunamasa bile
   sonuç görünür kalır).

Workflow `workflow_dispatch` ile parametre alır. Bu ortamda dispatch yetkisi
yoktur (HTTP 403), bu yüzden **gerçek koşu `ci/prompt-600mb.json` dosyasıyla
istenir**: push geldiğinde workflow bu dosyayı okur; dosya yoksa beş dakikalık
bir duman testi koşar. Aynı dosya `resume_run_id` / `resume_start_byte` alanları
ile **önceki bir koşunun kontrol noktasından devam** edebilir; devam ederken
korpus SHA-256'sı eşleşmezse koşu reddedilir.

## 5. Ölçüm: ne sayılıyor

Her probe satırı iki kez puanlanır:

1. **greedy yanıt** (açgözlü çözümleme) → `exact` eşleşmesi; yanıt `'.'` veya
   satır sonunda kesilir, etiketle birebir karşılaştırılır;
2. **teacher-forced** geçiş → beklenen yanıtın baytları tek tek beslenir ve
   - kaç beklenen baytı model **ilk sırada** sıraladı (`correct_bytes`,
     `byte_accuracy`),
   - beklenen yanıtın ortalama negatif log-olabilirliği (`mean_nll`).

İkinci ölçüm, tam eşleşme sıfır olduğunda bile modelin ne kadar yaklaştığını
gösterir; 256 sınıflı bir bayt modelinde şans düzeyi `1/256 = 0.0039`'dur ve
karşılaştırma bu tabana göre yapılmalıdır.

## 6. Yeniden üretim

```bash
make prompt-engine context-tests          # derleme + sözleşme testleri
python3 tools/verify_prompt_response_corpus.py
python3 tools/build_prompt_response_corpus.py generate corpus.txt --bytes 2000000 \
  --facts 256 --probe-out probe.tsv --overwrite
./build/vegai_prompt_engine train --corpus corpus.txt --checkpoint model.vgfb \
  --train-chunk-tokens 128 --width 16 --ffn 32 --local-window 16 --memory-slots 32
python3 tools/build_prompt_response_corpus.py filter-probe corpus.txt \
  --probe probe.tsv --out probe-trained.tsv --trained-bytes 2000000
./build/vegai_prompt_engine probe --checkpoint model.vgfb --probe probe-trained.tsv
./build/vegai_prompt_engine ask --checkpoint model.vgfb --prompt "Soru: ... nedir? Cevap: "
python3 tools/report_prompt_response_evidence.py .
```

## 7. Sınırlar

- Korpus **sentetiktir**; model bu veriyle Türkçe "öğrenmez", bu korpusun
  istatistiğini öğrenir.
- 600 MB tek bir koşuda tüketilemeyecek kadar uzunsa koşu **kaldığı yeri
  söyler** ve zincirleme devam eder; hiçbir yerde "tamamlandı" denmez.
- Bayt düzeyi modelleme, 2048 olgu bağını küçük bir durumda tutmayı
  gerektirir; `fact` doğruluğunun düşük çıkması mimarinin değil, bütçenin
  sonucudur ve ölçüm bunu olduğu gibi raporlar.
- `next` görevi (harf kaydırma) bir *kural*dır; örneklenen 100 satır
  ezberlenemez, ancak çok küçük bir model kuralı öğrenemeyebilir.
