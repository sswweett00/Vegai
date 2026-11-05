# İşlem güvenilirliği ve performans denetimi — 2026-10

Bu çalışma, test edilebilen işlem hatalarını ve somut darboğazları inceler;
projede artık hiçbir hata bulunmadığına veya bütün donanımlarda hızlanma
sağlandığına dair bir garanti değildir.

## Düzeltmeler

| Alan | Bulgu | Düzeltme ve test |
|---|---|---|
| Alt süreç toplam süre sınırı | Lider çıktıktan sonra torun açık boruyu tutarsa ayrı 0,5 saniyelik bekleme gerçek toplam süre sınırını aşabiliyordu. | POSIX ve portable pipe yollarında boşaltma dahil aynı deadline uygulanır; bitiş temizliği yine ek sabit süre alabilir ve `Popen` çağrısı burada kesilemez. İki pump, kısa deadline ve pipe sahibi torunla sınanır. |
| Türkçe corpus bütünlüğü | Var olan corpus dosyaları uyarısız değiştirilebiliyor, negatif byte bütçesi sınırsız sayılabiliyor, `max-docs` sayaçları bir fazla kayıt gösterebiliyor ve 8192-byte üstü kayıtlar sessizce atılabiliyordu. Manifestin eski split açıklaması gerçek SHA girdisini ve ilk 8 byte kullanımını eksik anlatıyordu. | Kilit ve exclusive dosya oluşturma, bütçe ve dizin doğrulaması, kesin sayaç, açık büyük-kayıt hatası ve doğru split açıklaması eklendi. XML parser tamamlanan kök sayfalarını kaldırır; 5.000/50.000 sayfalı testte tamamı okunur ve Python allocation artışı sınırlanır. Tek büyük sayfa veya metin dosyası hâlâ girdiye orantılı bellek kullanabilir. |
| Epoch raporu bütünlüğü | Yalnız manifest sayılarına güvenmek veya sonuç JSON'unu Markdown render/yazımından önce yayımlamak, değerlendirme kapsamını ya da tamamlanmış durumu yanıltıcı gösterebiliyordu. | Held-out kayıt/tahmin ve training sayaçları doğrulanır; kaynak/funnel tutarlılığı ve JSON Unicode/sayıları denetlenir. Raporlar geçici dosyalardan yayımlanır, `evidence.json` son başarı işaretidir; iki dosya transaction'ı veya güç kaybı garantisi değildir. Hatalı evidence yayımı regression testinde başarı raporu bırakmaz. |
| Alt süreç yaşam döngüsü | Lider süreç çıktı borusunu torununa bırakıp çıkınca, başka bir thread'de bekleyen buffered pipe'ın `close()` çağrısı da kilitlenebiliyor; zaman aşımı işlemiyordu. | LMM, HDC pretraining/continuation ve kalite inspect yolları `tools/bounded_process.py` kullanır. POSIX'te nonblocking pipe/selector, sonlu çıkış süresi ve lider bitmiş olsa da process-group temizliği uygulanır. `verify_bounded_process.py` miras kalan ve process group'tan kaçan pipe'ları sınar. |
| Alt süreç girdi/çıktısı | Başarılı çıkış yapan bir child'a isteğin yalnızca bir kısmı yazıldığında işlem başarılı sayılabiliyordu. Küçük limit aşımları buffered read yüzünden zaman aşımı olarak görünebiliyordu; çok küçük parçalarda nesne overhead'i büyüyebiliyordu. | Eksik yazılmış istek başarısızdır; stdout/stderr sınırları okunur okunmaz denetlenir ve stdout `bytearray` içinde tutulur. Aynı anda büyük stdin/stdout, çok sayıda küçük parça, sonlu/pozitif timeout ve ayrı wrapper hata sözleşmeleri sınanır. |
| 9 MiB ön kontrolü | `--overwrite`, binary/fixture doğrulaması veya diğer çıktı yolu reddedilmeden önce eski checkpoint ve raporu silebiliyordu; kırık rapor symlink'i failure report yazımını başka hedefe yöneltebiliyordu. | Her iki çıktı ve girdiler/fixture/resource ön kontrolü tamamlanana kadar eski çıktılar korunur; reddedilmiş symlink veya giriş çakışmasına failure report yazılmaz. Eğitim başladıktan sonra `--overwrite` hâlâ mevcut checkpoint'i değiştirmeye yetki verir; bu iki dosyalı bir transaction değildir. |
| Kalite kanıtının değişmezliği | Kalite değerlendiricisinin yeni ortak alt süreç bağımlılığı ayrıca kimliklendirilmezse değiştirilmiş bir runner eski kanıta bağlı sanılabilir. Rapor yolu da bu kaynağın üzerine yazmamalıdır. | Runner SHA-256 değeri yeni quality guard ve raporlara bağlanır, değerlendirme boyunca kontrol edilir ve çıktıyla çakışması reddedilir. `verify_quality_runner_binding.py` mutasyon, rapor/provenance ve dosya çakışması yollarını sınar. Eski v1 guard/raporlar denetim için okunabilir; yeni fingerprint ile sessizce devam ettirilmez. |
| Kalite karşılaştırmasının çıktısı | `continuous_quality_gate.py --output` bir checkpoint, kanıt raporu, anahtar veya policy dosyasına verilirse girdi dosyası sonuç JSON'u ile değiştirilebiliyordu. | Karşılaştırma yazmadan önce tüm girdiler, binary ve kaynaklarla aynı dosya/hard-link çakışmalarını reddeder. Kopyalanmış fixture dosyalarıyla başarılı karşılaştırma ve tüm çakışma yolları sınanır. |
| Production kalite wrapper'ı | Dış quality evaluator çağrısında wrapper düzeyinde süre sınırı yoktu; kaybolan runner dosyasının hash hatası da beklenen başarısızlık kategorisine çevrilmiyordu. | Inspection süresine 300 saniyelik idari marj eklenerek dış child sınırlanır; süre aşımı veya eksik runner fail-closed raporlanır. İç inspect'in kendi sınırı ayrıca uygulanır. |
| Production ilerleme belleği | Production wrapper, controller'ın her shard için yazdığı ilerleme satırlarının tamamını bellekte tutuyordu. | Production controller'ı `--quiet-progress` ile çağırır; per-shard kanıtı state dosyasında kalır, sonuçta son durum ve hata tanıları korunur. `verify_continuous_progress.py` 20.000 shard için çıktının büyümediğini ve varsayılan interaktif çıktının değişmediğini sınar. Production `child_output_tail` artık ilerleme satırlarını içermez. |
| LMM endpoint toplam süre ve bellek | Socket timeout'u yanıtın tamamı için toplam süre değildi; yavaş akan success/error gövdesi çağrıyı uzatabiliyordu. Tekrarlanan büyük inline image'lar da bellek kullanımını katlayabiliyordu. | Güvenilir leaf süreç ve kendi watchdog'u DNS/HTTP okumasını toplam süreyle sınırlar, redirect gövdesi okunmaz; toplam request 32 MiB, response 8 MiB ve image başına 10 MiB ile sınırlanır. `verify_lmm_endpoint_transport.py` gerçek loopback slow-drip, hata, redirect, parent death, gizlilik ve sınır senaryolarını çalıştırır. |
| LMM runtime provenance | Gateway raporu kendi kaynağını bağlıyordu fakat kullanılan contract/process helper'ları değişse bile held-out runtime fingerprint'i aynı kalabiliyordu. | `lmm_contract.py`, `lmm_process.py` ve `bounded_process.py` SHA/byte/path değerleri generation report, held-out ve dispatch fingerprint'ine alınır; çalışma boyunca yeniden kontrol ve çıktı çakışması koruması yapılır. `verify_lmm_source_binding.py` değiştirilmiş kopya üzerinde fail-closed testi yapar. Eski quality fingerprint'i yeni dispatch ile eşleşmez; held-out değerlendirme yeniden üretilmelidir. |
| Belirsiz JSON ve sayı tipleri | Bazı session/model/knowledge/quality ve training state okuyucuları tekrar eden JSON anahtarlarını son değer kazanacak biçimde, `NaN`/`Infinity` değerlerini veya `true` değerini shard sayısı olarak kabul edebiliyordu. Bazı strict JSONL okuyucuları hatalı uzun satırı reddetmeden önce sonuna kadar tüketiyordu. | Seçilen LMM ve production/continuous okuyucular iç içe tekrar eden anahtarları ve sonlu olmayan/taşan sayıları reddeder; shard sayıları ve generation tam integer ister. Strict JSONL okuyucuları uzun satırı hemen reddeder. `verify_system_contracts.py` gerçek hash-zincirli session ve transaction fixture'larıyla sınar. Diğer tüm proje JSON yollarının aynı kurala sahip olduğu iddia edilmez. |
| Session ledger yayımı | Append, 4 MiB event veya 16 MiB toplam ledger sınırını aşıp yine de dosyayı değiştirebiliyor; sonraki verify kendi yazdığı ledger'ı reddediyordu. | Her satır ve toplam boyut geçici dosyaya yazılırken kontrol edilir, yalnız tüm ledger sınır içindeyse atomik replace yapılır. Limit reddinde eski head/ledger ve kilit korunur; tam sınır, taşma ve temizleme `verify_system_contracts.py` ile test edilir. |
| Continuous state maliyeti | Her commit önce bütün eski journal/evidence nesnelerini `deepcopy` ile yeniden oluşturuyordu. | Yalnız üst state ve journal listesi kopyalanır; geçmiş girdiler controller içinde salt okunurdur, atomic state write başarısızsa eski commit değişmez. Bu, tüm state JSON'unun her commit'te yazılmasını veya uzun journal'ların toplam O(n²) serialization maliyetini ortadan kaldırmaz. |
| LMM entegrasyon testi | Local HTTP fixture port dosyası oluşturulup yazılırken test dosyayı boş okuyabiliyor, özellikle paralel testte endpoint `127.0.0.1:` olabiliyordu. | Üç fixture portu önce geçici dosyaya yazıp sonra atomik rename ile yayımlar; test eksik portu hazır sanmaz. |
| JSON akışı | Tek bir büyük JSON elemanı her chunk sonrasında baştan ayrıştırılıyor, toplam iş yaklaşık karesel büyüyordu. | Küçük kayıtlar hızlı yoldan, büyük kayıtlar sınırları kademeli bulunup bir kez ayrıştırılarak işlenir. Ham UTF-8 byte sınırı, kaçış/Unicode/chunk sınırları ve sentaks doğrulanır; inter-record whitespace biriktirilmez. `verify_training_streaming.py` decoder çağrı sayısını ve çok sayıda chunk sınırını kontrol eder. |
| Kayıt doğrulama | Aşırı iç içe JSON ve geçersiz Unicode ham Python exception'ı üretebiliyor, değiştirilebilir metadata yazıldıktan sonra bozuk JSON'a veya bekleyen Arrow batch'inin değişmesine yol açabiliyordu. | Doğrulama hataları şema/veri hatası olarak yüzeye çıkar; metadata yazım sırasında yeniden doğrulanıp kopyalanır. JSONL collect modu sonraki kayda geçebilir. JSONL/JSON ve Arrow/Parquet regression testleri vardır. |
| Mimari kapsam belgesi | Eski gap tablosu native tensor, optimizer ve causal objective'in hiç bulunmadığını söylüyordu; bu, ayrı frozen-core neural decoder/head trainer kaynaklarıyla çelişiyordu. | HDC ile ayrı neural yol açıkça ayrıldı; byte-BPE/attention/real32/head optimizer doğru kapsamla belirtildi. Full-model backward, model kalitesi ve native compiler doğrulaması iddia edilmedi. |
| NOVA mini-batch doğruluğu | `1/B` faktörü yalnız AdamW dalında başlatılıyor, NOVA'ya tanımsız değer gidiyordu. | Ortak başlatma, bağımsız iki-token ortalama/enerji referansı ve signaling-NaN/FPE debug regresyonu eklendi. Eski `gradient_accumulation > 1` NOVA checkpoint ve ölçümleri yeniden eğitilip değerlendirilmelidir; yükleme geçmiş hatalı eğitimi onarmaz. |
| Neural projection sınırı | Public projeksiyon son 16'lık blokta exact-size gizli dizinin sonunu okuyabiliyordu. | Varsayılan public yol yalnız geçerli satırları okur; eğiticide sıfır dolgulu hızlı yol kullanılır. 1, 2, 15, 16, 17, 31, 32 satır için referans ve bounds testleri eklendi. |
| Neural checkpoint devamı | `resume` veya `train-chunk` için herhangi bir seçenek verilince tüm kaydedilmiş hyperparametreler varsayılanlara dönüyordu; optimizer değişiminde uyumsuz/stale durum kaydedilebiliyordu. | Yalnız açık alanlar override edilir, yalnız verilen şekil alanları karşılaştırılır. Güncelleme almış checkpoint'te optimizer değişimi, eksik/karma optimizer state'i eğitim ve kayıttan önce reddedilir. Eski SGD dosyalarındaki kullanılmayan Adam momentleri okunup doğrulandıktan sonra atılır; kesik optimizer payload'ı FPE yerine hata verir. |
| Neural checkpoint sayısal doğrulaması | Optimizer varyansındaki quiet/signaling NaN, v3 momentindeki taşan real64 değer ve real32'ye sığmayan öğrenme hızı FPE-debug derlemesinde `SIGFPE` üretebiliyordu. Eksik momentli eğitilmiş checkpoint yüklenip ancak eğitimde reddediliyordu; bozuk sayaçla kaydetme eski dosyayı bozabiliyordu. | IEEE binary32/binary64 bit düzeyinde sonluluk kontrolü, dönüştürmeden önce v3 aralık/işaret kontrolü ve O(1) model biçimi/sayaç/optimizer scalar aralığı doğrulaması yapıldı. Eksik durum yüklemede reddedilir, bozuk model/state mevcut checkpoint açılmadan reddedilir. Debug ve optimize testler geçerli v3/v5 dosyalarını da sınar; başka floating-point gösterimleri ayrıca test edilmelidir. |
| Neural/context CLI sayı girişi | `--lr 1e309` GNU FPE-trap derlemesinde `IOSTAT` değerlendirilemeden `SIGFPE` üretiyordu; `1e-9999` gibi sıfıra yuvarlanan sıfır-dışı sayı veya list-directed `0.1,0.2` gibi ek token sessizce kabul edilebiliyordu. | İki CLI ondalık/bilimsel sayı sözdizimini ve argüman uzunluğunu denetler; dönüşüm sırasında IEEE durumunu saklar/geri yükler, taşan ve sıfıra yuvarlanan sıfır-dışı değerleri reddeder. Neural ilk-checkpoint config reddi fallback tokenizer oluşumundan öncedir. Bu kontrol diğer tüm proje CLI'larını kapsamaz. |
| Neural epoch koşu koruması | Aynı `--run-dir` tekrar kullanılırsa önceki tokenizer/checkpoint yeni koşunun başlangıcında bildirim olmadan ezilebiliyordu. İlgisiz config hataları pahalı tokenizer sonrasına kalabiliyordu. | Bilinen çıktı dosyalarına çarpışma ve eşzamanlı runner lock'u başarısızlığı baştan reddedilir; model/optimizer/schedule/CPU seçenekleri önceden sınanır. Mevcut tokenizer salt okunur kalır. Lock yalnız aynı runner'ları koordine eder; native checkpoint yazımı hâlâ yerinde, crash-atomic değildir ve ayrı yedek gerekir. |
| Neural mini-batch geçici belleği | Sütun-major logits matrisindeki satır strided olduğu hâlde yorum contiguous diyordu; softmax çağrısı her tahminde geçici buffer ayırıyordu. | Zaten var olan `workspace%logits` tekrar kullanılır. Ek model buffer'ı veya geçiş eklenmez; debug derlemede bu çağrıdaki array-temporary uyarısı kayboldu ve eşit bütçede checkpoint baytları aynı kaldı. |
| Eğitim kalite bütçesi | Fortran API'nin NOVA için 0,025 varsayılan LR'si CLI'nin 0,003 değerinden farklıydı ve örnek koşularda aşırı yükselen NLL üretiyordu. | API varsayılanı CLI ile aynı 0,003'e çekildi; kaydedilmiş checkpoint LR'si değişmez, eski SGD varsayılanını isteyen API kullanıcısı 0,025'i açıkça vermelidir. Eşit token/adım sayılı sentetik A/B aşağıdadır; optimum her veri ve modelde farklı olabilir. |
| Sabit bütçeli context eğitimi | Son eksik chunk sessizce atlanıyor; aşırı düşük target olasılığında NLL yaklaşık 69'a kırpılıyordu. | Bölünmeyen bütçe varsayılan olarak eğitim başlamadan reddedilir: fazladan update sessizce eklenmez. `--allow-partial-chunk true` veya API'de `allow_partial_chunk=.true.` ile kalan parça gerçek uzunluğuyla açıkça işlenebilir; bu tercih bir ek update ve ek hesaplama gerektirir. NLL mevcut softmax normalizer'ı ile logit alanında hesaplanır. |
| Seyrek uzman katmanı | Optimize SplitMix signed-overflow tanımsız davranışı ve rehash sonrası çift indeks kaydı vardı. Farklı forward'lardan biriken kirli sayfalar yalnız `k` boyutlu listeyi aşıyor, aynı `k` ile farklı şekilli layer'larda cache yanlış boyutla kalabiliyordu. | Güvenli modüler hesap, yayın öncesi rehash, dinamik kirli liste ve sayfa bayrağıyla sabit zamanlı üyelik, cache şekil kontrolü eklendi. Optimize/debug testleri, bilinen vektörler, indeks tekilliği ve çoklu backward sınanır. |
| Native UTF-8 korpus I/O | Her bayt ayrı unformatted Fortran read çağrısıyla okunuyordu. | Neural/BPE yolları bağımsız 64 KiB read-ahead kullanır; eski çağrı imzası buffersız davranışını korur. Bağlı unit yalnız okuyucuya aittir; yeniden konumlandırma/yeniden açmada okuyucu yeniden başlatılır. Bilinmeyen boyutlu akışlarda byte fallback vardır. `test_utf8_record_stream.f90` bayt eşitliğini ve sınırları; `verify_make_native.py` POSIX FIFO'yu sınar. |
| Make derlemesi | Her executable kaynakları tekrar derliyordu; arşiv iyileştirmesinden sonra da tek kaynak değişikliği bütün kütüphaneyi derliyordu. Tek `.mod` silinmesi fark edilmiyordu ve değişmiş derleme/bağlama bayrakları eski ikilileri kullanabiliyordu. | Bağımlılık sıralı ayrı nesneler paralel/artımlı derlenir; eksik modüller onarılır, hatalı kısmi hedef silinir, değişmiş compiler/flag'ler ilgili işi tetikler. `verify_make_build.py` graph/onarım/flag, `verify_make_native.py` gerçek compiler ile build ve FIFO testidir. Aynı yoldaki compiler binary'si yerinde değiştirilirse temiz derleme gerekir. |
| GNU/CMake uyumluluğu | GNU `-std=f2023` desteklemeyen compiler derlemede geç hata veriyordu. Ayrıca bazı uygulama kaynakları 132 sütunu aştığı hâlde CMake'de GNU satır-uzatma bayrağı yalnız kütüphaneye veriliyor, tam build başarısız oluyordu. | Daha geniş derleyici uyumu için varsayılan 2018'dir; `VEGAI_FORTRAN_STANDARD=2023` / `FORTRAN_STANDARD=f2023` opt-in'dir. GNU bayrak desteği configure aşamasında sınanır; kaynak satırları 132 byte sınırına indirilir ve graph testi bu sınırı denetler. Linux/GNU 12/Fortran 2018 gerçek CMake ve Make build'leri çalıştırılmıştır; diğer compiler/OS uyumluluğu ayrıca test edilmelidir. |

## Ölçüm ve kapsam

### Sayısal sınırlar ve eşit-bütçe kontrolü

Yeni regresyonlar, neural tek-belirteç ve batch yolunda 2000 natlık NLL'yi ve
underflow sonrası top-5 sırasını, sıfır gizli vektörle eğitimi, 1/4 OpenMP
thread deterministikliğini; context yolunda karşıt real64 uçlarını, temsil
edilemeyen kaybı, NaN/signaling NaN'ı, normun büyük/küçük sınırlarını ve üç
optimizer yolunda pozitif/negatif anlaşmayı sınar. Eski neural sonda NLL
708,396 ve top-5 doğru sayısı 1 iken yeni sonuç 2000 ve 0; eski context sondası
`[huge, -huge, 0]` ve hedef 1 için FPE ile dururken yeni loss 0'dır. Aşırı
boyutlu eski planlar negatif int64 sonuç verebiliyordu; yeni planlar reddeder.

GNU 12 `-O3 -std=f2018`, OpenMP kapalı, 1 thread, **aynı** 200.000 baytlık
sentetik korpus/model/tohum başına eşit token ve update bütçesiyle neural eski
yeni ikilileri beş seed'de karşılaştırıldı. Beşinin tamamında model, optimizer
ve diğer checkpoint baytları aynıydı; yalnız 137 seed'inde yeniden hesaplanan
kümülatif NLL alanı son bitte değişti. Held-out NLL farkı beşinde de ölçülen
hassasiyette 0; süre oranı (yeni/eski) medyan **1,041**, aralık **0,927–1,122**
([ham sonuç](experiments/neural-math-guards-2026-10.json)). Ayrı tekrarda medyan
**1,012**, aralık **0,977–1,039** ve held-out NLL farkı yine 0'dı
([ham tekrar](experiments/neural-math-guards-repeat-2026-10.json)).
CLI sayı ayrıştırması da eklenmiş **son ikiliyle** iki ayrı tekrarın medyanları
**0,957** ve **0,847**; bu on ölçümün aralığı **0,587–1,344** oldu
([ilk](experiments/neural-math-final-cli-2026-10.json),
[tekrar](experiments/neural-math-final-cli-repeat-2026-10.json)). Her tekrarda
NLL farkı 0, model/optimizer ve diğer state 5/5, tüm checkpoint 4/5 eşitti.
`--require-model-state-identity` yalnız v5 formatında kümülatif NLL'nin sekiz
baytını hariç tutarak kalan **tüm** checkpoint'i ve dosya boyutunu karşılaştırır;
`--require-checkpoint-identity` ise hâlâ dosyanın tamamının eşitliğini ister.

Context için 8192 token, width 32, vocab 256, beş seed; chunk 16 ve 256 ile
iki eski/yeni, sıralaması dönüşümlü ölçüm yapıldı. İlk ölçümde süre oranı
medyanları sırasıyla **0,984** ve **1,030**, tekrarda **0,986** ve **0,951**;
her iki turda da onar sonucun basılan 6 ondalıklı eğitim NLL'si eşitti
([ilk](experiments/context-math-guards-2026-10.json),
[tekrar](experiments/context-math-guards-repeat-2026-10.json)). Son CLI
ikilisiyle tekrarın medyanları chunk 16/256 için **1,016/1,016**; ayrı tekrarda
**0,984/0,996**, tüm ölçümler aralığı **0,801–1,263**, basılan NLL 20/20 eşittir
([ilk](experiments/context-math-final-cli-2026-10.json),
[tekrar](experiments/context-math-final-cli-repeat-2026-10.json)). Bu basılmış
NLL, parametrelerin aynı olduğu anlamına gelmez: ayrı 1024-token/64-update seed-21
kontrolünde checkpoint'ler farklıydı, yüklenen model alanları için en büyük
mutlak fark `2,67e-12`, optimizer için `1,06e-17`, toplam NLL farkı 0'dı.
Norm/softmax indirgeme sırası farklıdır; eski ve yeni sayısal davranış için
bit-özdeşlik vaat edilmez. `tools/benchmark_context_numerics.py` ile ölçüm
tekrarlanabilir.

Normal eğitimde yeni bir tam-array geçişi veya optimizer update'i eklenmedi:
context logit denetimi mevcut maksimum taramasına, exp toplamı da exp geçişine
katıldı; norm ve kırpma için aynı ölçüm yeniden kullanıldı. Doğrudan optimizer
API çağrısı doğrulanmış norm verilmezse ayrı norm tarar. Bu kısa ve gürültülü
ölçümler her koşuda süre/maliyet artmayacağını veya enerji eşitliğini
**kanıtlamaz**; önceki neural medyanları küçük artış, son koşular ise büyük
değişkenlik gösterdi. Bunlar sentetik görevlerdir; genel dil kalitesi veya tüm
kayan-nokta işlemleri için ispat değildir.

### Eşit eğitim bütçesinde sentetik kalite karşılaştırması

Linux/GNU 12 `-O3 -std=f2018`, OpenMP kapalı, Intel Xeon sanal CPU, 1 thread,
vocab 320, width 32, batch 16, sabit LR, tohum başına yaklaşık 200.000 bayt
sentetik korpus. Tokenizer yalnız
eğitim bölümünden çıkarıldı; aynı model/seed/veriyle eski API varsayılanı 0,025
ve yeni 0,003 karşılaştırıldı. Aynı seed içindeki token ve update sayıları her iki
kolda da aynıdır; tablodaki NLL ayrı sentetik held-out kayıtlarına aittir.

| Seed | Tahmin tokenı | Update | 0,025 NLL | 0,003 NLL | eğitim süresi eski → yeni |
|---|---:|---:|---:|---:|---:|
| 21 | 78.633 | 5.015 | 5,9959 | 3,6634 | 1,470 → 1,326 s |
| 42 | 81.578 | 5.196 | 4,3946 | 3,6483 | 1,426 → 1,412 s |
| 73 | 81.815 | 5.213 | 4,7524 | 3,6256 | 1,479 → 1,412 s |

Bu üç koşuda yeni değer daha iyi; yeni/eski süre oranı medyanı 0,954'tür.
Aynı seed'lerle önceki ayrı tekrarda medyan 0,997'dir
([ham tekrar](experiments/neural-equal-budget-3-seed-repeat-2026-10.json)).
Son checkpoint doğrulama değişikliklerini içeren ikiliyle tekrarlandığında NLL
sonuçları aynı, süre oranı medyanı 0,991 oldu
([güncel ham tekrar](experiments/neural-equal-budget-current-2026-10.json)).
Tek host ve saniyeler mertebesinde ölçümler değişkendir; eşit maliyeti diğer
donanımlarda **garanti etmez**.
Her iki kol aynı işlemleri ve optimizer durumunu kullanır; LR değişimi yeni
training pass veya optimizer belleği eklemez. Düşük LR her görev için en iyi
olmayabilir. Deney tekrarı: `python3 tools/benchmark_neural_quality_budget.py
--neural-bin ./build/vegai_neural_pretrain --corpus-bytes 200000`; ham
[JSON sonuçları](experiments/neural-equal-budget-2026-10.json) binary, veri ve
tokenizer SHA-256 değerlerini de içerir. İsteğe bağlı A/B'nin kendi ek deney
maliyeti vardır. Bias enerjisi ve peak-EMA üzerinde
başka adaylar da denendi; bazı batch/LR kombinasyonlarında NLL kötüleştiğinden
varsayılan algoritma yapılmadı.

### Bölünebilir context bütçesi

Sabit-bütçe context yolunda yeni varsayılan, bölünmeyen bütçeyi eğitimden önce
reddeder. Ayrı bir 256 token / 32 chunk / 8 update kontrolünde, kısa parça
korumasından önceki ve sonraki F2018 debug ikilileri aynı checkpoint baytlarını
üretti (SHA-256 `925721c5d6c401c2e1fa8c55ae9ab528d949aa97b370e80a768a92a9a63c66c9`).
Bu tek kısa kontrol bir performans/enerji ölçümü değildir; açık kısa-parça
tercihi ayrıca bir update yapar.

### Mini-batch geçici buffer karşılaştırması

Yine GNU 12 `-O3`, OpenMP kapalı ve aynı sentetik şekilde, eski geçici buffer
yolu ile yeniden kullanım yolu **aynı 0,003 LR** ve aynı token/adım bütçesinde
karşılaştırıldı. Dokuz seed'in tamamında checkpoint dosyaları bayt bayt eşleşti;
yeni/eski wall-time oranı medyanı 0,983, aralığı 0,955–1,036 idi ve score
fazı medyan oranı 0,998 idi ([ham JSON](experiments/neural-buffer-reuse-2026-10.json)).
Bundan önceki üç-seed tekrarında ise süre oranı medyanı 1,025 ölçüldü
([ham JSON](experiments/neural-buffer-reuse-3-seed-2026-10.json)); tüm
checkpoint'ler yine aynıydı. Her tahminde gizli bir allocation yerine zaten
ayrılmış `workspace%logits` kullanılır. Bu küçük/gürültülü örneklerde çıktı
korunmuştur ama güvenilir hız artışı, her koşuda maliyet eşitliği veya enerji
azalması kanıtlanmış değildir. Aynı araç farklı ikililer için tekrarlanabilir:

```bash
python3 tools/benchmark_neural_quality_budget.py \
  --reference-bin /path/to/old/vegai_neural_pretrain \
  --neural-bin ./build/vegai_neural_pretrain \
  --reference-lr 0.003 --candidate-lr 0.003 \
  --require-checkpoint-identity --corpus-bytes 200000
```

### Checkpoint doğrulama değişikliği: eşit-bütçe kontrolü

Önceki ikiliye karşı yeni loader/save ve tokenizer-yazımı düzeltmeleri, GNU 12
`-O3`, tek thread, aynı LR 0,003 ve aynı token/update bütçesiyle dokuz sentetik
seed'de sınandı. Dokuz checkpoint de bayt özdeş, held-out NLL farkları sıfırdı;
yeni/eski eğitim duvar süresi oranı medyanı 0,995, aralığı 0,920–1,097 idi
([ham JSON](experiments/neural-checkpoint-guards-2026-10.json)). Aynı ikililerle
üç seed'lik ayrı tekrarda medyan 0,964 oldu; checkpoint'ler yine aynıydı
([ham tekrar](experiments/neural-checkpoint-guards-repeat-2026-10.json)). Bu kısa ve
gürültülü tek-host deneyi evrensel hız veya enerji garantisi değildir; ek model
forward/backward geçişi veya optimizer belleği eklenmedi. Eski v3 checkpoint
okunurken dönüşüm öncesi moment taraması eklenir.

### Veri yolu ve I/O

Aşağıdaki iki sayı aynı sandbox'ta, kaynak dosyası üretildikten sonra ve ısınma
traversal'ı hariç üç `iter_records` traversal'ının medyanı olarak ölçüldü.
Dosyalar OS page cache içindeydi; ölçüm training veya disk cold-cache benchmark'ı
değildir. 8 MiB kayıt yalnız ASCII `text` içeren tek JSON array elemanıdır;
100.000 kayıt fixture'ı kısa `text` ve `id` alanları içerir.

| Fixture | Önce | Sonra | Not |
|---|---:|---:|---|
| 8 MiB tek kayıt | 0,924 s | 0,099 s | Bu örnekte yaklaşık 9,3× daha kısa okuma süresi |
| 100.000 kısa kayıt | 1,058 s | 1,006 s | Kısa kayıt hızlı yolu korunur |

Sayılar host/yük/Python/fixtüre bağlıdır ve genel performans taahhüdü değildir.
`tools/benchmark_training_data_io.py` ile hedef sistemde ayrı ölçüm yapılmalıdır.
Varsayılan `make all` kaynak derleme sayısı yapısal olarak HDS için 210'dan
21'e, context için 8'den 4'e iner. Ayrı bir Linux/GNU 12 ölçümünde, aynı `-O2
-std=f2018 -fopenmp` bayrakları ve `make -j4 all` ile yalnız
`turkish_text.f90` dosyasını touch sonrası eski tek-invocation arşiv grafiği
üç koşu medyanında 10,904 s, yeni nesne grafiği 2,723 s sürdü. Yeni grafikte
iki HDS modülü yeniden derlendi ve ilgili uygulamalar yeniden bağlandı. Bu
tek sistem/fixture ölçümüdür, temiz derleme veya diğer compiler'lar için hız
vaadi değildir.

Native okuyucu da yalnız I/O için üç kez ölçüldü; dosyalar OS page cache'te,
GNU 12.2, `-O3 -std=f2018`, 8 KiB output buffer ile kullanıldı:

| LF kayıtları | Eski medyan | 64 KiB okuyucu medyanı | Aynı satır/bayt sayısı |
|---|---:|---:|---:|
| 40.000 × 833 bayt | 2,191 s | 0,0311 s | 40.000 / 33.320.000 |
| 400.000 × 80 bayt | 2,160 s | 0,0525 s | 400.000 / 32.000.000 |

Yaklaşık 70×/41× fark yalnız bu sentetik, cache'li okuma döngüsündedir;
training, cold-disk veya kullanıcı iş yükü hızlanması değildir. Tam bayt
karşılaştırması ayrı native kontrat testindedir; benchmark checksum'u
kriptografik doğrulama değildir. Aynı kıyas hedef hostta şu şekilde yinelenebilir:

```bash
mkdir -p /tmp/vegai-stream-bench
python3 - <<'PY'
from pathlib import Path
for width, rows in ((833, 40000), (80, 400000)):
    record = (b'abcdefghij' * (width // 10 + 1))[:width] + b'\n'
    with Path(f'/tmp/vegai-stream-bench/{width}.txt').open('wb') as output:
        for _ in range(rows):
            output.write(record)
PY
gfortran -O3 -std=f2018 -J/tmp/vegai-stream-bench \
  src/utf8_record_stream.f90 tools/benchmark_utf8_stream.f90 \
  -o /tmp/vegai-stream-bench/benchmark
for file in /tmp/vegai-stream-bench/{833,80}.txt; do
  for mode in legacy buffered; do /tmp/vegai-stream-bench/benchmark "$file" "$mode"; done
done
```

20.000 sentetik geçmiş girdisiyle, yalnızca sonraki state nesnesini hazırlayan
fonksiyon üç kez ölçüldü; `tracemalloc` açıktı ve medyanlar `deepcopy` için
0,763 s / 29,003 MiB, journal-spine kopyası için 0,000108 s / 0,172 MiB ek
peak'tir. İki sonuç aynı JSON içeriğini üretir. Bu, gerçek shard eğitiminin veya
state serialization/fsync süresinin benchmark'ı değildir. Aynı loopback slow-drip
fixture'ında 0,25 s sınırı eski kodda 1,21 s sonra başarıyla dönüyordu; yeni
kod 0,25 s civarında süre aşımı verir. İki ölçüm de bu sandbox/Python/fixture ile
sınırlıdır.

## Doğrulama ve sınırlar

```bash
python3 tools/verify_bounded_process.py
python3 tools/verify_lmm_endpoint_transport.py
python3 tools/verify_lmm_source_binding.py
python3 tools/verify_system_contracts.py
python3 tools/verify_quality_runner_binding.py
python3 tools/verify_continuous_progress.py
python3 tools/verify_training_streaming.py
python3 tools/verify_neural_quality_budget.py
python3 tools/verify_context_numerics.py  # VEGAI_CONTEXT_BIN ile native CLI sayı sınırları da sınanır
python3 tools/verify_neural_turkish_epoch1.py  # VEGAI_NEURAL_BIN ile native mikro koşuları da sınar
python3 tools/verify_make_build.py
python3 tools/verify_make_native.py  # GNU Fortran 2018 destekli compiler gerekir
python3 tools/verify_training_data_io.py  # optional pyarrow gerekir
make check                 # Python'suz native suite
make legacy-python-check   # eski bağımsız Python suite
cmake --preset portable
cmake --build --preset portable --parallel
ctest --preset portable --output-on-failure
# Yeni GNU compiler için opt-in: cmake -S . -B build-f2023 -DVEGAI_FORTRAN_STANDARD=2023
```

Yeni stdlib testleri isteğe bağlı CTest Python suite'ine ve Linux/macOS/Windows
CI matrisine eklenmiştir. `verify_make_build.py` sahte compiler ile graph'ı;
`verify_make_native.py` gerçek GNU compiler, modül onarımı, bayrak değişimi ve
POSIX FIFO'yu sınar. CI ayrıca 2018 profiliyle tam CMake derlemesi çalıştırır.

Yerel testte önceden kurulu gfortran yoktu; GNU 12.2 kaynaklarından geçici
compiler derlendi. Flex bulunmadığından üçüncü taraf üretilmiş scanner ve
OpenMP için aynı GCC kaynaklarından türetilen modül ile sistem `libgomp`
kullanıldı; bu test toolchain'i bağımsız doğrulanmış üretim dağıtımı değildir.
Testler ayrı `nobody` kullanıcısıyla yürütüldü; `unshare -n` reddedildiğinden
bu bir ağ sandbox'ı da değildir. Linux/Fortran 2018 debug derlemesinde
`-fcheck=all -finit-real=snan -ffpe-trap=invalid,zero,overflow` ve gerçekten
etkin OpenMP (iki thread) ile 1.719 HDS/neural ve 261 context kontrolü geçti;
sparse 124/0, UTF-8 ve POSIX FIFO testleri de geçti. CMake debug/OpenMP native
suite 29/29, Release/OpenMP kapalı native + isteğe bağlı Python suite 71/71
geçti; sonuncuda opsiyonel gerçek OpenVINO provider testi runtime olmadığı
için **SKIP** idi. Önceki ayrı optimize/FPE deneyinin kapsamı ve ikili hash'leri
ölçüm dosyalarında kayıtlıdır. Native Turkish epoch runner AdamW, chunked AdamW
ve NOVA mikro koşuları geçti; o tarihteki Python tabanlı `make check` 27 hedef ve isteğe bağlı pyarrow I/O
doğrulaması da geçti.

Batch=1 mini fixture üzerinde eski/yeni BPE ve iki chunk'lı neural checkpoint
byte-byte aynı çıktı; süre alanları dışındaki metrikler eşleşti. Bu test hatalı
eski NOVA mini-batch checkpoint'lerini aklamaz. Sonuçlar resmi toolchain veya
Fortran 2023 doğrulaması, Windows/macOS, Intel compiler, gerçek OpenVINO cihazı
veya büyük eğitim koşusu değildir.

POSIX process group temizliği ayrı session'a kaçan kötü niyetli bir torunu
öldürmeyi garanti etmez; yalnız controller'ın pipe üzerinde kilitlenmesini
önler. Windows'ta pipe seçicileri taşınabilir olmadığından unbuffered daemon
thread kullanılır; child'dan bağımsız yaşayan bir torun pipe'ı açık tutarsa
ilgili thread/handle o pipe kapanana kadar kalabilir. Production wrapper'ın dış
quality timeout'u doğrudan değerlendiriciyi sonlandırır; beklenmeyen kesintide
ayrı session'daki inspect veya geçici snapshot kalabilir. Çok yavaş snapshot
I/O'sunda 300 saniyelik idari marj yetersiz kalabilir; süre ayarını ve scratch
alanını izleyin. Bu kontroller bir OS sandbox, cgroup veya Windows Job Object
yerine geçmez. Tam yazım kontrolü byte'ların pipe'a aktarılmasını doğrular;
child'ın bunları uygulama düzeyinde işlediğini kanıtlamaz.

## 2026-11-01 native taşınabilirlik ve güvenlik doğrulaması

Bu ek, önceki tarihli ölçümlerin yerine geçmez. Linux x86-64 sandbox'ında,
Fortran 2018 ile güncel native kaynak için şu sonuçlar alındı:

- OpenMP kapalı Release CTest **29/29**; `VEGAI_REQUIRE_OPENMP=ON`, iki thread
  probe'u ve `-fcheck=all -finit-real=snan -ffpe-trap=invalid,zero,overflow`
  içeren Debug CTest **30/30** geçti. OpenMP kapalıyken zorunlu seçeneği ve
  bulunamayan OpenMP durumları configure aşamasında reddedildi.
- Python bulunmayan kısıtlı `PATH` ile `make -j2 check` geçti: HDS/neural
  1737/0, fixed-budget context 265/0, sparse 124/0 ve native preflight/parser
  63/0. Bu, yalnız bu native hedefin Python gerektirmediğinin kanıtıdır;
  depodaki diğer Python araçlarının taşındığı anlamına gelmez.
- İki eşzamanlı CTest alt kümesi 15/15 ve 15/15 geçti. Eski sabit
  `context-smoke.vgfb` ve partial-checkpoint adlarına konulan sentinel dosyaları
  değişmedi. Scratch dizinleri rastgele seçilir ve güvenilir bir test build
  dizini gerektirir; bu, düşmanca dosya sistemi yarışlarına karşı sandbox
  garantisi değildir.
- 23 farklı corpus ve beş özel/default limit birleşimi için native manifest,
  mevcut Python scanner ve bağımsız SHA-256 sonuçları karşılaştırıldı: Release
  ve Debug'da ayrı ayrı **115 karşılaştırma** geçti. CRLF telemetrisi, belgelenen
  son bare-CR ve overlong-prefix farkları nedeniyle eski scanner'dan 32 durumda
  farklıydı; native sonuç gerçek CR+LF çiftleriyle ayrı doğrulandı. Bu Python
  karşılaştırması bağımsız test oracle'ıdır, native runtime bağımlılığı değildir.
- Resource reserve hesabı, dört vocabulary boyutu, iki context modu ve sınır
  komşuları dahil **208** binary64 değeri için Python `Fraction` tam rasyonel
  aritmetiğiyle Release ve Debug'da ayrı ayrı karşılaştırıldı. Çok küçük bir
  rezervin çarpımı kayan noktada tam `1`e yuvarlansa bile gerçek sonuç `1`den
  büyük olduğunda native plan 2 byte ayırır. Bu doğrulama non-binary64 fallback
  üzerinde çalıştırılmadı.
- Gerçek derleyiciyle Make bağımlılık/modül onarımı/FIFO testi ve sahte
  derleyici graph testi geçti. Debug binary'lerle neural epoch/chunk/NOVA ve
  context sayısal CLI regresyonları da geçti. Bu mikro koşular üretim ölçeğinde
  eğitim kalitesi veya maliyet garantisi değildir.

Geçici derleyici GCC 12.2 kaynaklarından oluşturuldu; üretilmiş lexer ve build
header'ları bağımsız doğrulanmış resmi bir üretim toolchain'i değildir, OpenMP
modülü aynı GCC kaynaklarından üretilip sistem `libgomp` ile kullanılmıştır.
Bu turdaki testler normal sandbox kullanıcısıyla yürütülmüştür; ağ izolasyonu
kanıtlanmamıştır. Windows/macOS, GCC 14 CI matrisi, Intel/Fortran 2023,
non-binary64 gösterim ve gerçek GPU/NPU/OpenVINO runtime **bu koşuda test
edilmedi**. İş akışı tanımları başarı kanıtı değildir. Checkpoint'ler hâlâ yerinde
yazılır; maliyet sınırını korumak için ek kopya yapılmaz ve bağımsız yedek veya
snapshot işletim sorumluluğudur.

## 2026-11-01 ek eğitim ve CLI doğrulaması

- Dolu halka backward ve kırpılmış decay türevi v6'da düzeltildi; v5 checkpoint
  eski kuralını korur. Gerçek önceki binary ile normal ve clamp uçlarında
  devam checkpoint'leri byte-byte aynıdır (sırasıyla SHA-256
  `40c6744ffea25fc0eb1d659fa8cfe1ecbeae6ac222b22856dc60bdaa1df0a8ff`,
  `900e1952d4669bec33b594183acee153099818125b24ed440fc13fe46dd39da5`). Eski
  binary v6'yı reddeder. Checkpoint recurrent state içermez; CLI resume yeni
  state ile başlar.
- Prompt resume `--lr` artık yalnız LR'yi override eder; diğer optimizer
  seçenekleri checkpoint'te kalır. V6 NUL/0xff eğitimini destekler, v5 NUL'u
  açıkça reddeder; hatalı girdi ve başarısız eğitim mevcut çıktı sentinel'ini
  değiştirmez. Prompt parser int32 sarmasını, NaN/inf, kesik argümanı ve
  FPE'ye yol açan taşan/underflow literal'leri reddeder. Aşırı küçük geçerli
  temperature ve büyük seed FPE ve `-ftrapv` ile sınandı. Aynı yazılı corpus
  ve checkpoint yolu reddedilir, ancak symlink/hardlink alias garantisi yoktur.
- Türkçe BOUN üzerinde önceden dev ile seçilen v6/0,006 ayrı testin her iki
  bütçesinde 5/5 seed'de NLL'yi düşürdü; eşit LR ve CLI zamanlaması dahil tüm
  kapsam/sınırlamalar [deney raporunda](experiments/context-prompt-byte-quality-2026-11.md).
  Başka bir küçük fixture'da eşit-LR farkı olumsuzdur; genel kalite veya mutlak
  maliyet artmama garantisi verilmez.
- Bu ekin son Release ve iki-thread OpenMP/FPE Debug CTest'i **35/35 + 35/35**;
  gerçek FPM 0.12 seri ve OpenMP koşuları 1737/0 HDS, 330/0 context, 124/0
  sparse, 63/0 preflight ve UTF-8 PASS. FPM prompt CLI ayrı sözleşmesi geçti;
  runner'sız test güvenle reddedildi ve sentinel korundu. Aynı build dizinine
  karşı iki eşzamanlı CTest 7/7 + 7/7 ve üç sentinel koruma testi geçti.
- Python bulunmayan `PATH` ile native Make testleri, quality benchmark derlemesi
  ve gerçek engine/train/inspect `portable-runtime-check` geçti. Runtime smoke
  artık CMake test altyapısıyla Python'suz çağrılabilir; CMake test aracıdır,
  Fortran engine'in çalışma zamanı bağımlılığı değildir. Legacy Python smoke
  ve diğer Python araçları ayrıca durmaktadır; tüm işlevlerin Fortran'a
  taşındığı veya gerçek Windows/macOS/Intel/OpenVINO çalıştırıldığı söylenemez.

## 2026-11-02: prompt sınırları ve optimize eğitim tekrarı

- Linux x86-64 üzerinde yeniden hazırlanan, bağımsız olarak üretim için
  doğrulanmamış GNU Fortran 12.2 ile güncel native kaynak: seri Release CTest
  **35/35**, iki thread OpenMP, `-fcheck=all`, `-finit-real=snan`,
  `-ffpe-trap=invalid,zero,overflow` ve `-ftrapv` Debug CTest **36/36**.
  İlk ağır eşzamanlı build/test koşusunda beş seed'li kalite smoke'u 60 saniye
  sınırına takıldı; ayrı tekrar 30,51 saniyede geçti, test süresi bütçesi 180
  saniyeye çıkarıldı ve tam Debug tekrarında geçti. Eğitim iş yükü veya update
  bütçesi artırılmadı; daha yüksek timeout performans garantisi değildir.
- Native prompt regresyonları geçersiz UTF-8 JSON için `null` ve hex çıktısını,
  NUL/0xff byte'larını, 4.096/4.097 probe satırını, beklenen cevabın sondaki
  boşluklarını, uzun/truncated argümanları, 8.192 baytlık prompt sınırını,
  küçük kesirlerin JSON'da sıfıra yuvarlanmamasını, büyük sonlu NLL yazımını,
  tek satırda ve satırlar arasında NLL toplamı taşmasını ve hatalı dosya
  yollarında kullanıcı checkpoint sentinel'inin korunmasını çalıştırdı.
  Probe taşma/hatalı kayıt durumunda nonzero döner; daha önce yazılmış kısmi
  stdout'un rollback garantisi yoktur. `whole_file` gerçek sondan başlanmış
  önceki byte aralıklarının mutlaka işlendiğini kanıtlamaz.
- Python yürütülebiliri bulunmayan bir `PATH` ile Make prompt ve runtime
  kontrolleri, 1.737/0 HDS, 330/0 context, 124/0 sparse ve 66/0 native preflight
  geçti; quality A/B aracı da aynı ortamda derlendi ve geçersiz, sondaki boşluklu
  veya uzun yolları reddetti. FPM 0.12 seri/OpenMP testleri ve FPM prompt
  uygulamasının native JSON/devam sözleşmeleri geçti. Bağımlılık grafiğinin hem
  fake-compiler hem gerçek GNU Fortran doğrulamaları geçti; fake derleyici tek
  başına derleme/çalışma kanıtı değildir.
- Optimize gradient yolu BOUN'un önceki test-80 verisi ve aynı 5 seed'le tekrar
  ölçüldü; bu **bağımsız yeni veri değildir**. V5 LR 0,003'e karşı V6 LR 0,006
  medyan NLL farkı kısa/uzun eğitimde **−0,206391/−0,060040 nat/byte**, medyan
  salt eğitim süre oranı **0,9938/0,9745** oldu; kısa bütçede tek seed oranı
  **1,2189** idi. Aynı LR kontrolü ve tüm seed sonuçları
  `experiments/context-prompt-byte-quality-2026-11.md` ile ham dosyalardadır.
  Bu sonuçlar maliyetin her koşuda artmayacağını veya genel görev kalitesini
  kanıtlamaz; önceki dış CLI zamanlaması güncel binary'nin doğrudan ölçümü değil.
- Python'a bağlı özelliklerin tümü taşınmadı. Gerçek Windows/macOS, Intel
  Fortran, GPU/NPU, OpenVINO ve büyük gerçek eğitim bu turda çalıştırılmadı.
  Checkpoint overwrite atomik değildir; ek checkpoint kopyası maliyet kararı
  gereği eklenmemiştir ve bağımsız yedek/snapshot hâlâ gereklidir.

### İsteğe bağlı native split çakışma denetimi

- `vegai_corpus_overlap`, UTF-8 tam satırlarını held-out indeksiyle eşleştirir;
  eğitim yoluna otomatik eklenmez. Çalıştırıldığında iki dosyayı okumak ve
  held-out indeksini tutmak için ayrı I/O/bellek maliyeti vardır. Hash adayları
  tam bayt/uzunluk karşılaştırmasıyla doğrulanır; çıktı 0/1/2 ile ayrıştırılır.
- Son CTest: seri Release **36/36**, iki-thread OpenMP Release **37/37** ve
  OpenMP/FPE/`-ftrapv` Debug **37/37**. Python'suz Make hedefi, FPM seri/OpenMP
  uygulaması ve 250 rasgele case'lik
  bağımsız Python oracle karşılaştırması geçti. Python oracle yalnız geliştirme
  testidir; native aracı çalıştırmak için Python gerekmez. Test 4.096/4.097
  bayt, 100.000/100.001 benzersiz kayıt, 64 MiB/sonraki kayıt ve 1.024/1.025
  gerçek hash çakışmasının iki sınırını; ayrıca Linux'ta yazarı olmayan FIFO'nun
  `OPEN` öncesi reddini doğrular. En küçük int64 hash değeri ilk checked-compiler
  koşusunda `MODULO` içinde SIGABRT üretti; bucket hesabı taşmasız unsigned-word
  aritmetiğine taşındı ve beş uç hash bit deseniyle tekrar geçti. Bu makinede
  BOUN train-200/train-1000 ile dev-80/test-80 dört eşleşmesinde tam
  satır örtüşmesi sıfırdı.
- Bu yalnız tam satır denetimidir; farklı normalizasyon, kısmi veya semantik
  sızıntıları ve aynı boyutlu eşzamanlı girdi mutasyonunu tespit etmez. İmmutable
  girdi, kaynak yönetişimi ve gerektiğinde harici snapshot/kimlik kontrolü hâlâ
  gereklidir; ayrıntılar `CORPUS_OVERLAP.md` içindedir.
