# Native train/held-out satır çakışması kontrolü

`vegai_corpus_overlap`, eğitim dosyası ile ayrı bir held-out dosyasında aynı
**tam metin kaydının** bulunup bulunmadığını denetleyen isteğe bağlı, salt-okunur
Fortran aracıdır. Python gerekmez. Eğitim koduna bağlanmaz: çağrılmadığında ek
okuma, allocation, gradient güncellemesi veya checkpoint kopyası yapmaz. Araç
çağrıldığında ise iki dosyayı ayrıca okuyup held-out kayıtlarını indeksler;
bu I/O ve belleğin gerçek maliyeti vardır ve eğitim süresine gizlenmemelidir.

```sh
make corpus-overlap
./build/vegai_corpus_overlap train.txt heldout.txt
# Test runner olarak CMake kullanır; kontrol edilen program Python çağırmaz.
make corpus-overlap-check
```

CMake derlemesinde hedef `vegai_corpus_overlap`, FPM'de uygulama adı
`corpus_overlap`'tır. Çıkış kodları:

- **0**: İşlenen tam kayıtlar arasında çakışma yok; stdout JSON raporu içerir.
- **1**: En az bir çakışma var; stdout JSON raporu içerir.
- **2**: Argüman, I/O, veri, platform veya kapasite hatası; stderr tanısı vardır,
  stdout üzerinde başarılı gibi görünen rapor üretilmez.

Rapor, boş olmayan train/held-out satır sayılarını, held-out benzersiz kayıt
sayısını, çakışan train satırı sayısını ve en az bir kez eşleşen benzersiz
held-out kayıt sayısını içerir. `first_train_line` ve `first_heldout_line` ilk
çakışan train satırının ve eşleşen kaydın held-out'taki ilk fiziksel satır
numarasıdır; hiç çakışma yoksa sıfırdır. Boş satırlar sayılmaz ama fiziksel
satır numaralarında yer alır. Kayıt içeriği raporda yayımlanmaz.

## Tam eşitlik ve sınırlar

- Girdi, UTF-8 doğrulanan baytlardır; NUL ve geçersiz UTF-8 açık hatadır. LF
  ayraçtır; her kaydın sonundaki **tek CR** kaldırılır, böylece LF ve CRLF
  satırları karşılaştırılır. Boş kayıtlar atlanır; sondaki boşluklar, tab'lar,
  büyük/küçük harf ve kalan bütün baytlar anlamlıdır. Son LF zorunlu değildir.
- Her ham kayıt en çok 4.096 bayt olabilir. Held-out indeksi en çok 100.000
  benzersiz kayıt ve toplam 64 MiB kayıt verisi tutar; bu rakama veri yapısı
  ve çalışma zamanı ek yükü **dahil değildir**. Tek hash bucket zinciri
  en çok 1.024 kayıtla sınırlıdır. Bir sınır aşılırsa kontrol başarısız olur;
  kayıtlar sessizce atlanıp yanlış `overlap=false` raporu üretilmez.
- Hash yalnız indeksleme içindir: her adayda uzunluk ve bütün baytlar ayrıca
  eşitlik için karşılaştırılır. Dolayısıyla hash çakışması tek başına veri
  çakışması sayılmaz. Hash kriptografik kimlik veya provenance değildir.
- Dosyalar bilinen, sıfırdan büyük boyutlu ve tarama boyunca değişmeyen normal
  dosyalar olmalıdır. Boyut değişimi kontrol edilir; bu bir kilit/snapshot
  değildir ve **aynı boyutlu eşzamanlı mutasyonu tespit etmez**. Üretim
  karşılaştırması için değişmez girdi veya harici snapshot/kimlik denetimi
  kullanın; pipe ve özel aygıtlar desteklenen sözleşme değildir. Bilinen
  sıfır/negatif boyut `OPEN` öncesinde reddedilir; bu, dosya yolunun kötü niyetle
  değiştirilmesine karşı yarışsız bir filesystem garantisi değildir.
- Bir tam satır eşleşmesinin olmaması; satır parçalarının, farklı Unicode
  normalizasyonlarının, yeniden ifade edilmiş metnin, yakın kopyaların veya
  kaynak düzeyindeki sızıntının olmadığını kanıtlamaz. Bu araç tek başına split
  yönetişiminin, semantik leakage denetiminin ya da held-out kalite ölçümünün
  yerine geçmez. Boyut sınırı nedeniyle büyük held-out kümelerini parçalarken
  **tüm** parçalar ayrı ayrı aynı train üzerinde denetlenmelidir.

Native sözleşme testi gerçek hash çakışmalarını, tekrarları, CRLF/LF ve sondaki
boşluğu, geçersiz UTF-8/NUL'u ve sınırların iki tarafını (4.096/4.097 bayt,
100.000/100.001 benzersiz kayıt, 1.024/1.025 hash zinciri, 64 MiB/sonraki
kayıt) ve int64 hash'in en küçük/büyük bit desenlerini çalıştırır. POSIX'te
`mkfifo` varsa yazarsız FIFO testi de yapılır.
Girdi çakışması ve I/O/limit hataları ayrık nonzero çıkış kodlarıdır.
