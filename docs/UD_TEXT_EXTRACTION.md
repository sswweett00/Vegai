# UD metnini Python olmadan hazırlama

`vegai_ud_text_extract`, bir CoNLL-U dosyasındaki ilk **N** adet tam
`# text = ` yorumunun metin kısmını, her kayıt sonunda LF olacak şekilde yeni
bir dosyaya çıkaran isteğe bağlı Fortran aracıdır. Veri hazırlama **çevrimdışı**
yapılır; eğitim koduna ek okuma, kopya, gradyan veya güncelleme eklenmez.
Kaynak korpusları ve lisanslarını kullanıcı sağlar; bu araç kaynak veriyi
indirmez veya depoya koymaz.

```sh
# CMake: varsayılan native derleme / hedef
cmake --build build/portable-serial --target vegai_ud_text_extract vegai_control
./build/portable-serial/vegai_ud_text_extract tr_boun-ud-train.conllu train-200.txt 200
./build/portable-serial/vegai_control corpus-manifest train-200.txt \
  --expect-sha256 0889ecce1aabec92284cb4cb3994aaa1e862609487a2adc526fa02f71622ac74

# GNU Make; test yürütücüsü CMake'tir, denetlenen program Python çağırmaz.
make ud-text-extract ud-text-extract-check

# FPM otomatik uygulama adı; argümanlar `--` sonrasında verilir.
fpm run --target ud_text_extract -- tr_boun-ud-train.conllu train-200-fpm.txt 200
```

Örnek komutta ilk N satırın kimliği yalnızca belgelenen, sabitlenmiş BOUN
revision'ı için beklenir; başka sürümler aynı hash'i vermeyebilir. `train-200.txt`
önceden **var olmamalıdır**. `corpus-manifest`, ayrıca HDC tokenizer sınırlarını
raporlar; Türkçe byte-model verisindeki non-ASCII karakterler HDC'nin ASCII
sözleşmesine uygun olmayabilir. Bu örnekte `--strict-contract` kullanılmaz;
hash eşleşmesi dil veya içerik kalitesini onaylamaz.

## Sözleşme ve sınırlar

- Sayı `1..10.000.000` arasında tamsayıdır. `# text = ` öneki büyük/küçük
  harfe ve boşluklara duyarlıdır; başka CoNLL-U yorumları ve token satırları
  atlanır. Bu bir CoNLL-U sözdizimi/etiket doğrulayıcısı değildir.
- LF kayıt ayırıcıdır; fiziksel kaydın sonundaki tek CR kaldırılır. Metindeki
  diğer byte'lar, sondaki boşluk ve tab dahil aynen korunur; Unicode
  normalizasyonu yapılmaz. Son kaynak kaydında LF zorunlu değildir.
- İşlenen fiziksel kayıtların her biri en fazla **65.536 byte** olabilir;
  aşılırsa hata verilir. Seçilen metin boş, NUL içeren veya geçersiz UTF-8
  ise hata verilir. Araç istenen N kayıt bulunduğunda durur; sonrasını ve
  seçilmeyen yorumların UTF-8 geçerliliğini doğruladığını iddia etmez.
- Bilinen, sıfırdan büyük boyutlu değişmez dosya beklenir. Boyut değişimleri
  denetlenir; bu denetim kilit değildir, aynı boyutlu eşzamanlı mutasyonu veya
  kötü niyetli yol değiştirmeyi yakalamaz. Pipe/özel aygıtlar desteklenen
  sözleşme değildir. Güvenilir dizin ve değişmez input/snapshot kullanın.
- Mevcut çıktı `status='new'` ile açılmaz; başarısız okuma, sayım, UTF-8 veya
  yazmada yeni oluşturulan kısmi çıktı silinmeye çalışılır. Silme/kapama
  hatasında dosya kalabilir; nonzero çıkıştan sonra artığı kullanmayın.
  Standart Fortran her OS'te atomik dosya oluşturma/silme veya yarışsız alias
  tespiti garantisi vermez; özellikle paylaşılan ya da kötü niyetli dizinler
  için işletim sistemi düzeyinde koruma gerekir.
- Başarı çıkışı JSON `schema`, `records`, `output_bytes` içerir; hata çıkışı
  stderr tanısı ve nonzero koddur, başarılı gibi görünen stdout üretilmez.
  En az istenen sayı kadar kayıt yoksa çıktı eksik diye reddedilir.

`test/native_ud_text_extract.cmake` Türkçe baytlar, CRLF, sondaki boşluk/tab,
son-LF'siz girdi, 65.536/65.537 sınırı, NUL/geçersiz UTF-8, eksik sayı,
geçersiz limit ve mevcut çıktının korunmasını sınar. Windows/macOS çalışması
ancak ilgili gerçek CI sonucu görüldüğünde doğrulanmış sayılmalıdır.
