# Doğal dil verisinde byte-model değerlendirmesi

`vegai_natural_eval`, mevcut 256-byte V5/V6/V7 prompt checkpoint'ini **eğitmeden**
UTF-8 satır dosyasında puanlar. Python, ağ erişimi ve tokenizer gerekmez. Bu
isteğe bağlı ayrı okumadır; eğitimde ek tam dizi taraması, optimizer güncellemesi,
checkpoint kopyası veya eğitim iç döngüsüne ek iş yapmaz.

```sh
# CMake varsayılan derlemesinde hedefler; GNU Make eşdeğeri: make natural-eval
cmake --build build --target vegai_natural_eval vegai_corpus_overlap
# Önce tam satır çakışmasını kontrol edin; bu semantik sızıntı kontrolü değildir.
build/vegai_corpus_overlap train.txt heldout.txt
build/vegai_natural_eval model.vgfb heldout.txt 32
```

Her **LF kaydı** bağımsız bir örnektir: model state'i sıfırlanır, ilk byte
bağlam olarak verilir ve kalan byte'lar gerçek önceki byte'larla
(*teacher forcing*) tahmin edilir. İlk byte, LF ve satırlar arası geçişler
skorlanmaz. `mean_nll`, tüm puanlanan byte'lara ağırlıklı ortalama negatif
log-olabilirliktir (nat/byte; düşük daha iyi). `prefix_mean_nll` yalnız solda
1 ile belirtilen sınır arasında byte bağlamı olan tahminlerin ortalamasıdır;
varsayılan 32, kabul edilen aralık 1–4096 byte'tır. Bu Unicode karakteri veya
sözcük sayısı değildir. Boş ve tek byte'lı satırlar `records` sayısına girer
ama prediction üretmez; `scored_records`, `predictions` ve
`prefix_predictions` ayrıca raporlanır. Skorlanacak byte yoksa işlem hata verir.
Çıktı `vegai-cold-record-eval/v1` JSON şemasındadır.

Girdi LF ile ayrılmış, NUL içermeyen geçerli UTF-8 olmalıdır. Her kayıt en
fazla 4096 byte olabilir; daha uzun veya hatalı kayıt ve bilinmeyen/sıfır dosya
boyutu reddedilir. Son LF zorunlu değildir. **CR normalize edilmez:** CRLF'deki
CR, satırın bir byte'ı olarak puanlanır. Unicode normalizasyonu ve örtüşme
ayıklaması otomatik yapılmaz. Dosya açılmadan önce ve okuma öncesi/sonrası
boyut kontrol edilir; sıfır veya bilinmeyen boyut (GNU/Linux'ta FIFO dahil)
reddedilir. Bu bir kilit/hash değildir; aynı boyutlu eşzamanlı değişiklikleri
ve ön kontrolden sonra yol değiştirme yarışını saptamaz. Güvenilir dizinde
immutable veri ve bağımsız dosya hash/manifesti
kullanın. Bu araç bellek kullanımını kayıt başına sınırlar; sonuç dosyası,
checkpoint veya veri değiştirmez. Standart Fortran farklı adların symlink veya
hardlink alias'ı olup olmadığını da garanti etmez.

## Eşit bütçeli karşılaştırma

`vegai_natural_quality_ab`, aynı V6 başlangıç modeli, seed, metin sırası,
şekil, token/prediction ve update sayısı ile iki öğrenme oranını karşılaştırır;
her iki model iki ayrı held-out dosyasında **birer kez eğitildikten sonra**
değerlendirilir. Beş seed ve dönüşümlü kol sırası kullanılır. Süre yalnız eğitim
çağrısıdır: okuma, değerlendirme ve checkpoint I/O dahil değildir. Kısa ve
uzun çalışmaları ayrı raporlayın; veri setleri ve kaynakları ayrı tutun.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVEGAI_BUILD_TESTS=ON
cmake --build build --target vegai_natural_quality_ab
# train heldout-A heldout-B chunk base-LR-milyonda small|prompt aday-LR-milyonda
build/vegai_natural_quality_ab train.txt dev-a.txt dev-b.txt 512 6000 prompt 10000
# GNU Make: make natural-quality-ab natural-quality-contract-check
```

Araç train ön ekini tam chunk katına kırpar; gerçek prompt CLI ise son kısa
chunk'ı da işler. Aynı üç dosya yolu ve byte-byte aynı dosyalar reddedilir,
ancak kısmi/semantik veri sızıntısı veya dosya alias'ı bu araçla dışlanmaz.
Üç ölçüt raporlanır: sıfırdan başlatılmış **kesintisiz akış** NLL'si, her LF
satırında reset edilen **cold-record** NLL'si ve ilk 32 bayt bağlamı içindeki
**prefix** NLL'si. Her metrik için medyan aday−temel farkı ve NLL'si düşen
seed sayısı da ayrıca verilir. Tüm seçimler test kümelerine bakılmadan dev
üzerinde belirlenmeli; veri kaynağı, lisans, hash, seed ve olumsuz sonuçlar da
saklanmalıdır.

Byte NLL, geçerli UTF-8 üretimi veya küçük bir corpus sonucu; anlamayı,
konuşma kalitesini, güvenliği, endüstri modellerinden üstünlüğü ya da başka
OS/donanım/veri ve uzun bütçelerde hız garantisini **kanıtlamaz**. Ayrı görev
setleri, gerçek kullanıcı değerlendirmesi ve hedef ortamda ölçüm gerekir.
