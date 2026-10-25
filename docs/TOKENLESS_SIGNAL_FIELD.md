# Tokenless Signal Field — sınırları tanımlı byte-event temsili

## Amaç ve dürüst kapsam

`src/tokenless_field.f90`, Vegai'nin ASCII kelime tokenizer'ından ve harici
LMM tokenizer paketlerinden **ayrı** çalışan native bir temsildir. Girdi olarak
ham dosya octet'lerini (`0…255`) doğrudan alır; BPE/WordPiece/Unigram merge'i,
öğrenilmiş vocabulary, Unicode normalizasyonu, sözlük lookup'ı veya prompt
şablonu kullanmaz. UTF-8 metin bir örnektir: modül UTF-8'i çözmez, yalnızca
orijinal byte dizisini event olarak işler. Geçersiz UTF-8 de aynı nedenle
reddedilmeden byte dizisi olarak temsil edilebilir.

Bu, öğrenilmiş subword tokenizer'a bağımlı olmayan özgün bir giriş yolu sağlar;
**bir dil anlayışı veya evrensel LLM/LMM ikamesi sağlamaz**. Sabit byte alfabesi
bir fiziksel/iletişim kodlama birimidir, semantic vocabulary değildir; yine de
hiçbir sonlu bilgisayar temsili “birimsiz” değildir. Buradaki doğru iddia
şudur:

> En fazla 32.767 octet'lik tek bir bounded input için belirlenen bit-cebiri,
> Walsh dönüşümü ve hata sınırları deterministik ve çalıştırılabilir olarak
> doğrulanır.

Aşağıdaki iddialar **yapılmaz**:

- ham byte field'in doğal dil, kod, görüntü, ses veya video anlamını otomatik
  olarak anladığı;
- field'in kaynak byte dizisini lossless olarak sakladığı;
- herhangi bir gerçek-dünya gözleminin, fizik yasasının ya da ölçümün doğru
  olduğu;
- sistemin LLM/LMM'den her domain'de daha zeki, daha doğru veya daha hızlı
  olduğu;
- field'in serbest metin üretimi, next-token olasılığı, autodiff veya model
  eğitimi yaptığı.

Gerçek neural generation/multimodal encoding gerektiğinde mevcut
[LMM runtime sınırı](LMM_RUNTIME.md) korunur. HDC retrieval, `reasoning_core`
ve Resonator Network, field üzerinde operator-supplied sembol/evidence ile
çalışabilir; semantic doğruluk yine o evidence'in yönetişimi ve held-out
hedef-domain değerlendirmesiyle kanıtlanmalıdır.

## Temsil sözleşmesi

Boyut, mevcut native çekirdekle aynıdır:

```text
D = HV_BITS = 16,384 = 2^14
packed payload = 256 × int64 = 2,048 byte
```

Bir paketli field biti bipolar değere çevrilir:

```text
x_i = +1  bit i = 1 ise
x_i = -1  bit i = 0 ise
```

Her raw byte `b` için sabit domain-separated seed'den `B(b)` atomu; bir-based
pozisyon `p` için ayrı seed'den `O(b,p)` atomu üretilir. Seed sabitleri
representation parametresidir; corpus, vocabulary, model weight veya hardcoded
cevap değildir. Bir stream `b₁ … bₙ` için modül iki field yayınlar:

```text
order_field        = O(b₁,1) XOR O(b₂,2) XOR ... XOR O(bₙ,n)
distribution_field = deterministic-majority(B(b₁), ..., B(bₙ))
```

`distribution_field` kasıtlı olarak event sırasına duyarsızdır. `order_field`
pozisyonu event'e katar ve test edilen bir reorder'da değişir; buna rağmen XOR
ve sabit boyut yüzünden çakışmasız veya lossless sequence encoding olduğu
iddia edilmez. `field_fingerprint` da yalnız bir rolling XOR/rotation teşhis
özetidir; kriptografik digest veya source identity değildir.

`field_encode_bytes` şu fail-closed koşulları uygular:

| Koşul | Sonuç |
|---|---|
| `n = 0` | `FIELD_ERR_EMPTY_INPUT` |
| `1 ≤ n ≤ 32767` | encode edilir |
| `n > 32767` | `FIELD_ERR_CAPACITY`; truncate/partial output yok |
| HDC signed accumulator taşması | `FIELD_ERR_CAPACITY`; output sıfırlanır |
| CLI file inquiry/open/read hatası | `FIELD_ERR_IO`; partial field yayınlanmaz |

32.767 limiti keyfi throughput iddiası değildir. Her majority counter `int16`
olarak saklanır ve en fazla `+32767` / `-32767` unit vote alır; böylece sayaç
wraparound'a girmeden sınır açıkça doğrulanır. Daha uzun stream'ler için
yukarı katman source-digest/provenance bağlı chunk sınırı seçmelidir. Bu modül
chunk'ları sessizce kesmez veya “lossless birleşti” iddiasında bulunmaz.

## Kesin sonlu Walsh sözleşmesi

`D=2^14` olduğu için unnormalised Walsh-Hadamard matrisi `H` uygulanır:

```text
H_ij ∈ {-1,+1}
HᵀH = D I
s = Hx
x = Hs / D                 (yalnız geçerli bipolar field spektrumu için)
||s||² = D ||x||² = D² = 268,435,456
```

Bu son satır Parseval eşitliğinin bu sınırlı dijital temsildeki tam halidir.
`field_walsh_forward`, `field_spectral_energy`,
`field_parseval_residual` ve `field_walsh_inverse` bu sözleşmeyi yürütülebilir
kılar.

### Integer güvenlik hesabı

Kod signed overflow davranışına dayanmaz:

| Aşama | Kabul edilen / kanıtlanan mutlak üst sınır | `int64` durumu |
|---|---:|---|
| forward WHT ara katsayısı | `D = 16,384` | güvenli |
| forward spectrum katsayısı | `D = 16,384` | güvenli |
| inverse WHT ara katsayısı | `D² = 268,435,456` | güvenli |
| aralık kontrollü spectral energy toplamı | `D³ = 4,398,046,511,104` | güvenli (`< 2^63`) |
| beklenen bipolar Parseval enerjisi | `D² = 268,435,456` | tam temsil |

Inverse önce her coefficient'in `[-D,D]` içinde olduğunu denetler; ikinci WHT
sonrasında her değerin `D`'ye tam bölündüğünü ve sonucu yalnız `-1` veya `+1`
olduğunu denetler. Bu nedenle keyfi bir `int64` dizisini sessizce field'e
çevirmez. Geçersiz spectrum `FIELD_ERR_INVALID_SPECTRUM` üretir ve sıfır
field döndürür.

Bu ispat **yalnızca** bu finite integer programının aritmetiğine aittir.
Parseval doğruluğu bir sensörün kalibrasyonunu, bir deneyin modelini, fiziksel
birimin doğru seçimini ya da dış dünya iddiasını doğrulamaz. Böyle bir iddia
ayrı olarak ölçüm cihazı, unit/provenance, kalibrasyon belirsizliği, hata
modeli ve domain validation gerektirir.

## Kayıp ve provenance sınırı

Sabit boyutlu bir `D`-bit field'in olası durumu en fazla `2^D`'dir; buna karşı
arbitrary uzunlukta byte stream kümesi sınırsızdır. Pigeonhole prensibi gereği
tek bir field arbitrary stream'ler için injective olamaz. Bu sebeple:

1. raw source dosyası veya immutable object reference saklanır;
2. source için harici kriptografik digest (ör. SHA-256), source/license ve
   preprocessing provenance tutulur;
3. field, digest'in yerine değil digest'e bağlı derived artifact olarak
   yayınlanır;
4. collision-sensitive kararlar field fingerprint ile değil tam source identity
   ve gerekiyorsa tam HDC equality/evidence ile doğrulanır.

Vegai'nin `corpus_manifest.py`, production dataset manifest ve capability
curriculum source-snapshot mekanizmaları bu ayrı identity/provenance plane için
kullanılabilir. `vegai_field` bilerek SHA-256 implement etmez: çıktısındaki
`raw_source_identity_requirement` satırı, kısa bir non-cryptographic
fingerprint'i provenance kanıtı gibi göstermeyi engeller.

## CLI ve doğrulama

Derleyin ve raw binary file inceleyin:

```bash
make field
printf 'Merhaba\n' > /tmp/raw-input.bin
./build/vegai_field inspect /tmp/raw-input.bin

# Python bağımsız oracle + derlenmiş CLI contract'i
make field-check
```

CMake yolunda `vegai_field` ayrı executable'dır ve CTest'e
`vegai_tokenless_signal_field` olarak kaydedilir:

```bash
cmake -S . -B build -DVEGAI_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R tokenless
```

`inspect` raporu şunları sabit formatta yayınlar:

- input byte sayısı ve explicit maksimum;
- vocabulary/learned-tokenizer kullanılmadığı;
- raw input'un field içinde lossless olmadığı;
- order/distribution field için **non-cryptographic** fingerprint;
- exact Walsh energy, Parseval residual ve inverse bit equality;
- semantic veya physical-world claim yapılmadığı.

Doğrulama üç bağımsız yüzeydedir:

1. `test/test_all.f90`: arbitrary signed `int8` octet'ler, determinism,
   order/distribution davranışı, forward/inverse, Parseval, invalid spectrum,
   empty ve capacity hata yolları;
2. `tools/verify_tokenless_field.py`: dependency-free Python cebir oracle'ı;
3. oracle'a `--field-bin` verildiğinde raw file CLI report'unu ve missing/empty/
   oversize fail-closed davranışını ayrı process olarak sınar.

Test fixture'ları representation mekanizmasını sınar; dil facts'i, prompt,
cevap veya training example değildir.

## Üretim kullanımı için doğru karar kuralı

Bu modül şu dar problem sınıfında anlamlı bir building block'tur: raw binary
olayların deterministic, bounded, vocabulary-free HDC feature field'e
indirgenmesi; ardından operator tarafından tanımlanmış symbol/evidence
memory'sinde retrieval veya bounded constraint/factor işlemi. Bir deployment
öncesinde en az şunlar gereklidir:

1. field'e giren byte formatı, canonicalization policy'si (burada yoksa
   `none`) ve chunk boundary açıkça versionlanmalı;
2. raw source identity, license/consent ve preprocessing zinciri digest ile
   bağlanmalı;
3. target-domain ground truth, held-out split ve appropriate loss/quality
   metriği tanımlanmalı;
4. false positive/false negative, collision, abstention ve resource limitleri
   release gate'te ölçülmeli;
5. external-world veya physical decision için uncertainty/calibration/safety
   kontrolü field cebirinden ayrı uygulanmalı.

Bu koşullar sağlanmadan “tokenizer-free” ifadesi yalnız giriş mekanizmasını
tanımlar; zeka, anlam, fiziksel doğruluk ya da universal üstünlük kanıtı
anlamına gelmez.
