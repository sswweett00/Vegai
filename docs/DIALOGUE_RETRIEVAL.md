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

- `src/dialogue_retrieval.f90`: bounded index, deterministic prompt encoding,
  confidence/margin gate ve bounded turn history. Corpus training veya
  checkpoint semantigini degistirmez.
- `app/chat.f90`: `ask` ve `repl` CLI adaptoru. Cevap üretmez; yalnizca modülün
  sonucunu raporlar.
- `tools/verify_dialogue_retrieval.py`: gerçek derlenmis `vegai_pretrain` ve
  `vegai_chat` ile runtime TSV cevabi, abstention, strict failure ve REPL
  contract'ini denetler.

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

Runtime fail-closed kurallari:

- yalniz LF ve `<= 8.192` byte fiziksel kayit;
- CR/CRLF, `>127` byte, ikinci TAB, bos prompt/response reddedilir;
- prompt en cok 1.024, response en cok 2.048 printable ASCII byte olabilir;
  oversized `--prompt` veya REPL physical record'u truncate edilmeden reddedilir;
- bilgi bankasi prompt'undaki **her** token yuklenmis checkpoint'in active
  vocabulary'sinde olmalidir; OOV anahtar sessizce indexlenmez;
- ayni HDC prompt embedding'i (response ayni olsa bile) ikinci kez yayinlanamaz;
- `--max-entries` kapasitesi asilirsa index buyumez;
- herhangi bir TSV publication/load hatasinda partial index destroy edilir ve
  `entries_loaded=0` olur; duzeltilmis revision icin operator yeniden init/load
  etmelidir.

Cevap alani tokenization veya execution icin kullanilmaz; display edilecek
operator verisidir. Bu nedenle bilgi bankasini guvenilir/veri-yonetisiminden
gecmis bir kaynak olarak ele alin. Bir response içindeki instruction benzeri
metin executable degildir, fakat kullaniciya oldugu gibi gösterilir.

Mevcut core ASCII sozlesmesi Unicode Turkce'yi dogrudan kabul etmez. Upstream
normalizasyon/transliteration surumu, source license'i ve bilgi-bankasi hash'i
kaydedilmelidir. `examples/dialogue/mini_turkish_hdc_dialogue.tsv` tamamen
sentetik, ASCII transliterasyonlu bir integration fixture'dir; gercek bilgi veya
kalite iddiasi degildir.

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

1. Prompt ASCII tokenizer ile canonicalize edilir.
2. Active checkpoint semantic hypervector'leri deterministic weighted majority
   bundle'a cevrilir.
3. Son `--history-turns` user prompt'u varsa, current prompt weight'i ve
   history weight'i ile ikinci bounded bundle olusturulur. Varsayilan policy
   current=`2`, her history turn=`1`, history capacity=`3`tur; tumu CLI ile
   degistirilebilir, `--history-turns 0` state'i kapatir.
4. Index tum entry'leri exact Hamming mesafesiyle tarar. En iyi response ancak
   `--min-similarity` ve en iyi/ikinci-en-iyi `--min-margin` esiklerini
   gecerse döndürülür.

Exact tarama deterministik ve recall kayipsizdir, fakat bir turn icin `O(N ×
16.384-bit)` maliyetlidir. Bu modul uzak/dağıtık milyarlarca dokuman retrieval
servisi oldugunu iddia etmez. `--max-entries` ile RAM siniri aciktir; index
yaklasik olarak entry basina 2.048 byte prompt HV + 1.024 byte prompt + 2.048
byte response + sayaçlar kullanir. CLI `knowledge index resident bytes` ile
veri-duzlemi estimate'ini yayinlar. Daha buyuk deployment'ta index sharding,
external search backend ve versioned retrieval evaluation ayrı bir adapter
karari olmalidir; HDC exact semantics gizlice approximate result olarak
sunulmaz.

`--require-all-known`, live prompt'ta da OOV token goruldugunde fail-closed
ister. Varsayilan daha esnek mod, en az `--min-known-tokens` active token varsa
bilinmeyen filler token'lari sayar ve raporlar; cevap yine similarity/margin
kapisindan gecmek zorundadir.

## Test contract'i

CTest'teki `vegai_dialogue_retrieval` gerçek Fortran binary'leriyle:

- runtime TSV tarafindan belirlenen cevabin aynen döndüğünü;
- ayni prompt icin yalniz TSV response'u degistirmenin user-visible cevabi da
  degistirdigini (source-code cevabi olmadigini);
- known ama uygun olmayan prompt'un abstain etmesini;
- OOV user prompt, malformed TSV ve capacity/duplicate prompt error'larini;
- stateful `repl` yolunu;
- native Fortran property testinde bounded session ring, strict TSV load,
  deterministic exact retrieval ve reset davranisini denetler.

Bu testler retrieval/data-contract dogrulamasidir. Held-out natural-language
kalitesi, güvenlik veya factual accuracy icin versioned bilgi bankasi, source
review, held-out query seti ve insan/task-specific degerlendirme gerekir.
