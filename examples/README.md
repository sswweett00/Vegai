# Çalıştırılabilir küçük örnekler

`corpus/toy_hdc_corpus.txt`, lisans/bakım yükü oluşturmayan küçük, ASCII-only
bir smoke-test corpus'udur. Anlamlı kalite ölçümü veya geniş dil kapsamı iddia
etmez; yalnız iki corpus geçişi, checkpoint ve query akışını göstermeye
yarar.

```bash
# CMake ile derlendiyse
./build/vegai_pretrain train examples/corpus/toy_hdc_corpus.txt toy.vgai \
  --max-vocab 64 --min-frequency 2 --epochs 2 --query king

./build/vegai_pretrain inspect toy.vgai --query dog
./build/vegai_pretrain resume examples/corpus/toy_hdc_corpus.txt toy.vgai \
  --epochs 1 --query queen
```

`toy.vgai` çalışma çıktısıdır; gerçek bir corpus veya checkpoint'i depoya
commit etmeyin. Veri kaynağı, lisans, rıza/kişisel veri analizi, preprocessing
komutu ve corpus digest'i proje dışı bir deney manifestinde kaydedilmelidir.
Yerel corpus digest/preflight için `python3 tools/corpus_manifest.py CORPUS
--output manifest.json ...` kullanılabilir.
Ayrıntılı veri yönetişimi ve model semantiği için
[`docs/PRETRAINING.md`](../docs/PRETRAINING.md) belgesine bakın.
