# LMM portföy operator şablonları

Bu dizin gerçek model veya benchmark sonucu içermez. JSON dosyaları,
`tools/lmm_portfolio.py` için operatorün kopyalayıp tamamlayacağı strict schema
şablonlarıdır:

- `portfolio-template.json`: candidate manifest, gateway command, quality report
  ve evaluation card yollarını bağlar;
- `routing-task-coding-template.json`: coding domain için residency, safety,
  modality, context, latency ve cost hard gate'lerini gösterir;
- `evaluation-card-template.json`: governed target-domain ölçüm card'ının
  alanlarını gösterir;
- `gateway-command-template.json`: private `{request}` / `{evidence}` argv
  binding'ini gösterir.

`evaluation-card-template.json` içindeki sıfır SHA-256 değerleri **kasıtlı
placeholder**'dır. Bunlar gerçek score, gerçek quality report veya çalıştırılabilir
release artifact'i değildir. Publication öncesinde aşağıdakilerin tamamı gerçek,
immutable ve birbirine hash-bound olmalıdır:

1. lisanslı model package manifest'i;
2. ilgili modelle üretilip gate'i geçen `lmm_quality_evaluation.py` report'u;
3. ayrı governed held-out domain suite report'u ve lower-bound istatistikleri;
4. aynı `cost_unit` ile ölçülen latency/cost değerleri;
5. model manifest'iyle eşleşen, no-shell gateway argv command'i.

Örnek tamamlandıktan sonra repository kökünden aşağıdaki gibi route edin:

```bash
python3 tools/lmm_portfolio.py route \
  --portfolio /secure/portfolio.json --task /secure/coding-task.json \
  --request /secure/request.json --evidence /secure/evidence.json \
  --output lmm-runs/route.json
```

Çalışan ama yalnız protocol amaçlı sentetik fixture,
`tools/verify_lmm_portfolio.py` tarafından temporary workspace'te üretilir ve
CI'da doğrulanır. Onun ölçüm değerleri gerçek model kalitesi veya domain
üstünlüğü iddiası değildir. Ayrıntılı şema ve operasyon sınırları için
[`docs/LMM_PORTFOLIO_ROUTING.md`](../../../docs/LMM_PORTFOLIO_ROUTING.md)'ye
bakın.
