# LMM/LLM kapsam analizi ve Vegai üretim sınırı

Bu belge, repodaki kaynak, CLI, checkpoint, test ve operational tool'ların
incelenmesinden sonra mevcut yetenek ile gerçek bir büyük dil/multimodal model
(LMM/LLM) training system'i arasındaki sınırı açıklar. Amaç capability
inflation'ı önlemektir: orchestration eklemek, HDC modelini Transformer veya
multimodal neural runtime yapmaz.

## İncelenen mevcut sistem

| Alan | Vegai'de doğrulanmış mevcut durum |
|---|---|
| Core representation | Fortran 2023 packed 16.384-bit hypervector/VSA (`hyper_types`, `symbolic_memory`) |
| Learning | iki-pass ASCII corpus, frequency vocabulary, deterministic HDC context bundle ve source-ordered Hebbian prototype update (`hdc_pretraining`) |
| Factor reasoning | bounded exact certificate ile Resonator Network multi-factor factorization |
| Determinism | fixed seeds, bounded allocation/error contracts, serial/OpenMP checkpoint byte equality, checkpoint/resume |
| Continual data | strict ASCII/LF SHA-256 immutable shards, private snapshot/candidate generation/atomic state journal ile exactly-once commit |
| CPU operation | portable/native Fortran build, OpenMP immutable-context preparation, conservative RAM/checkpoint preflight |
| Optional Intel boundary | OpenVINO yalnız supported positional/hybrid/hierarchical immutable packed-HDC context preparation için; explicit CPU policy native fused kernel'dir. Center-dependent coherent cross-fold mode CPU/OpenMP-only'dir; GPU/NPU execution ayrıca doğrulanmalıdır |
| Yeni production control plane | disk-backed exact raw-record dedupe, content-addressed shards, deterministic held-out split, exact leakage gate, max-vocabulary capacity preflight, binary/data/policy provenance |
| Grounded diyalog retrieval | checkpoint semantic HV + operator TSV prompt/response index, exact Hamming retrieval, confidence/margin abstention ve bounded turn history; cevap TSV'den gelir |
| Grounded retrieval quality gate | versioned held-out JSONL, stateless real-binary exact prompt/abstention metrikleri, immutable input digest binding, privacy-conscious report ve explicit release threshold'lari |

Bu sistem, bounded semantic prototype/retrieval ve symbolic factor reasoning için
üretim kalitesinde operation sözleşmeleri taşıyabilir. Model kapasitesi ve
öğrenme semantiği HDC slot/vocabulary/context yapısına bağlıdır; neural parameter
sayısı, next-token likelihood veya generative model kapasitesi değildir.

`vegai_chat` bu sınırı degistirmez: user-visible metin, operatorun versioned TSV
bilgi bankasindan retrieval ile gelir; model yeni paragraf/token sample etmez.
Similarity/margin gate'i gecilmezse abstain eder. Bu, hardcoded reply yerine
data-driven grounded UX icin yararlidir; LLM sohbet zekasi, factual guarantee
veya safety policy yerine gecmez.

Yeni held-out evaluator bu dar retrieval contract'inde bilinmeyen kalite
durumunu azaltir: expected prompt veya expected abstention'i gerçek compiled
chat binary'si ile olcer ve release threshold'unu fail-closed uygular. Ancak
LMM evaluation gate satirindaki eksigi kapatmaz; perplexity, multimodal task,
contamination, factuality veya safety/red-team evaluation yapmaz.

## Gerçek LMM/LLM için eksik zorunlu katmanlar

Aşağıdakiler "flag" veya wrapper ile eklenebilecek küçük eksikler değildir;
yeni data plane, model runtime ve test yüzeyi gerektirir.

| Zorunlu LMM katmanı | Vegai'deki durum | Neden mevcut HDC katmanıyla ikame edilemez |
|---|---|---|
| Unicode/subword tokenizer | Yok; `text_tokenizer` ASCII-byte, 48-byte token, 1.024 token/record sözleşmesidir | BPE/Unigram vocabulary training, normalization ve offset/sequence semantics gerekir |
| Tensor/matrix runtime | Yok | Transformer linear layers, embeddings, attention ve activations packed-HV XOR/majority işlemleri değildir |
| Autodiff + optimizer | Yok | Backpropagation, gradient accumulation, AdamW/Lion, clipping, loss scaling ve optimizer state gerekir |
| Transformer/sequence objective | Yok | Causal/encoder-decoder attention, positional encoding, logits, cross entropy ve sampling/generation gerekir |
| Multimodal encoders/fusion | Yok | image/audio/video decoding, vision/audio encoder, alignment/fusion, modality batching ve safety policy gerekir |
| Distributed training | Yok | data/model/pipeline parallelism, collectives, gradient synchronization, elastic fault recovery ve sharded optimizer/checkpoint gerekir |
| Neural mixed precision | Yok | BF16/FP16/FP8 numerical policy, master weights, loss scaling ve hardware kernels gerekir |
| LMM evaluation gate | Yok | held-out perplexity/task suites, contamination policy, modality quality, safety/red-team ve calibration gerekir |
| GPU/NPU neural training backend | Yok | adapter inventory/OpenVINO context dispatch, neural training kernel veya CPU+GPU+NPU joint compute kanıtı değildir |
| Dataset governance at LMM scale | Kısmi | exact duplicate/leakage gate vardır; semantic near-duplicate, licensing/PII/consent, benchmark contamination ve multimodal rights pipeline ayrıca gerekir |

## Bu değişiklikte bilinçli olarak eklenenler

`production_dataset.py` ve `production_pretrain.py`, yukarıdaki neural runtime
boşluklarını gizlemeye çalışmaz. HDC yolunda gerçekten uygulanabilir, test
edilebilir işlemsel eksikleri kapatır:

1. local strict corpus için snapshot-temelli atomic dataset publication;
2. RAM'e sığmayan corpus'lar için SQLite-backed exact record dedupe;
3. raw-record digest'ten stable train/validation assignment ve exact leakage
   rejection;
4. shard payload/ledger/data manifest/binary/policy SHA identity chain;
5. `min_frequency=1` continual policy ile exact train unique vocabulary count
   ve `--max-vocab` fail-closed preflight;
6. controller'a immutable manifest digest pin'i; child training sırasında
   ledger replacement'ın kabul edilmemesi;
7. existing candidate-checkpoint/atomic-state exactly-once semantics'in gerçek
   compiled CLI ile CTest integration doğrulaması;
8. operator TSV retrieval için ayrı frozen held-out exact-grounding/abstention
   evaluation, binary/checkpoint/KB digest binding ve release gate.

Bunlar veri kaybı, silent duplicate, held-out contamination, capacity overflow,
restart operasyon riski ve dar grounded-retrieval kalitesinin ölçülmeden
publish edilmesi riskini azaltır. Semantic language quality, generative ability
veya multimodal competence kanıtlamaz.

## Gerçek LMM hedefi için dürüst ayrı proje sınırı

Bir LMM/LLM hedefi kabul edilirse uygun iş, `hdc_pretraining.f90` içine birkaç
option eklemek değildir. Ayrı bir architecture decision record ve en az aşağıdaki
kanıtlar gerekir:

1. model family/objective/parameterization ve numerical reproducibility
   contract'i;
2. Unicode/subword + modality-specific input data contract'i;
3. tensor/autodiff/optimizer/checkpoint ABI'si;
4. single-device correctness oracle ve gradient/numerical regression tests;
5. multi-device collective/failure/restart semantics;
6. target Intel CPU/iGPU/GPU/NPU veya seçilen accelerator üzerinde gerçek kernel
   execution mask, correctness ve measured throughput evidence;
7. licensed/consented dataset registry, semantic dedupe/contamination policy ve
   held-out evaluation protocol;
8. memory/SSD/network/resource scheduler ve experiment provenance/retention
   planı;
9. safety, privacy, misuse ve deployment threat model'i.

Bu koşullar olmadan "devasa LMM production training" iddiası teknik olarak
savunulabilir değildir. Mevcut Vegai roadmap'i buna alternatif olarak
**deterministik, scalable HDC production operation** sunar.

İşlemsel ayrıntılar için [PRODUCTION_PRETRAINING.md](PRODUCTION_PRETRAINING.md),
HDC data/checkpoint sözleşmesi için [PRETRAINING.md](PRETRAINING.md), continual
transaction için [CONTINUOUS_PRETRAINING.md](CONTINUOUS_PRETRAINING.md),
held-out retrieval gate/audit için [QUALITY_ASSURANCE.md](QUALITY_ASSURANCE.md)
ve Fortran core invariantları için [ARCHITECTURE.md](ARCHITECTURE.md) referanstır.
