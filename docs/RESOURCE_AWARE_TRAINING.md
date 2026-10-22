# RAM, SSD ve çok çekirdek için kaynak-duyarlı HDC pretraining

Bu belge Vegai'nin mevcut **gerçek** eğitim backend'ini açıklar: portable
Fortran/OpenMP ile CPU üzerinde çalışan deterministic HDC context learning.
GPU/NPU adapter'larının görünmesi veya bir iGPU'nun mevcut olması, tek başına
GPU/NPU compute kullanıldığı anlamına gelmez. Şu an checked-in binary için
`CPU compute used=true`, `GPU compute used=false` ve `NPU compute used=false`
durumlarıdır.

> “Tam uyumluluk” donanım fiziksel topolojisinin yazılımla kontrol edilebileceği
> anlamına gelmez. OS; DIMM channel interleaving, NUMA page placement, SSD queue
> scheduling ve termal/power sınırlarını yönetir. Vegai, bunları zorlamaz;
> görünür kaynaklara göre bounded bir plan yapar, portable CPU fallback'i korur
> ve bilinmeyen/uygulanmayan accelerator'ı kullanılmış gibi raporlamaz.

## 1. Eğitimden önce RAM / storage planı üretin

```bash
python3 tools/training_resource_plan.py \
  --corpus /data/corpus.txt \
  --checkpoint /fast-ssd/checkpoints/model.vgai \
  --max-vocab 50000 --format json --output run-resource-plan.json \
  --require-ready
```

Araç corpus'u RAM'e yüklemez; sadece corpus path'ini ve byte boyutunu doğrular.
Aşağıdaki hesapları raporlar:

- iki 16.384-bit HDC plane + `int64` experience counter için slot başına kesin
  packed model payload'u: `4,104 byte`;
- vocabulary/hash ABI allowance, 2 MiB bounded OpenMP context staging ve
  varsayılan `%25` headroom ile conservative available-RAM reservation;
- checkpoint için conservative disk boyutu ve seçilen checkpoint path'inin
  filesystem free-space değeri;
- CPU physical-core bilgisi bulunabiliyorsa onu tercih eden `OMP_NUM_THREADS`,
  `OMP_PROC_BIND=close`, `OMP_PLACES=cores` önerisi;
- NUMA, RAM-channel, GPU ve NPU envanteri için best-effort provenance.

Bu bir allocation garantisi değildir: compiler'ın `logical` boyutu, OpenMP
runtime, allocator alignment, page cache, başka süreçler ve gerçek vocabulary
sayısı resident set'i değiştirebilir. Ancak OOM'a sessizce yaklaşmak yerine
train öncesi bir alt sınır verir. `--require-ready`, hesaplanan RAM veya
checkpoint storage yetersizse train'i başlatmadan nonzero döner.

### Dual-channel / multi-channel RAM

Eş DIMM'ler ve doğru slotlar firmware/anakart tarafından dual veya multi-channel
olarak yapılandırılır. Vegai normal heap allocation kullandığı için OS'nin
interleaved memory policy'sinden yararlanır; portable kullanıcı-space kodu
RAM channel'ını güvenilir biçimde açıp kapatamaz. Privileged DMI bilgisi yoksa
rapor `not-detected-without-privileged-firmware-inventory` döndürür. Bu bir
hata ya da single-channel tespiti değildir.

### SSD ve checkpoint yolu

Corpus satır satır stream edilir; corpus byte boyutu resident model estimate'ine
eklenmez. Vegai normal dosya I/O kullandığından SSD, HDD, network volume veya
removable filesystem üzerinde çalışabilir. SSD sadece I/O gecikmesi/throughput
bakımından yararlı olabilir; tool medium type veya belirli hız iddiası yapmaz.

Crash-safe continual çalışma için `tools/continuous_pretrain.py`, private shard
snapshot, candidate checkpoint, `fsync` ve atomik state-journal replace akışını
kullanır. Candidate/generation paths aynı güvenilir filesystem altında
olmalıdır; network/distributed filesystem atomicity semantiğini deployment
öncesi ayrıca doğrulayın.

## 2. Deterministic çok çekirdekli eğitim

`hdc_pretraining` artık her input record için iki aşama kullanır:

1. immutable `base_bits` üzerinden tüm center context'leri bağımsız biçimde
   hazırlanır; OpenMP varsa bu aşama `parallel do` ile çok core kullanabilir;
2. prototype ve experience updates corpus kaynak sırasıyla commit edilir.

İkinci aşama bilinçli olarak sıralıdır. Aynı token bir record içinde tekrar
ediyorsa experience-count/tie seed sırası semantiktir. Planner, her center'ın
beklenen counter değerini önceden belirler ve commit'te doğrular; hatalı
scheduler sonucu sessiz semantic drift yerine hata verir. Bu nedenle
`OMP_NUM_THREADS=1` ve `OMP_NUM_THREADS=N` checkpoint byte'ları aynı olmalıdır.
`--serial-context-build`, yalnız debug/regression amacıyla parallel hazırlığı
kapatır; checkpoint formatı veya öğrenme semantiği değildir.

Önerilen başlangıç:

```bash
# Planner'ın çıktısından alınan değerleri önce aynı makinede benchmark edin.
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

./build/vegai_pretrain train /data/corpus.txt /fast-ssd/checkpoints/model.vgai \
  --max-vocab 50000 --min-frequency 3 --window 4 --position-shift 73 \
  --context-mode hierarchical --epochs 3 --strict --parallel-context-build
```

Küçük satırlar, düşük context window, disk-bound corpus veya hyperthread-only
thread sayıları hızlanmayabilir. Aynı machine üzerinde gerçek child-process
ölçümü ve byte equality için:

```bash
python3 tools/compare_pretraining_threads.py \
  --pretrain-bin ./build/vegai_pretrain --parallel-threads 2 --repetitions 3
```

Timing yalnız `vegai_pretrain train` child process'ini kapsar; compile, fixture
üretimi, checkpoint compare ve harness dışarıdadır. Hız oranı host/workload'a
özgüdür; core sayısıyla lineer ölçeklenme vaadi değildir.

İlk doğrulanmış gerçek CI ölçümünde ([run 39839135866](https://github.com/sswweett00/Vegai/actions/runs/39839135866),
AMD EPYC 7763 / Linux) long-record fixture için 1 thread `0.538901 s`, 2 thread
`0.465863 s` verdi: **1.1568×** hız oranı ve checkpoint byte equality `True`.
Bu yalnız iki-thread, o runner ve o workload için bir ölçümdür; gerçek corpus ve
Core Ultra 5 125H gibi hedef hardware üzerinde tekrar ölçülmelidir.

## 3. Semantic kalite için varsayılan

Yeni `train` varsayılanı `hierarchical` context mode'dur. Bu mod `hybrid`in
local positional + topical context'ini korur; yalnız bir record'da en az 12
active token varsa deterministic whole-record topical summary'yi bounded üç
katkı olarak ekler. Kısa records için sonuç `hybrid` ile bit-tam aynıdır.

Repository controlled long-record diagnostic'inde `hierarchical`, local
`hybrid` mean Top-1 similarity'sini `0.168998`den `0.826843`e çıkardı
(`+0.657845`). Bu yalnız tasarlanmış disjoint topical, 16-token in-corpus
fixture için bir regression/quality signalidir; doğal dil zekâsı, held-out task
accuracy veya gerçek corpus kalite garantisi değildir. Eski experiment'i veya
kısa-record baseline'ı yeniden üretmek için açıkça `--context-mode hybrid`
verin.

Kaliteyi artırmanın en güvenli yolları şunlardır:

1. lisanslı corpus'u normalize edin, deduplicate edin ve tokenizer contract'ına
   uygun ASCII-safe token stream üretin;
2. train/dev/evaluation kaynaklarını ayırın; test kayıtlarının training'e
   sızmadığını manifest/digest ile denetleyin;
3. gerçek corpus için held-out retrieval veya task-özel metric tanımlayın;
4. `--max-vocab`, `--min-frequency`, window, epoch ve plasticity schedule'ını
   aynı held-out protokolde değiştirin; yalnız training time ile kalite iddiası
   yapmayın;
5. provenance'a corpus digest, CLI config, source commit, resource plan,
   OpenMP environment, compiler ve hardware profile ekleyin.

## 4. GPU + NPU + CPU hakkında dürüst sınır

HDC hot path'i packed bitwise context assembly ve source-ordered Hebbian commit
olduğu için klasik dense neural graph değildir. Her GPU/NPU offload'u faydalı
olmaz; host-device transfer, driver/runtime, integer bit semantics ve
checkpoint determinism ayrı doğrulanmalıdır.

Dolayısıyla mevcut sürüm, adapter bulunduğunda GPU/NPU'yu gizlice kullanmaz.
`tools/training_resource_plan.py --require-accelerator` bu nedenle **fail
closed** döner. Gerçek joint CPU/GPU/NPU eğitim eklemek için hedef platform ve
provider açıkça seçilmelidir; örneğin Intel Core Ultra için OpenVINO/oneAPI,
NVIDIA için CUDA veya AMD için ROCm. Provider şu sözleşmeleri sağlamadan
"kullanılıyor" diye belgelenmemelidir:

- provider/driver/runtime sürümü ve supported OS/architecture manifesti;
- HDC bit arithmetic, tie seed, overflow ve checkpoint byte-equivalence veya
  açık, versioned numeric-difference policy;
- CPU fallback, provider yokken nonzero/explicit warning davranışı;
- CPU-only ve accelerator run'ları için aynı controlled quality protocol;
- transfer/dispatch/harness dışlanmış gerçek train timing ve energy/thermal
  ölçümü.

Bu sınır, iGPU/NPU bulunan sistemin CPU fallback'ten mahrum kalmamasını ve
olmayan bir multi-accelerator hızlandırmayı yanlışlıkla iddia etmemeyi sağlar.

## 5. Intel Core Ultra için optional OpenVINO/oneAPI readiness

Intel Core Ultra seçeneği için repository, portable CPU binary'sine zorunlu
runtime eklemeden bir OpenVINO provider probe'u içerir. Probe yalnız adapter
listesi değildir; seçilen device üzerinde iki gerçek graph çalıştırır:

1. mixed high/low-bit girdili direct `uint64 XOR` diagnostics; ve
2. `hv_bitacc_*` ile aynı carry-plane/majority mantığını kullanan `uint32`
   lane HDC bundle/finalize graph'ları. Default `window=4` için full hybrid
   16 item, hierarchical ise record-summary ile 19 item test edilir.

```bash
# Yalnız Intel accelerator denemesi yapılacak makinede:
python3 -m pip install -r requirements-openvino.txt

# AUTO yalnız device tercihidir; tek graph'ın CPU+GPU+NPU üzerinde aynı anda
# çalıştığı iddiası değildir.
python3 tools/openvino_provider.py --device auto --format json \
  --output openvino-provider.json

# Hedefte GPU/NPU envanteri, exact HDC graph ve graph'ın CPU fallback yerine
# gerçekten bir accelerator üzerinde execute edilmesi gerekiyorsa:
python3 tools/openvino_provider.py --device 'AUTO:GPU,NPU,CPU' \
  --require-openvino --require-gpu --require-npu \
  --require-hdc-probe --require-hdc-accelerator
```

Doğrudan `uint64` bitwise ops, bazı OpenVINO plugin/sürümlerinde 64 bitin tümünü
korumayabilir. Bu nedenle probe'daki direct `uint64` sonucu yalnız diagnostics
amaçlıdır; başarısız olması tek başına HDC ABI'nin başarısız olduğu anlamına
gelmez. Aday ABI her Fortran `int64` bit-pattern'i için explicit olarak
`lane[2*w]=low32(word[w])`, `lane[2*w+1]=high32(word[w])` üretir. Bu `uint32`
lane'lerde XOR/AND/OR/NOT ve bit-sliced counter her bit için bağımsız çalıştığı
için packed 16,384-dimensional semantiği koruyabilir. Probe hem lane
round-trip'i hem de even (seeded tie) / odd majority context sonuçlarını CPU
oracle ile byte-eşit kontrol eder. Raw byte reinterpretation/endian varsayımı
sözleşmenin parçası **değildir**.

OpenVINO yalnız optional probe ile import edilir. Paket/device yoksa portable
Fortran trainer bozulmaz; `--require-*` seçeneği nonzero ile durur. `AUTO`,
`MULTI` ve `HETERO` OpenVINO device-policy terimleridir: bunlar tek başına bir
HDC train graph'ının CPU, GPU ve NPU arasında aynı anda doğru/yararlı biçimde
bölündüğünü kanıtlamaz.

Şu an exact lane probe başarılı olsa dahi `training_dispatch` alanı bilinçli
olarak `GPU=false`, `NPU=false`, `hdc_context_offload_integrated=false`
döndürür. Sebep, Fortran trainer'ın source-ordered prototype commit ve
checkpoint byte-equality sözleşmesini henüz OpenVINO tensor ABI'siyle
bağlamamış olmasıdır. Bu açık sınır, device göründüğü için yanlış bir
accelerator training iddiasını engeller. Gerçek Intel context-offload provider
ancak batched immutable context graph, transfer/memory ownership, fallback,
serial-vs-provider checkpoint equality ve real target-host timing testleri
tamamlandığında aktif sayılmalıdır.
