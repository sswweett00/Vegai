# RAM, SSD ve çok çekirdek için kaynak-duyarlı HDC pretraining

Bu belge Vegai'nin gerçek eğitim backend'lerini açıklar. Varsayılan portable
Fortran/OpenMP yolu CPU üzerinde deterministic HDC context learning çalıştırır.
GPU/NPU adapter'larının görünmesi veya bir iGPU'nun mevcut olması tek başına
GPU/NPU compute kullanıldığı anlamına gelmez. Intel Core Ultra için explicit
OpenVINO context-provider seçilmediğinde `CPU compute used=true`, `GPU compute
used=false`, `NPU compute used=false` durumları geçerlidir; opt-in provider
çalıştığında gerçek worker execution mask'i ayrı runtime summary'de raporlanır.

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

Dolayısıyla varsayılan portable CPU sürümü, adapter bulunduğunda GPU/NPU'yu
gizlice kullanmaz. Generic `tools/training_resource_plan.py --require-
accelerator` provider/device execution'ını görmediği için hâlâ **fail closed**
döner. Intel Core Ultra için explicit `--context-provider openvino` yolu,
yalnız immutable context stage'ini gerçek OpenVINO worker'a dispatch edebilir;
runtime output ve `--openvino-require-accelerator` gerçek GPU/NPU execution'ı
ayrıca doğrular. NVIDIA/AMD gibi başka provider'lar için ayrı, eşdeğer contract
gereklidir. Provider şu sözleşmeleri sağlamadan "kullanılıyor" diye
belgelenmemelidir:

- provider/driver/runtime sürümü ve supported OS/architecture manifesti;
- HDC bit arithmetic, tie seed, overflow ve checkpoint byte-equivalence veya
  açık, versioned numeric-difference policy;
- CPU fallback, provider yokken nonzero/explicit warning davranışı;
- CPU-only ve accelerator run'ları için aynı controlled quality protocol;
- transfer/dispatch/harness dışlanmış gerçek train timing ve energy/thermal
  ölçümü.

Bu sınır, iGPU/NPU bulunan sistemin CPU fallback'ten mahrum kalmamasını ve
olmayan bir multi-accelerator hızlandırmayı yanlışlıkla iddia etmemeyi sağlar.

## 5. Intel Core Ultra için optional OpenVINO/oneAPI context provider

Intel Core Ultra seçeneği için repository, portable CPU binary'sine zorunlu
runtime eklemeden gerçek bir optional OpenVINO context provider içerir. HDC
akışındaki mutable olmayan bölüm (positioned base-vector item'larının
bit-sliced bundle/finalize işlemi) OpenVINO worker'a batch olarak gönderilir;
source-order prototype/Hebbian commit, experience schedule ve checkpoint yazımı
Fortran CPU tarafında kalır. Böylece GPU/NPU yalnız uygun immutable context
işinde kullanılır; semantiği sıralı olan update aşaması zorla offload edilmez.

Worker yalnız adapter listesi değildir; selected device üzerinde iki gerçek HDC
shape'ı çalıştırır. Default `window=4` için full hybrid 16 item, hierarchical
record-summary ile 19 item test edilir. CPU, GPU veya NPU worker çıktısı
Fortran'ın `hv_bitacc_finalize` sonucuyla byte-eşit olmak zorundadır; aksi halde
provider başarısız sayılır.

```bash
# Intel accelerator denemesi yapılacak makinede:
python3 -m pip install -r requirements-openvino.txt
mkdir -p .vegai-openvino-work

# CPU/OpenMP fallback açık: kısa records (<64 eligible center) CPU'da kalır;
# worker/runtime/device hatası da açıkça runtime summary'de fallback olarak görünür.
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./build/vegai_pretrain train CORPUS.txt checkpoints/model.vgai \
  --context-provider openvino \
  --openvino-worker tools/openvino_context_worker.py \
  --openvino-python python3 --openvino-device 'AUTO:GPU,NPU,CPU' \
  --openvino-work-dir .vegai-openvino-work --openvino-min-contexts 64 \
  --openvino-fallback-cpu

# Production/benchmark gate: worker, exact graph ve gerçek GPU/NPU execution
# zorunludur; CPU fallback nonzero ile reddedilir.
./build/vegai_pretrain train CORPUS.txt checkpoints/model.vgai \
  --context-provider openvino \
  --openvino-worker tools/openvino_context_worker.py \
  --openvino-python python3 --openvino-device 'AUTO:GPU,NPU,CPU' \
  --openvino-work-dir .vegai-openvino-work --openvino-min-contexts 64 \
  --openvino-required --openvino-require-accelerator
```

`--openvino-work-dir` mevcut, yazılabilir ve aynı anda yalnız bir train process
ile kullanılan bir dizin olmalıdır. Worker request/response stream dosyalarını
atomik response replace ile kullanır; bunlar train sonunda silinir. `--openvino-
min-contexts` transfer/process overhead'ini küçük record'larda önler. Bu
threshold altında CPU/OpenMP context build kasıtlıdır; runtime summary şu
alanları ayırır: CPU-built contexts, OpenVINO-dispatched contexts, worker
invocation, fallback contexts ve gerçekten execute eden CPU/GPU/NPU device
flag'leri. Bu alanlar checkpoint'e yazılmaz.

### Exact tensor ABI ve fail-closed preflight

```bash
python3 tools/openvino_provider.py --device auto --format json \
  --output openvino-provider.json

# Hedefte hem GPU hem NPU envanteri, exact HDC graph ve gerçek accelerator
# execution gerekiyorsa:
python3 tools/openvino_provider.py --device 'AUTO:GPU,NPU,CPU' \
  --require-openvino --require-gpu --require-npu \
  --require-hdc-probe --require-hdc-accelerator
```

Doğrudan `uint64` bitwise ops, bazı OpenVINO plugin/sürümlerinde 64 bitin tümünü
korumayabilir. Bu nedenle probe'daki mixed high/low-bit direct `uint64` sonucu
yalnız diagnostics amaçlıdır; başarısız olması tek başına HDC ABI'nin başarısız
olduğu anlamına gelmez. Provider her Fortran `int64` bit-pattern'i için explicit
olarak `lane[2*w]=low32(word[w])`, `lane[2*w+1]=high32(word[w])` üretir. Bu
`uint32` lane'lerde XOR/AND/OR/NOT ve bit-sliced counter her bit için bağımsız
çalıştığı için packed 16,384-dimensional semantiği korur. Worker protocol
little-endian Intel/x86 ABI'sidir; host byte reinterpretation/endian varsayımı
kullanmaz ve big-endian host'ta fail closed olur.

`AUTO`, `MULTI` ve `HETERO` OpenVINO device-policy terimleridir: tek başına bir
graph'ın CPU, GPU ve NPU üzerinde aynı anda yürüdüğünü kanıtlamaz. Vegai'nin
truth contract'i daha dar ve ölçülebilirdir: worker output header'ı gerçek
execution device mask'ini taşır; `--openvino-require-accelerator` GPU veya NPU
mask'i yoksa başarısız olur. Sıralı CPU commit + doğrulanmış accelerator context
kombinasyonu dışında joint training iddiası yapılmaz.

Provider execution-only'dir: aynı corpus/config için CPU referans ve OpenVINO
context run checkpoint byte equality ile test edilmelidir. CI'daki optional
OpenVINO bridge test bu eşitliği CPU OpenVINO plugin'iyle zorunlu kılar; gerçek
Core Ultra GPU/NPU throughput veya energy sonucu ancak hedef host'ta ayrıca
ölçülüp kaydedilmelidir. Bu repository şu anda target GPU/NPU üzerinde ölçülmüş
bir hızlanma rakamı iddia etmez.
