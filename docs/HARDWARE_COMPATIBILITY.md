# Donanım ve işletim sistemi uyumluluğu

Vegai'nin çalıştırma sözleşmesi, vendor'a özgü GPU kernel'i değil **portable
Fortran 2023 CPU backend**'idir. Bu tercih Intel Core Ultra 5 125H gibi mobil
x86-64 sistemlerde, daha eski uyumlu x86-64 CPU'larda ve uygun Fortran compiler
ile ARM64 gibi başka mimarilerde aynı kaynak kodunun derlenebilmesini hedefler.

> “Uyumlu” burada portable profile ile derlenebilme ve CPU üzerinde doğru
> çalışabilme anlamındadır. Her CPU/OS/compiler kombinasyonunda aynı throughput,
> aynı SIMD genişliği veya aynı checkpoint dosya ABI'si vaat edilmez. Benchmark
> sonucu ancak ölçüldüğü host için geçerlidir. Mevcut default training mode,
> uzun records için quality-oriented `hierarchical`tir; kısa records için local
> `hybrid` semantiğini korur.

## 1. CPU profilleri

### `portable` — varsayılan ve dağıtılabilir profil

CMake'in varsayılanı ve `make all` artık host-ISA bayrağı kullanmaz. Böylece
`-march=native` kaynaklı, başka bir CPU'da illegal-instruction riski taşıyan
binary üretimi varsayılan davranış olmaktan çıkar.

```bash
# GNU Make convenience yolu: GNU Fortran + POSIX shell gerekir.
make clean-all
make all                         # ./build/vegai_pretrain, portable profile

# CMake: Linux, macOS ve Windows için önerilen yol.
cmake -S . -B build-portable -DVEGAI_CPU_TARGET=portable \
  -DVEGAI_ENABLE_OPENMP=ON -DVEGAI_BUILD_TESTS=ON
cmake --build build-portable --parallel
ctest --test-dir build-portable --output-on-failure
```

Bu profil yalnız compiler'ın hedef mimarisi için güvenli baseline instruction
setini kullanır. Aynı x86-64 ABI ailesindeki farklı CPU'lar için uygun seçimdir;
ARM64 hedefi için binary'nin ARM64 host/toolchain üzerinde yeniden derlenmesi
gerekir.

### `native` — yalnız build host için opt-in profil

```bash
make native                      # ./build-native/vegai_pretrain
# veya CMake:
cmake -S . -B build-native -DVEGAI_CPU_TARGET=native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --parallel
```

GNU Fortran altında native profil `-march=native -mtune=native`, Intel
compiler altında eşdeğer host-tuning seçeneğini kullanır. Bu binary'yi başka
makineye dağıtmayın; hız avantajı varsa yalnız aynı/uyumlu CPU ailelerinde
kullanılmalıdır.

`tools/compare_pretraining_build_profiles.py`, portable ve native ikilileri
aynı gerçek train komutuyla karşılaştırır ve checkpoint byte'larının eşitliğini
zorunlu kılar:

```bash
python3 tools/compare_pretraining_build_profiles.py \
  --portable-pretrain-bin ./build/vegai_pretrain \
  --native-pretrain-bin ./build-native/vegai_pretrain \
  --repetitions 3 --omp-threads 1
```

## 2. Intel Core Ultra 5 125H

Ultra 5 125H için güvenli seçim portable profildir. Bu repo CPU backend'i
kullandığı için işlemciye özel AVX-512, CUDA, oneAPI GPU, NPU veya Arc API
zorunluluğu yoktur. Prize takılı bir sistemde local hız istenirse `make native`
ile ayrıca host-tuned ikili üretilebilir; portable/native benchmark sonucu
mutlaka aynı bilgisayarda ölçülmelidir.

Pretraining'in hot path'i deterministic packed-bit/SIMD context assembly'dir.
OpenMP artık bir record içindeki immutable center context'lerini paralel
hazırlayabilir; prototype/experience commit'i bit-tam determinism için kaynak
sırasında kalır. `OMP_NUM_THREADS` gerçek CPU multi-core kullanımını etkiler,
ama kısa records, memory bandwidth, I/O, power/thermal state ve scheduler
nedeniyle 125H'nin core/thread sayısıyla doğrusal hızlanma varsayılmamalıdır.
`tools/compare_pretraining_threads.py` ile aynı hostta byte equality + gerçek
child-process sürelerini ölçün.

## 3. Entegre grafik, discrete GPU ve NPU

Vegai şu an GPU compute backend'i içermez. CUDA, OpenCL, SYCL/oneAPI, Metal,
Vulkan compute veya vendor NPU runtime'ı yüklenmez. Sonuç olarak:

- Intel Arc/iGPU, AMD Radeon iGPU, Apple integrated GPU veya discrete GPU'nun
  varlığı CPU execution için önkoşul değildir;
- ekran sürücüsünün vendor API desteği yoksa bile portable CPU backend çalışır;
- GPU ve NPU inventory tek başına GPU/NPU hızlandırması iddiası değildir;
- generic `--require-accelerator` resource plan'da fail-closed davranır;
  adapter tespitinin compute dispatch'e dönüştüğünü varsaymaz;
- Intel Core Ultra için opt-in OpenVINO context provider immutable packed-HDC
  bundle/finalize aşamasını CPU/GPU/NPU policy'sine dispatch edebilir; runtime
  summary gerçek execution mask'ini raporlar ve `--openvino-require-accelerator`
  CPU fallback'i reddeder;
- provider için lane ABI, driver, transfer, fallback ve CPU-reference
  checkpoint-byte-equality sözleşmesi ayrı doğrulanır; adapter varlığı bu
  sözleşmenin yerine geçmez.

Host bilgisini güvenli/taşınabilir bir JSON olarak kaydetmek için:

```bash
python3 tools/hardware_compatibility.py --format json \
  --require-recognized-os --output hardware-profile.json
```

Araç Linux, macOS ve Windows için best-effort CPU topology, RAM availability,
RAM-channel discovery policy, GPU/NPU adapter inventory üretir; hostname, MAC,
serial number veya tam environment dump kaydetmez. RAM/SSD checkpoint path,
NUMA ve OpenMP placement için gerçek preflight ise:

```bash
python3 tools/training_resource_plan.py --corpus CORPUS.txt \
  --checkpoint checkpoints/model.vgai --max-vocab 50000 --require-ready
```

Ayrıntılı RAM/SSD/multi-core/quality/accelerator sınırları için
[RESOURCE_AWARE_TRAINING.md](RESOURCE_AWARE_TRAINING.md)'ye bakın. Intel Core
Ultra target'larında `tools/openvino_provider.py` preflight'i direct uint64
diagnostic ve exact uint32-lane HDC bundle probe'u sağlar; `tools/openvino_
context_worker.py` aynı lane ABI'sini actual immutable-context dispatch'te
kullanır. Train CLI runtime output'ı GPU/NPU execution'ı yalnız worker mask'i
bunu doğruladığında true raporlar.

## 4. İşletim sistemi ve compiler yolu

### Linux

GNU Fortran + CMake/Ninja ile portable profile CI'da gerçek olarak doğrulanır.
OpenMP bulunamazsa derleme başarısız olmaz; cleanup scan ve context construction
doğru fakat serial kalır.

### macOS

Homebrew GNU Fortran veya Intel Fortran ile CMake portable profile kullanın.
GCC formula'sının major-version suffix'i hosta göre değiştiği için uygun
`gfortran-N` executable'ını seçin:

```bash
FC=gfortran-N cmake -S . -B build-portable -DVEGAI_CPU_TARGET=portable
cmake --build build-portable --parallel
ctest --test-dir build-portable --output-on-failure
```

Apple Silicon için ayrıca ARM64 toolchain üzerinde yeniden derleme gerekir;
x86-64 macOS binary'si universal binary değildir.

### Windows

Windows'ta GNU Make yerine CMake önerilir. GNU Fortran (örneğin MSYS2/MinGW)
veya Intel ifx ile bir Developer/PowerShell terminalinde uygun compiler'ı
CMake'e verin:

```powershell
cmake -S . -B build-portable -DVEGAI_CPU_TARGET=portable `
  -DVEGAI_ENABLE_OPENMP=ON -DVEGAI_BUILD_TESTS=ON
cmake --build build-portable --config Release --parallel
ctest --test-dir build-portable -C Release --output-on-failure
.\build-portable\vegai_pretrain.exe help
```

Compiler'ın Fortran 2023 conformance seçeneği desteklenmiyorsa CMake bunu
uyarı olarak gösterir; platforma uygun current compiler veya explicit compiler
flag kullanılmalıdır.

## 5. Checkpoint sınırı

Source/build portability, checkpoint binary interchange garantisi değildir.
`.vgai` dosyası stream-unformatted formatta compiler/ABI/endianness ailesine
bağlıdır. Başka OS, CPU mimarisi veya compiler'a geçerken corpus manifesti,
source commit, compiler bilgisi ve config'i saklayın; hedef hostta yeniden train
etmek veya aynı ABI toolchain'i kullanmak güvenli yoldur.

## 6. Gerçek ölçüm kaydı

`tools/compare_pretraining_modes.py` ve build-profile comparison raporları
CPU modeli, OS, mimari, logical CPU sayısı ve GPU adapter sayısını JSON
provenance içine ekler. Timing kapsamı yine yalnız gerçek `vegai_pretrain train`
child process'idir; hardware discovery, compilation ve harness süreleri
hariçtir.

Son doğrulanmış portable/native gerçek ölçüm
[GitHub Actions run 39836218246](https://github.com/sswweett00/Vegai/actions/runs/39836218246)
üzerinde Intel(R) Xeon(R) 6973P-C / Linux x86_64 hostunda yapıldı: portable
`0.126613 s`, native `0.113506 s`, native/portable oranı `1.1155×` ve
checkpoint byte equality `True`. Bu bir host-specific CI sonucu olup Ultra 5
125H, Apple Silicon veya başka CPU'lar için doğrudan throughput vaadi değildir.
Ayrıntılı önce/sonra tablo için
[PRETRAINING_MODE_COMPARISON.md](PRETRAINING_MODE_COMPARISON.md)'ye bakın.

Yeni deterministic OpenMP context-build ölçümü
[run 39839135866](https://github.com/sswweett00/Vegai/actions/runs/39839135866)
üzerinde AMD EPYC 7763 / Linux hostunda 1 thread `0.538901 s`, 2 thread
`0.465863 s`, oran `1.1568×` ve checkpoint byte equality `True` verdi. Bu yalnız
long-record controlled fixture ve 2-thread armı için geçerlidir; herhangi bir
CPU, RAM channel, SSD veya iGPU/NPU throughput vaadi değildir.
