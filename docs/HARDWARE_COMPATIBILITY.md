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
> `hybrid` semantiğini korur. Opt-in `coherent` aynı long-record bütçesinde
> cross-fold global evidence kullanır; current OpenVINO worker ABI'si bunu
> taşımadığından CPU/OpenMP-only'dir.

## 1. Uyumluluk kapsamı ve doğrulama katmanları

"Bütün işletim sistemleri ve bütün donanımlar" için peşinen mutlak çalışma veya
aynı performans garantisi verilemez: compiler sürümü, libc/runtime, dosya sistemi,
driver, CPU ISA/endianness ve accelerator ABI'si bunların her birini etkiler.
Vegai bunu ölçülebilir katmanlara ayırır:

| Katman | Sözleşme | Kanıt / davranış |
|---|---|---|
| Portable CPU kaynak yolu | Standard Fortran 2023 + host-ISA bayrağı olmayan CMake profili | Aynı hedef OS/CPU üzerinde yeniden derlenir; zorunlu GPU/NPU runtime'ı yoktur. |
| Serial fallback | OpenMP bulunmadan da aynı CPU semantiği | `portable-serial` preset'i `VEGAI_ENABLE_OPENMP=OFF` ile derler. |
| OpenMP CPU yolu | Runtime bulunursa immutable context aşamasını paralelleştirir | Sıralı commit korunur; throughput hosta özeldir. |
| Optional accelerator | OpenVINO provider yalnız açıkça seçilirse kullanılır | Driver/device/bit-exact contract yoksa CPU fallback veya explicit failure; adapter listesi hızlandırma iddiası değildir. |
| Checkpoint taşınması | ABI-local stream-unformatted dosya | OS/CPU/compiler geçişinde checkpoint'i taşınabilir saymaz; hedefte yeniden train veya aynı ABI gerekir. |

`CMakePresets.json` üç açık profil sağlar: `portable`, `portable-serial` ve
host-only `native`. `.github/workflows/portable-platforms.yml`, source/launcher
sözleşmesini Linux x86-64, macOS ARM64 ve Windows x86-64 üzerinde serial
portable profile ile çalıştıracak şekilde tanımlanmıştır. ARM64 source/runtime
sözleşmesi macOS ARM64 worker üzerinde ölçülür; Linux ARM64 ise aynı portable
preset'i hedefte çalıştırılması gereken bir build adayıdır. Bu workflow her
platformda gerçek `hds_engine` çalıştırır, küçük bir corpus ile `vegai_pretrain
train` çağrısı yapar ve oluşan checkpoint'i **aynı binary** ile inspect eder.
Bu, compiler/OS/launcher/path uyumluluğu kanıtıdır; platformlar arası checkpoint
interchange veya bir CPU'nun diğerinden hızlı olduğu iddiası değildir.

Bilinmeyen veya CI'da henüz ölçülmemiş BSD, RISC-V, POWER, s390x ya da özel
embedded hedeflerde kaynak portable-build *adayı*dır; hedefte Fortran 2023
compiler + CMake ile configure/build/smoke sonucu alınmadan desteklenmiş olarak
sunulmaz. `hardware_compatibility.py` architecture family, pointer width ve byte
order'ı raporlar; `unknown` bir envanter sonucu derleme başarısızlığı değildir,
ama doğrulanmış-platform rozeti de değildir.

## 2. Native Fortran profile ve resource preflight

`vegai_control` standart Fortran 2023 control-plane executable'ıdır. Python,
`/proc`, PowerShell, `sysctl`, WMIC veya başka OS shell inventory komutu çağırmaz.
Bu nedenle portable olan compiler/environment gerçeklerini raporlar; RAM ve
free-storage değerleri gerekiyorsa operatorün denetlenmiş input'u olarak kabul
eder.

```bash
./build/portable-serial/vegai_control profile --format json
./build/portable-serial/vegai_control resource-plan CORPUS.txt \
  --checkpoint checkpoints/model.vgai --max-vocab 50000 \
  --available-memory-bytes 17179869184 \
  --available-storage-bytes 34359738368 --require-ready --format json
./build/portable-serial/vegai_control smoke CORPUS.txt checkpoints/smoke.vgai
```

`resource-plan`, model/vocabulary/context/update/cache reservationını ve
checkpoint üst sınırını `int64` aritmetiğiyle sabit formülden hesaplar.
`--require-ready` ile RAM veya SSD değeri verilmemişse ya da hesaplanan sınırın
altındaysa non-zero döner. Bu, allocator/page-cache/NUMA/GPU residency tahmini
değildir; gözlenmemiş bir makine özelliğini başarı olarak raporlamaz.

Python `hardware_compatibility.py` ve `training_resource_plan.py` daha zengin
OS/vendor inventory için geçici legacy adapter'lardır. Default Fortran runtime
bunları çağırmaz. Fortran'a taşınma envanteri ve `ISO_C_BINDING` gerektiren
SDK sınırları [FORTRAN_2023_MIGRATION.md](FORTRAN_2023_MIGRATION.md)'dedir.

## 3. CPU profilleri

### `portable` — varsayılan ve dağıtılabilir profil

CMake'in varsayılanı ve `make all` artık host-ISA bayrağı kullanmaz. Böylece
`-march=native` kaynaklı, başka bir CPU'da illegal-instruction riski taşıyan
binary üretimi varsayılan davranış olmaktan çıkar.

```bash
# GNU Make convenience yolu: GNU Fortran + POSIX shell gerekir.
make clean-all
make all                         # ./build/vegai_pretrain, portable profile

# CMake preset: Linux, macOS ve Windows için önerilen yol.
cmake --preset portable
cmake --build --preset portable --parallel
ctest --preset portable

# OpenMP runtime istemeyen en geniş fallback profile:
cmake --preset portable-serial
cmake --build --preset portable-serial --parallel
ctest --preset portable-serial

# Preset kullanmayan eşdeğer CMake çağrısı:
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

## 4. Intel Core Ultra 5 125H

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

## 5. Entegre grafik, discrete GPU ve NPU

Portable varsayılan Vegai binary'si zorunlu bir GPU/NPU compute backend'i
veya vendor runtime yüklemez: CUDA, OpenCL, SYCL/oneAPI, Metal ve Vulkan
compute bağımlılığı yoktur. Intel OpenVINO kurulursa yalnız immutable HDC
context stage'i için opt-in bir worker vardır; source-ordered training commit
CPU'da kalır. Sonuç olarak:

- Intel Arc/iGPU, AMD Radeon iGPU, Apple integrated GPU veya discrete GPU'nun
  varlığı CPU execution için önkoşul değildir;
- ekran sürücüsünün vendor API desteği yoksa bile portable CPU backend çalışır;
- GPU ve NPU inventory tek başına GPU/NPU hızlandırması iddiası değildir;
- generic `--require-accelerator` resource plan'da fail-closed davranır;
  adapter tespitinin compute dispatch'e dönüştüğünü varsaymaz;
- Intel Core Ultra için opt-in OpenVINO context provider immutable packed-HDC
  bundle/finalize aşamasını CPU/GPU/NPU policy'sine dispatch edebilir; runtime
  summary gerçek execution mask'ini raporlar ve `--openvino-require-accelerator`
  CPU fallback'i reddeder. Explicit fallback `--openvino-device CPU` ise CPU
  plugin'i başlatmak yerine native fused/OpenMP kernel'ini kullanır;
- `coherent`in center-dependent cross-fold item dizisi lane ABI'de yoktur;
  `--context-provider openvino --context-mode coherent` validation aşamasında
  fail-closed reddedilir, başka bir mode'a sessizce yönlendirilmez;
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

## 6. İşletim sistemi ve compiler yolu

### Linux

GNU Fortran + CMake/Ninja ile portable profile CI'da gerçek olarak doğrulanır.
OpenMP bulunamazsa derleme başarısız olmaz; cleanup scan ve context construction
doğru fakat serial kalır.

### macOS

Homebrew GNU Fortran veya Intel Fortran ile CMake portable profile kullanın.
GCC formula'sının major-version suffix'i hosta göre değiştiği için uygun
`gfortran-N` executable'ını seçin:

```bash
FC=gfortran-N cmake --preset portable
cmake --build --preset portable --parallel
ctest --preset portable

# OpenMP runtime olmadan aynı CPU semantiğini smoke-test etmek için:
FC=gfortran-N cmake --preset portable-serial
cmake --build --preset portable-serial --parallel
```

Apple Silicon için ayrıca ARM64 toolchain üzerinde yeniden derleme gerekir;
x86-64 macOS binary'si universal binary değildir.

### Windows

Windows'ta GNU Make yerine CMake önerilir. GNU Fortran (örneğin MSYS2 UCRT64
`mingw-w64-ucrt-x86_64-gcc-fortran`) veya Intel ifx ile uygun compiler'ı CMake'e
verin. En geniş runtime fallback için OpenMP'siz portable preset'i seçin:

```powershell
cmake --preset portable-serial
cmake --build --preset portable-serial --parallel
.\build\portable-serial\vegai_pretrain.exe help
python tools\verify_portable_runtime.py `
  --engine-bin .\build\portable-serial\hds_engine.exe `
  --pretrain-bin .\build\portable-serial\vegai_pretrain.exe
```

MSYS2 shell içindeyseniz aynı yollar slash biçiminde kullanılabilir. Windows
runner smoke check'i checkpoint'i aynı `.exe` ile train+inspect eder; `.vgai`
dosyasını macOS/Linux binary'sine kopyalayıp okumayı test veya sözleşme olarak
sunmaz.

Compiler'ın Fortran 2023 conformance seçeneği desteklenmiyorsa CMake bunu
uyarı olarak gösterir; platforma uygun current compiler veya explicit compiler
flag kullanılmalıdır.

### Aynı-host portable runtime gate

CMake ile üretilen binary için platformdan bağımsız launch/train/inspect gate:

```bash
python3 tools/verify_portable_runtime.py \
  --engine-bin ./build/portable-serial/hds_engine \
  --pretrain-bin ./build/portable-serial/vegai_pretrain
```

Windows'ta executable suffix'i `.exe` olur. Bu gate command-line path, temporary
checkpoint creation ve aynı binary ile inspect akışını doğrular; benchmark
zamanı kaydetmez ve compiler/OS değişince checkpoint'in taşınacağını iddia
etmez.

## 7. Checkpoint sınırı

Source/build portability, checkpoint binary interchange garantisi değildir.
`.vgai` dosyası stream-unformatted formatta compiler/ABI/endianness ailesine
bağlıdır. Başka OS, CPU mimarisi veya compiler'a geçerken corpus manifesti,
source commit, compiler bilgisi ve config'i saklayın; hedef hostta yeniden train
etmek veya aynı ABI toolchain'i kullanmak güvenli yoldur.

## 8. Gerçek ölçüm kaydı

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
