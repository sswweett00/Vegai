# Donanım ve işletim sistemi uyumluluğu

Vegai'nin çalıştırma sözleşmesi, vendor'a özgü GPU kernel'i değil **portable
Fortran 2023 CPU backend**'idir. Bu tercih Intel Core Ultra 5 125H gibi mobil
x86-64 sistemlerde, daha eski uyumlu x86-64 CPU'larda ve uygun Fortran compiler
ile ARM64 gibi başka mimarilerde aynı kaynak kodunun derlenebilmesini hedefler.

> “Uyumlu” burada portable profile ile derlenebilme ve CPU üzerinde doğru
> çalışabilme anlamındadır. Her CPU/OS/compiler kombinasyonunda aynı throughput,
> aynı SIMD genişliği veya aynı checkpoint dosya ABI'si vaat edilmez. Benchmark
> sonucu ancak ölçüldüğü host için geçerlidir.

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

Pretraining'in hot path'i şu an deterministic packed-bit/SIMD kodudur. OpenMP
cleanup scan'leri için isteğe bağlıdır; `OMP_NUM_THREADS` değerini artırmak,
pretraining corpus loop'unu otomatik olarak core sayısı kadar bölmez. Bu nedenle
125H'nin core/thread sayısından doğrusal hızlanma varsayılmamalıdır.

## 3. Entegre grafik, discrete GPU ve NPU

Vegai şu an GPU compute backend'i içermez. CUDA, OpenCL, SYCL/oneAPI, Metal,
Vulkan compute veya vendor NPU runtime'ı yüklenmez. Sonuç olarak:

- Intel Arc/iGPU, AMD Radeon iGPU, Apple integrated GPU veya discrete GPU'nun
  varlığı CPU execution için önkoşul değildir;
- ekran sürücüsünün vendor API desteği yoksa bile portable CPU backend çalışır;
- GPU inventory yalnız benchmark provenance amacıyla kaydedilir, GPU
  hızlandırması iddiası değildir;
- ileride GPU backend eklenirse ayrı numerical determinism, driver ve fallback
  sözleşmesi gerekir; mevcut CPU checkpoint equality garantisi otomatik olarak
  GPU backend'e taşınmış sayılmaz.

Host bilgisini güvenli/taşınabilir bir JSON olarak kaydetmek için:

```bash
python3 tools/hardware_compatibility.py --format json \
  --require-recognized-os --output hardware-profile.json
```

Araç Linux, macOS ve Windows için best-effort CPU/GPU adapter inventory üretir;
hostname, MAC, serial number veya tam environment dump kaydetmez.

## 4. İşletim sistemi ve compiler yolu

### Linux

GNU Fortran + CMake/Ninja ile portable profile CI'da gerçek olarak doğrulanır.
OpenMP bulunamazsa derleme başarısız olmaz; cleanup scan doğru ama serial kalır.

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
hariçtir. Son doğrulanmış portable/native gerçek CI ölçümü
[PRETRAINING_MODE_COMPARISON.md](PRETRAINING_MODE_COMPARISON.md) içinde
kaydedilir.
