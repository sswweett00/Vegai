# Apex optimizer comparative study — current report

## Yönetici Özeti — 5 madde

1. Apex çekirdeğine dokunmadan Phase 0 için benchmark sözleşmesi, deterministik RNG/CSV, istenen task ailelerinin kaynak tanımları ve gradient-test iskeleti genişletildi.
2. `bench/ref` içinde 16 vector/diagonal variant ve factored Adafactor/Shampoo/Muon matrix varyantları mevcut; iki eksik optimizer ailesi eklendi, ancak compiler ile doğrulanmadı.
3. Faz 1 için on görev ayarı × beş seed’lik, 1.000-adım/60-saniye sınırlı fixed-config runner eklendi; real64 Apex/vector çalıştırması ve matrix factorization’daki matrix referanslarını dispatch edecek. Compiler yokluğu nedeniyle yürütülmedi, CSV/ölçüm üretmedi.
4. Faz 2 için Apex dahil deterministik 20-kandidat grid’i ve train/validation-split search runner eklendi; her aday eşit adım/duvar-saati bütçesiyle eğitilip validation loss’a göre işaretleniyor. Compiler yokluğu nedeniyle derlenmedi/çalıştırılmadı.
5. Bağımsız test partition’ı eklendi; validation ile seçilen aday, test skorlamasından sonra ayrı değerlendiriliyor. Phase 3 analiz programına CSV/selection denetimi, test-loss temelli eşleşmiş istatistikler ve Pareto çıktısı eklendi; ayrıca geçici fixture kullanan Fortran uçtan uca test yazıldı. Kaynaklar ayrıştırıldı ve build komutları dry-run edildi; compiler olmadığı için test derlenmedi/çalıştırılmadı. DoD tamamlanmadı ve üstünlük iddiası yok.

## Apex v2 Mimarisi

Apex'in mevcut public module/submodule ve eğitim entegrasyonu önceki değişikliklerden bu branch'te mevcuttu. Bu benchmark turunda `src/apex_optimizer.f90` ve `src/apex_optimizer_impl.f90` değiştirilmedi.

- `apex_config_t`: real64 runtime hyperparameter/scheduler yapılandırması ve enum scheduler.
- `apex_state_t`: private real64 parametre/momentum dizileri, update counter ve config.
- `apex_factors_t`: update başına ortak scalar katsayılar.
- `apex_error_t`: enum-backed durum ve açıklama.
- Public yüzey: config validation, initialize/update, callback/generic update, column/scalar kernel, state accessor, stream save/load ve Fortran namelist config okuyucu.
- Uygulama: submodule, IEEE finite kontrolleri ve `do concurrent`; neural training yolu tek momentum tensörü taşır.

| Bileşen/ablation | Önce-sonra gözlemi | Kanıt |
|---|---|---|
| Momentum / faz cevabı | Bu turda ablation yapılmadı | N/A |
| Gamma limiter | Bu turda ablation yapılmadı | N/A |
| Clipping / weight decay | Bu turda ablation yapılmadı | N/A |
| Scheduler | Bu turda ablation yapılmadı | N/A |
| Tek momentum state | Kaynak alanları mevcut; karşılaştırmalı bellek/kalite ölçümü yok | Kaynak incelemesi, benchmark yok |

Ablation sonucu veya bileşen kazancı iddia edilmemektedir.

## Benchmark Matrisi

Tüm istenen görev × rakip × hassasiyet × seed hücreleri için Δ, %95 GA ve p **N/M (not measured)** durumundadır. N/M, sıfır fark veya eşdeğerlik değildir.

| Metrik boyutu | Δ | %95 GA | p | Apex durumu |
|---|---:|---|---:|---|
| Hedef kayba adım / duvar saati | N/M | N/M | N/M | Bilinmiyor |
| Nihai kayıp / doğruluk | N/M | N/M | N/M | Bilinmiyor |
| State bytes/parameter | N/M karşılaştırma | N/M | N/M | Yalnız Apex payload hesabı mevcut |
| FLOPs / gerçek süre | N/M | N/M | N/M | Ölçülmedi |
| Train/validation genelleme açığı | N/M | N/M | N/M | Ölçülmedi |
| NaN/Inf, loss spike, seed varyansı | N/M | N/M | N/M | Benchmark failure oranı yok |
| LR/beta/epsilon sağlamlığı | N/M | N/M | N/M | Grid taraması yok |
| 10³→10⁷ ölçekleme | N/M | N/M | N/M | Ölçülmedi |
| Noise/sparse gradient | N/M | N/M | N/M | Ölçülmedi |
| Warm-start / transfer | N/M | N/M | N/M | Ölçülmedi |
| real32 / real64 | N/M | N/M | N/M | Ayrı deney yok |

Rakip implementasyonların listesi ve eksik matris `GAP_ANALYSIS.md` içindedir. Tam tablo üretecek veri bulunmamaktadır.

## Pareto Cephesi

Apex'in Pareto konumu **belirsizdir**. Hiçbir competitor gözlemi olmadığı için ne domine edildiği ne de Pareto-optimal kaldığı kanıtlanmıştır. Bu rapor Apex'in Pareto-optimal olduğunu ileri sürmez.

## Bellek ve Hız

- Apex optimizer state: bir real64 momentum / parametre = **8 B/parametre**, kaynak seviyesi payload hesabı.
- Apex state parametre + momentum: **16 B/parametre**, kaynak seviyesi payload hesabı.
- `bench/bench_apex_smoke.f90` CSV'sinde `state_bytes` parametre+momentum payload'ını raporlayacak; runtime allocator/header, RSS ve rakip state boyutları ölçülmüyor.
- FLOPs/adım, ms/adım ve throughput: **N/M**. 5N bandwidth veya diğer throughput tahmini bu raporda benchmark sonucu olarak kullanılmıyor.

## Sağlamlık

LR/beta/epsilon ısı haritası, seed mean±std, non-finite oranı ve sensitivity analysis yoktur. `bench/bench_apex_smoke.f90` yalnızca quadratic başlangıçlarının beş sabit seed ile kurulmasını ve Apex'in non-finite sonuç vermemesini kaydedecek şekilde yazıldı; bu program çalıştırılmamıştır. Bu nedenle sağlamlık sonucu çıkarılamaz.

## Doğrulama Kanıtları

- Benchmark komutları: `make bench-test`, `make bench-contract-test`, `make bench-task-test`, `make bench-tuning-test`, `make bench-statistics-test`, `make bench-phase3-test`, `make bench-ref-test`, `make bench-matrix-ref-test`, `make bench-vector-baseline`, `make bench-phase1-baseline`; Apex harness'i `make bench-apex-smoke`.
- Task/reference/matrix/vector/phase1-benchmark compile girişimleri ile `make bench-tuning-test`, `make bench-statistics-test`, Phase 3 analyzer build’i ve yeni `make bench-phase3-test` derlemesi `gfortran: No such file or directory` ile durdu. Test veya benchmark çalışmadı; ölçüm CSV'si yok.
- `make test`: bu ortamda `gfortran: No such file or directory` ile derleme aşamasında durdu; test çalıştırılmadı.
- Compiler/build probes: `gfortran`, `ifx`, `ifort`, `flang`, `flang-new`, `lfortran`, `fpm`, `cmake` yok.
- Host: x86_64; `/proc/cpuinfo` satırı `Intel(R) Xeon(R) Processor @ 2.60GHz` bildiriyor. Compiler sürümü ve `-O` flag'i yok/raporlanamaz.
- `make bench-contract-test`, `make bench-task-test`, `make bench-tuning-test`, `make bench-statistics-test`, `make bench-phase3-test`, `make bench-ref-test`, `make bench-matrix-ref-test`, `make bench-apex-smoke`, `make bench-vector-baseline` ve `make bench-phase1-baseline` compile recipes reached and each stopped with `gfortran: No such file or directory`; no benchmark test or CSV pilot ran.
- Ham CSV, compiler manifesti, rivals, bootstrap/Wilcoxon sonuçları mevcut değil. `bench/results/` içinde placeholder observation yok.
- `fparser` Fortran 2008 parser ile yeni benchmark modül, runner ve test kaynakları ayrıştırıldı; bu compiler doğrulaması değildir.

## Varsayımlar ve Sınırlar

- Öncelik: doğrulanmamış referans veya ölçüm uydurmamak. Birçok istenen yöntem tensor/matrix biçimi, Hessian kestirimi ya da ayrı training/eval semantiği gerektirir; yalnızca vector formülleriyle taklit edilmesi adil/literatüre sadık karşılaştırma değildir.
- `apex_config.toml` istenen declarative şablondur; mevcut Apex runtime loader Fortran namelist okur, TOML'u henüz parse etmez.
- Son kullanıcı isteği benchmark bağlamında bir dizi optimizer'ın adını açıkça içerdiğinden, bu isimler benchmark dokümanlarında rakip etiketi olarak geçebilir. Üretim Apex koduna rival dispatch eklenmedi.
- Faz 1 fixed-config runner ile Faz 2 tuning/istatistik kaynak yardımcıları eklendi; ancak Faz 1 çalıştırılmadığı ve Faz 2 arama runner’ına entegre olmadığı için faz çıkış kriterleri sağlanmadı. Derleyici doğrulaması ve empirical baseline kapısı açık; Apex algoritma tasarımı değiştirilmedi.

## Definition of Done

| Ölçüt | Durum | Kanıt / eksik |
|---|---|---|
| Her rakip için literatüre dayalı benchmark-only source | ⚠️ | 16 vector/diagonal methods + factored matrix Adafactor/Shampoo/Muon sources; source prototypes/tests present, compiler and independent golden verification absent |
| Tek komutla tam benchmark, 5 seed × bütün matris | ❌ | Yeni runner 10 task setting × 5 seed ve uyumlu kaynak yöntemlerini dispatch eder; tuning/real32/istatistik yok, Sophia sınırlı; compiler ile doğrulanmadı |
| Apex Pareto-domine değil | ❌ | Competitor sonuçları yok; Pareto hesaplanmadı |
| En az 6 boyutta istatistiksel üstünlük | ❌ | Veri ve test yok; iddia yok |
| Diğer boyutlarda kanıtlı eşitlik/Pareto | ❌ | Veri yok |
| Her Apex bileşeni için ablation | ❌ | Ablation yapılmadı |
| `-Wall -Wextra` uyarısız derleme | ❌ | Fortran compiler yok |
| Save/load bit-exact | ⚠️ | Önceden test kodu mevcut, ancak yürütülmedi |
| Tüm benchmarklarda NaN/Inf yok | ❌ | Benchmark matrisi çalışmadı |
| README örneği çalışan | ⚠️ | Örnek mevcut, compiler ile doğrulanmadı |
| Hardware/compiler/flags belgeli | ⚠️ | CPU kimliği biliniyor; compiler/flags yok; `BENCHMARKS.md` açıkça kaydediyor |
