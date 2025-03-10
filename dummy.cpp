#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <execution>
#include <format>
#include <iostream>
#include <numeric>
#include <ratio>
#include <vector>

template<typename _TyFirst, typename... _TyRest> static consteval long double sum(const _TyFirst& arg, const _TyRest&... rest) noexcept {
    return (arg + ... + rest);
}

template<class _Ty> [[nodiscard]] static consteval long double prod(const _Ty& _input) noexcept { return _input; }

template<class _TyFirst, class... _TyRest>
[[nodiscard]] static consteval long double prod(const _TyFirst& _argf, const _TyRest&... _argr) noexcept {
    return _argf * ::prod(_argr...);
}

static_assert(::prod(1, 2, 3, 4, 5, 6, 7, 8, 9) == 362880);
static_assert(::prod(11, 12, 13, 14, 15, 16, 17, 18, 19) == 33522128640);
static_assert(::prod(0, 12, 13, 14, 15, 16, 17, 18, 19) == 0);

template<typename _TyFirst, typename... _TyPack>
[[nodiscard]] static consteval long double ssum(const _TyFirst& argf, const _TyPack&... argr) noexcept {
    if constexpr (!sizeof...(_TyPack))
        return argf;
    else
        return argf * ::ssum(argr...);
}

auto wmain() -> int {
    ::srand(std::chrono::steady_clock::now().time_since_epoch().count()); // ::time(nullptr), fucking C++ idioms
    std::vector<float> randoms(100'000'000);
    std::generate(std::execution::parallel_unsequenced_policy {}, randoms.begin(), randoms.end(), []() noexcept -> float {
        return ::rand() / static_cast<float>(RAND_MAX); // NOLINT(cppcoreguidelines-narrowing-conversions)
    });
    const auto&& sum = std::reduce(std::execution::parallel_unsequenced_policy {}, randoms.cbegin(), randoms.cend(), 0.0L);
    std::wcout << std::format(L"Sum is {:.10Lf}\n", sum);

    std::wcout << std::format(
        L"Sum is {:.10Lf}\n", // 48.75263192701636
        ::sum(
            0.10935386,
            0.06900139,
            0.5788174,
            0.19613282,
            0.48033021,
            0.07079293,
            0.84113776,
            0.52200446,
            0.74132568,
            0.60679746,
            0.49308194,
            0.81371664,
            0.39470257,
            0.08008278,
            0.2897574,
            0.01962978,
            0.84347144,
            0.95845645,
            0.2177334,
            0.16798786,
            0.78897779,
            0.35559866,
            0.26260677,
            0.56207042,
            0.74992883,
            0.65694657,
            0.56462049,
            0.05005654,
            0.89898416,
            0.31854104,
            0.45036904,
            0.76320521,
            0.74689053,
            0.82057483,
            0.45322377,
            0.62002086,
            0.6318527,
            0.20689088,
            0.8532891,
            0.21375683,
            0.97829674,
            0.53646426,
            0.46029885,
            0.95607277,
            0.35847282,
            0.41472668,
            0.29841233,
            0.19127843,
            0.39820698,
            0.38683719,
            0.17416589,
            0.04043982,
            0.06945854,
            0.56306712,
            0.81639645,
            0.39497424,
            0.33032778,
            0.8712706,
            0.14554415,
            0.84991162,
            0.73142854,
            0.42212841,
            0.11285257,
            0.41080013,
            0.54451398,
            0.99177095,
            0.39959419,
            0.36859316,
            0.2700977,
            0.95406255,
            0.49388748,
            0.74441906,
            0.88686591,
            0.87241902,
            0.30011177,
            0.38290519,
            0.49740014,
            0.39997921,
            0.50642487,
            0.92274731,
            0.52228936,
            0.61109578,
            0.90177256,
            0.97701555,
            0.11679246,
            0.02116926,
            0.43952457,
            0.98339754,
            0.35596019,
            0.40587468,
            0.61227551,
            0.72327056,
            0.21568937,
            0.28905425,
            0.11758979,
            0.34666426,
            0.49922016,
            0.28945899,
            0.12444429,
            0.28972811
        )
    );
    return EXIT_SUCCESS;
}
