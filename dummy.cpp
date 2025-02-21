#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <execution>
#include <format>
#include <iostream>
#include <numeric>
#include <ratio>
#include <vector>

auto wmain() -> int {
    ::srand(::time(nullptr));
    std::vector<float> randoms(100'000'000);
    std::generate(std::execution::parallel_unsequenced_policy {}, randoms.begin(), randoms.end(), []() noexcept -> float {
        return ::rand() / static_cast<float>(RAND_MAX);
    });
    const auto&& sum = std::reduce(std::execution::parallel_unsequenced_policy {}, randoms.cbegin(), randoms.cend(), 0.0L);
    std::wcout << std::format(L"Sum is {:.10Lf}\n", sum);
    return EXIT_SUCCESS;
}
