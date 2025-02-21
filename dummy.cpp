#include <algorithm>
#include <cstdlib>
#include <format>
#include <iostream>
#include <numeric>
#include <ratio>
#include <vector>

auto wmain() -> int {
    std::vector<float> randoms(100'000'000);
    std::generate(randoms.begin(), randoms.end(), ::rand);
    const auto sum = std::accumulate(randoms.cbegin(), randoms.cend(), 0.0L);
    std::wcout << std::format(L"Sum is {:.10Lf}\n", sum);
    return EXIT_SUCCESS;
}
