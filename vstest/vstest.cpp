#include <algorithm>
#include <cstdlib>
#include <execution>
#include <format>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

auto wmain() -> int {
    std::random_device                     rdev {};
    std::mt19937_64                        rngine { rdev() };
    std::uniform_real_distribution<double> ureal {};

    std::vector<double> nums(5'000'000);
    std::generate(std::execution::par_unseq, nums.begin(), nums.end(), [&rngine, &ureal]() noexcept -> double { return ureal(rngine); });

    std::wcout << std::format(L"Sum is {:.6Lf}\n", std::reduce(std::execution::par_unseq, nums.cbegin(), nums.cend())).c_str();
    return EXIT_SUCCESS;
}
