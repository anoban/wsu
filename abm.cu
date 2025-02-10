// clang-format off
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
// clang-format on

#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

class person final {
    public:
        static constexpr unsigned _RUMOUR_LENGTH { 200 };

    private:
        wchar_t _rumour[_RUMOUR_LENGTH]; // NOLINT(modernize-avoid-c-arrays)
        bool    _has_rumour;

    public:
        __host__ __device__ constexpr __stdcall person() noexcept : _rumour {}, _has_rumour {} { }

        __host__ __device__ explicit __stdcall person(const wchar_t* const _string) noexcept : _rumour {}, _has_rumour { true } {
            ::wcsncpy_s(_rumour, _string, _RUMOUR_LENGTH);
        }

        __host__ __device__ void __stdcall converse(const person& _other) noexcept {
            if (_other._has_rumour) { // if the other person has a rumour, listen to it
                ::wcsncpy_s(_rumour, _other._rumour, _RUMOUR_LENGTH);
                _has_rumour = true;
            }
        }

        __host__ __device__ constexpr bool has_rumour() const noexcept { return _has_rumour; }

        __host__                                        __device__ constexpr __stdcall person(const person&) noexcept = default;
        __host__                                        __device__ constexpr __stdcall person(person&&) noexcept      = default;
        __host__ __device__ constexpr person& __stdcall operator=(const person&) noexcept                             = default;
        __host__ __device__ constexpr person& __stdcall operator=(person&&) noexcept                                  = default;
        __host__                                        __device__ constexpr __stdcall ~person() noexcept             = default;

        // person + person
        [[nodiscard]] __host__ __device__ constexpr unsigned long long __stdcall operator+(const person& _other) const noexcept {
            return _has_rumour + _other._has_rumour;
        }

        // person + value
        [[nodiscard]] __host__ __device__ constexpr unsigned long long __stdcall operator+(const unsigned long long& _sum) const noexcept {
            return _has_rumour + _sum;
        }

        // value + person
        [[nodiscard]] __host__ __device__ friend constexpr unsigned long long __stdcall operator+(
            const unsigned long long _sum, const person& _other
        ) noexcept {
            return _sum + _other._has_rumour;
        }
};

static constexpr unsigned long population_size { 800'000 }, max_spreaders { 30 }, max_days { 100'000 }, max_contacts { 21 };

// inoculate the rumour in the population
template<unsigned long long _size>
static __global__ void __stdcall inoculate(_In_ const unsigned long long& _population_size, _In_ const wchar_t (&_string)[_size]) {
    curandGenerator_t dev_rndgen {};
    ::curandCreateGenerator(&dev_rndgen, curandRngType::CURAND_RNG_PSEUDO_MT19937);                     // create the
    ::curandSetPseudoRandomGeneratorSeed(dev_rndgen, static_cast<unsigned long long>(::time(nullptr))); // seed the random number generator
}

auto wmain() -> int {
    std::mt19937_64                         rengine { std::random_device {}() };
    std::uniform_int_distribution<unsigned> randint { 0, population_size - 1 };
    const person                            dumbass { L"There are aliens in area 51, my brother's friend in CIA told me!!" };

    std::vector<person>             population(population_size);
    std::vector<unsigned long long> daily_records(max_days);

    population.at(0).converse(dumbass); // the first point of contact

    person* device_vector {};
    ::cudaMalloc(&device_vector, sizeof(person) * population_size);

    ::inoculate<<<1, 1>>>(1000LLU, L"There are aliens in area 51, my brother's friend in CIA told me!!");

    // simulate subsequent contacts
    unsigned random_selection {}, contacts {}; // NOLINT(readability-isolate-declaration)
    std::wcout << L"population size :: " << population_size << L'\n';
    for (const auto& d : std::ranges::views::iota(0U, max_days)) {
        for (const auto& i : std::ranges::views::iota(0U, max_spreaders)) {
            random_selection = randint(rengine); // pick the spreader
            for (const auto& c : std::ranges::views::iota(0U, max_contacts)) {
                contacts = randint(rengine);                                       // spreader's contacts for the day
                population.at(random_selection).converse(population.at(contacts)); // make contact
            }
        }

        daily_records.at(d) = std::accumulate(population.cbegin(), population.cend(), 0LU);

        std::wcout << daily_records.at(d) / static_cast<double>(population_size) << L',';
    }

    return EXIT_SUCCESS;
}
