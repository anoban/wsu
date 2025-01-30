#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

class person final {
    private:
        static constexpr unsigned _RUMOUR_LENGTH { 200 };
        wchar_t                   _rumour[_RUMOUR_LENGTH]; // NOLINT(modernize-avoid-c-arrays)
        bool                      _has_rumour;

    public:
        __host__ __device__ __stdcall person() noexcept : _rumour {}, _has_rumour {} { }

        __host__ __device__ explicit __stdcall person(const wchar_t* const _string) noexcept : _rumour {}, _has_rumour { true } {
            ::wcsncpy_s(_rumour, _string, _RUMOUR_LENGTH);
        }

        __host__ __device__ void __stdcall converse(const person& _other) noexcept {
            if (_other._has_rumour) { // if the other person has a rumour, listen to it
                ::wcsncpy_s(_rumour, _other._rumour, _RUMOUR_LENGTH);
                _has_rumour = true;
            }
        }

        __host__ __device__ bool has_rumour() const noexcept { return _has_rumour; }

        __host__                              __device__ __stdcall person(const person&) = default;
        __host__                              __device__ __stdcall person(person&&)      = default;
        __host__ __device__ person& __stdcall operator=(const person&)                   = default;
        __host__ __device__ person& __stdcall operator=(person&&)                        = default;
        __host__                              __device__ __stdcall ~person() noexcept    = default;

        // person + person
        __host__ __device__ unsigned long long __stdcall operator+(const person& _other) const noexcept {
            return _has_rumour + _other._has_rumour;
        }

        // person + value
        __host__ __device__ unsigned long long __stdcall operator+(const unsigned long long& _sum) const noexcept {
            return _has_rumour + _sum;
        }

        // value + person
        __host__ __device__ friend constexpr unsigned long long __stdcall operator+(
            const unsigned long long _sum, const person& _other
        ) noexcept {
            return _sum + _other._has_rumour;
        }
};

static constexpr unsigned long population_size { 800'000 }, max_spreaders { 30 }, max_days { 100'000 }, max_contacts { 21 };

auto wmain() -> int {
    std::mt19937_64                         rengine { std::random_device {}() };
    std::uniform_int_distribution<unsigned> randint { 0, population_size - 1 };
    const person                            dumbass { L"There are aliens in area 51, my brother's friend in CIA told me!!" };

    std::vector<person>             population(population_size);
    std::vector<unsigned long long> daily_records(max_days);

    population.at(0).converse(dumbass); // the first point of contact

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
