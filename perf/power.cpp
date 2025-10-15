// statistical functions that need to be handrolled or stolen from GSL :)
// t ppf
// nct sf
// f ppf
// ncf sf
// norm cdf
// chi2 ppf
// ncchi2 sf
// brenth

#include <cstdlib>

namespace stats {

    // the nc prefix stands for noncentral variants of the distributions

    namespace t {
        constexpr inline long double __stdcall ppf(
            _In_ const long double& quantile, _In_ const long long& dof, _In_ const long double& loc, _In_ const long double& scale
        ) noexcept { }
    } // namespace t

    namespace nct {
        constexpr inline long double __stdcall sf() noexcept { }
    } // namespace nct

    namespace f {
        constexpr inline long double __stdcall ppf() noexcept { }
    } // namespace f

    namespace ncf {
        constexpr inline long double __stdcall sf() noexcept { }
    } // namespace ncf

    namespace norm {
        constexpr inline long double __stdcall cdf() noexcept { }
    } // namespace norm

    namespace chisq {
        constexpr inline long double __stdcall ppf() noexcept { }
    } // namespace chisq

    namespace ncchisq {
        constexpr inline long double __stdcall sf() noexcept { }
    } // namespace ncchisq

} // namespace stats

int wmain() {
    //
    return EXIT_SUCCESS;
}
