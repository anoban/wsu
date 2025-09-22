#include <iostream>
#include <cstdlib>

inline namespace fold_expressions {

	template<typename _TyFirst> [[nodiscard]] static consteval long double sum(const _TyFirst& _val) noexcept {
			return _val;
	}

	template<typename _TyFirst, typename... _TyRest> [[nodiscard]] static consteval long double sum(const _TyFirst& _val, const _TyRest&... _vals) noexcept {
			return _val + fold_expressions::sum(_vals...);
	}

}

auto main() -> int {


	::wprintf_s(L"Sum of 99, 72, 42,  7, 39, 72, 98, 36, 46, 91, 75, 45, 30, 51, 81, 93, 32, 67, 48, 30, 57, 60, 39, 18, 36, 16,  6, 41, 40, 96 is %.5Lf\n", fold_expressions::sum(99, 72, 42,  7, 39, 72, 98, 36, 46, 91, 75, 45, 30, 51, 81, 93, 32,
       67, 48, 30, 57, 60, 39, 18, 36, 16,  6, 41, 40, 96));
	::_putws(L"Was that 1563?");

	return EXIT_SUCCESS;	
}
