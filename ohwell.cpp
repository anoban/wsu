#include <concepts>
#include <iostream>
#include <type_traits>

#include <sal.h> // for g++

template<class _TyChar> concept is_iostream_compatible = std::is_same_v<char, _TyChar> || std::is_same_v<wchar_t, _TyChar>;

template<class _TyChar> requires ::is_iostream_compatible<_TyChar>
std::basic_ostream<_TyChar>& comma(_Inout_ std::basic_ostream<_TyChar>& _ostr) noexcept(noexcept(_ostr.put(static_cast<_TyChar>(',')))) {
    if constexpr (std::is_same_v<_TyChar, char>)
        _ostr.put(',');
    else
        _ostr.put(L',');
    return _ostr;
}

auto main() -> int {
    std::cout << "Hi there John" << ::comma << " Jane and Jacob" << std::endl;
    return EXIT_SUCCESS;
}
