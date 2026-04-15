#include <cstdlib>
#include <ctime>

// #define __RETURN_CONSTANT 1

auto wmain([[maybe_unused]] int argc, [[maybe_unused]] wchar_t* argv[]) -> int {
    ::srand(::time(nullptr));
#ifdef __RETURN_0
    return EXIT_SUCCESS;
#elif defined(__RETURN_CONSTANT)
    return 1275;
#else
    return ::rand();
#endif
}
