#include <cstdlib>

#define __RETURN_CONSTANT 1

auto wmain([[maybe_unused]] int argc, [[maybe_unused]] wchar_t* argv[]) -> int {
#ifdef __RETURN_0
    return EXIT_SUCCESS;
#elif defined(__RETURN_CONSTANT)
    return 124;
#else
    return ::rand();
#endif
}
