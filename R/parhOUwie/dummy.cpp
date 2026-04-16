#include <cstdlib>
#include <ctime>

#define _AMD64_
#define WIN32_LEAN_AND_MEAN
#include <synchapi.h>

// #define __RETURN_CONSTANT 1

auto wmain([[maybe_unused]] int argc, [[maybe_unused]] wchar_t* argv[]) -> int {
#ifdef __RETURN_0
    return EXIT_SUCCESS;
#elif defined(__RETURN_CONSTANT)
    return 1274;
#else
    ::srand(::time(nullptr));
    ::Sleep(10 * (::rand() % 100));
    return ::rand() * 2;
#endif
}
