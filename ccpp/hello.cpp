#include <cstdio>
#include <cstdlib>

static inline void __cdecl say_goodbye(void) { ::_putws(L"I'm about to head out, good bye!"); }

int wmain(_In_ const int argc, _In_reads_(argc) const wchar_t* argv[]) {
    if (::atexit(::say_goodbye) /* atexit() returns 0 if succeeds */) { // NOLINT(readability-implicit-bool-conversion)
        ::fputws(L"atexit() registration failed!", stderr);
        return EXIT_FAILURE;
    }
    for (long long i = 0; i < argc; i++) ::wprintf_s(L"Argument number %04lld: %s\n", i, argv[i]); // I still got it :)

    return EXIT_SUCCESS;
}
