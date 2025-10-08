#include <stdio.h>
#include <stdlib.h>

static inline void __cdecl say_goodbye(void) { _putws(L"I'm about to head out!"); }

int wmain(_In_ const int argc, _In_reads_(argc) const wchar_t* argv[]) {
    if (atexit(say_goodbye)) fputws(L"atexit() registration failed!", stderr);
    for (long long i = 0; i < argc; i++) wprintf_s(L"Argument number %04lld: %s\n", i, argv[i]);

    return EXIT_SUCCESS;
}
