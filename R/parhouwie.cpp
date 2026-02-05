// clang .\parhouwie.c -Wall -Wextra -static -march=native -DNDEBUG -O3 -std=c++20
// launch the R interpretor in parallel for the hOUwie model fits

// clang-format off
#define _AMD64_ // architecture
#define WIN32_LEAN_AND_MEAN
#include <errhandlingapi.h>
#include <libloaderapi.h>
#include <processthreadsapi.h>
#include <sysinfoapi.h>
#include <WinDef.h>
#include <WinBase.h>
#include <WinUser.h>
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)

namespace utils {

    // get the string representation of a _WIN32 error code
    static inline const wchar_t* __stdcall error_code_to_string( // NOLINT(readability-redundant-inline-specifier)
        _In_ const unsigned long errcode
    ) noexcept {
        static constexpr unsigned long long ERROR_MSG_BUFFSIZE { 512 }; // length of the error message buffer in number of wchar_t s
        static wchar_t                      errmsgbuffer[ERROR_MSG_BUFFSIZE] = { 0 }; // needs to be in static memory
        // without this the previously written buffer can get partially overwritten and returned in subsequent function invocations
        ::memset(errmsgbuffer, 0, sizeof(errmsgbuffer));

        unsigned long nbyteswritten = ::FormatMessageW(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, errcode, 0, errmsgbuffer, ERROR_MSG_BUFFSIZE, nullptr
        );

        if (!nbyteswritten) { // will be 0 if the call above to FormatMessageW failed; if that, the error string is not found in the system, try Ntdsbmsg.dll
            HINSTANCE handle_ntdsbmsg = ::LoadLibraryW(L"Ntdsbmsg.dll");
            if (!handle_ntdsbmsg) { // will be NULL if the DLL failed to load
                ::fputws(L"Failed to load Ntdsbmsg.dll", stderr);
                return errmsgbuffer; // must be an ampty buffer here
            }

            nbyteswritten = ::FormatMessageW(
                FORMAT_MESSAGE_FROM_HMODULE | FORMAT_MESSAGE_IGNORE_INSERTS,
                handle_ntdsbmsg,
                errcode,
                0,
                errmsgbuffer,
                ERROR_MSG_BUFFSIZE,
                nullptr
            );
            ::FreeLibrary(handle_ntdsbmsg); // detach the DLL from the process
        }
        return errmsgbuffer;
    }

} // namespace utils

namespace houwie {

    enum class DISCRETE_MODELS : unsigned char {
        ER,  // all rates are identical
        SYM, // symmetrically identical rates
        ARD  // all rates are allowed to be different (asymmetrically)
    };

    enum class CONTINUOUS_MODELS : unsigned char {
        OUM,
        OUMA [[deprecated("OUwie currently recommends not using the variable alpha models for continuous trait evolution")]],
        OUMV,
        OUMVA [[deprecated("OUwie currently recommends not using the variable alpha models for continuous trait evolution")]]
    };

    [[nodiscard]] static inline std::wstring __stdcall generate_rscript( // NOLINT(readability-redundant-inline-specifier)
        const std::wstring&       phylogeny,
        const std::wstring&       traitdata,
        const DISCRETE_MODELS&    discrete_model,
        const CONTINUOUS_MODELS&  continuous_model,
        const std::wstring&       savedir,
        const std::wstring&       conttrait,
        const std::wstring&       disctrait,
        const std::wstring&       suffix,
        const bool&               null_model,
        const unsigned&           rate_cat,
        const unsigned long long& nsims = 30
    ) noexcept {
        static constexpr wchar_t TEMPLATE_SCRIPT[] {
            L"suppressPackageStartupMessages({"
            L"    library('ape')"
            L"    library('corHMM')"
            L"    library('OUwie')"
            L"})"
            L"phylogeny <- ape::read.tree(\"{}\")"
            L"data <- read.csv(\"{}\")"
            L"stopifnot(all(phylogeny$tip.label == data$binominal))"
            L"model <- OUwie::hOUwie(phy = phylogeny, data = data, rate.cat = {}, discrete_model = \"{}\", continuous_model = \"{}\", nSim = {}, null.model = {})"
            L"saveRDS(object = model, file = \"{}\")"
        };

        return std::vformat(TEMPLATE_SCRIPT, std::make_wformat_args());
    }
}

// typically, each R process (inside Jupyter) only takes up about ~9% of the CPU, so this could absolutely benefit from paralellization
// wait 5 seconds between launching new processes, so we don't run into (possible???) file I/O issues inside the R instances

static wchar_t DUMMY_R_COMMANDLINE[512] {
    // the readline() at the end will keep the console window open until a user input is provided
    L"C:/R-4.5.2/bin/Rscript.exe --no-save -e \"write.csv(installed.packages()[, c('Package', 'Version')], file = paste0('./rpacks', sample(1:100, 1), '.csv'))\""
};

// the issue is that when the R interpreter gets called, the expression gets passed with all the quotes stripped away - leads to syntax errors
// figure out why the quotes get stripped away and how to preserve them when they are loaded into the R interpreter
// TURNS OUT THAT THE EXPRESSION ARGUMENT (-e) MUST BE ENCLOSED IN DOUBLE QUOTES NOT SINGLE QUOTES!!

int wmain(_In_ [[maybe_unused]] int argc, [[maybe_unused]] _In_ wchar_t* argv[]) {
    static const wchar_t* const         R_INTERPRETER_PATH { L"C:/R-4.5.2/bin/Rscript.exe" }; // the install directory of the R.exe binary
    static constexpr unsigned long long INTERPRETER_CMDLINE_BUFFSIZE { (1024 * 64) };         // being a bit too generous here
    SYSTEM_INFO                         sysinf { 0 };

    ::GetSystemInfo(&sysinf);
    ::wprintf_s(L"Number of processors: %lu\n", sysinf.dwNumberOfProcessors); // this machine has 18 cores, which is quite suprising
    // https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex

    ::_putws(DUMMY_R_COMMANDLINE);

    PROCESS_INFORMATION childprocinfo {};

    STARTUPINFOW childstarupinfo = {
        .cb          = sizeof(STARTUPINFOW),
        .dwFlags     = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES | STARTF_FORCEONFEEDBACK,
        .wShowWindow = SW_HIDE // hide the console window
    };

    // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw
    // https://learn.microsoft.com/en-us/windows/win32/procthread/creating-processes
    if (!::CreateProcessW(
            R_INTERPRETER_PATH, // DO NOT LEAVE THIS EMPTY I.E. nullptr
            DUMMY_R_COMMANDLINE,
            nullptr,
            nullptr,
            TRUE,
            HIGH_PRIORITY_CLASS | CREATE_NEW_CONSOLE,
            // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getpriorityclass
            // look up the above for process priorities and scheduling
            nullptr,
            nullptr,
            &childstarupinfo,
            &childprocinfo
        )) {
        ::fwprintf_s(stderr, L"%s error in call to CreateProcessW!\n", utils::error_code_to_string(::GetLastError()));
        return EXIT_FAILURE;
    }

    switch (::WaitForSingleObject(childprocinfo.hProcess, INFINITE)) { // wait for the child process to finish
        case WAIT_ABANDONED :
            ::fputws(L"Mutex object was not released by the child thread before the caller thread terminated.\n", stderr);
            break;
        case WAIT_TIMEOUT  : ::fputws(L"The time-out interval has elapsed, and the object's state is nonsignaled.\n", stderr); break;
        case WAIT_FAILED   : ::fwprintf_s(stderr, L"Error %lu: Wait failed.\n", ::GetLastError()); break;
        case WAIT_OBJECT_0 : ::_putws(L"Wait success!"); break; // The state of the specified object is signaled, wait success
        default            : break;
    }

    unsigned long childproc_exitcode = 0xFF;

    ::GetExitCodeProcess(childprocinfo.hProcess, &childproc_exitcode);
    ::wprintf_s(L"Exit code of the child process is %lu\n", childproc_exitcode);

    // https://learn.microsoft.com/en-us/windows/win32/procthread/creating-processes
    ::CloseHandle(childprocinfo.hProcess); // close the child process
    ::CloseHandle(childprocinfo.hThread);  // close the child thread

    return EXIT_SUCCESS;
}

// NOLINTEND(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)
