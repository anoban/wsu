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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)

namespace utils {

    // get the string representation of a _WIN32 error code
    // NOLINTNEXTLINE(readability-redundant-inline-specifier)
    [[nodiscard]] static inline const wchar_t* __stdcall error_code_to_string(_In_ const unsigned long& errcode) noexcept {
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

    // NOLINTNEXTLINE(readability-redundant-inline-specifier)
    [[clang::always_inline, nodiscard]] static inline constexpr const wchar_t* __stdcall __discmod_to_wstr(
        _In_ const DISCRETE_MODELS& model
    ) noexcept {
        switch (model) {
            case DISCRETE_MODELS::ER  : return L"ER"; break;
            case DISCRETE_MODELS::SYM : return L"SYM"; break;
            case DISCRETE_MODELS::ARD : return L"ARD"; break;
        }
    }

    // NOLINTNEXTLINE(readability-redundant-inline-specifier)
    [[clang::always_inline, nodiscard]] static inline constexpr const wchar_t* __stdcall __contmod_to_wstr(
        _In_ const CONTINUOUS_MODELS& model
    ) noexcept {
        switch (model) {
            case CONTINUOUS_MODELS::OUM   : return L"OUM"; break;
            case CONTINUOUS_MODELS::OUMA  : return L"OUMA"; break;
            case CONTINUOUS_MODELS::OUMV  : return L"OUMV"; break;
            case CONTINUOUS_MODELS::OUMVA : return L"OUMVA"; break;
        }
    }

    [[clang::always_inline, nodiscard]] static inline const wchar_t* __stdcall __path_to_serialize( // NOLINT(readability-redundant-inline-specifier)
        _In_ const DISCRETE_MODELS&   discrete_model,
        _In_ const CONTINUOUS_MODELS& continuous_model,
        _In_ const wchar_t* const     savedir, // assumed ends with a forward slash, expected to be in the format "C:/Users/Documents/"
        _In_ const wchar_t* const     conttrait,
        _In_ const wchar_t* const     disctrait,
        _In_ const bool&              nullmodel,
        _In_ const wchar_t* const     suffix
    ) noexcept {
        static constexpr unsigned long long MAX_MODEL_NAME_LENGTH { MAX_PATH }; // 260
        static wchar_t                      buffer[MAX_MODEL_NAME_LENGTH] {};
        ::memset(buffer, 0, sizeof(buffer));
        // e.g. ARD_OUMV_RD_MYCO_CD_395sp.Rds
        ::swprintf_s(
            buffer,
            MAX_MODEL_NAME_LENGTH,
            L"%s%s_%s_%s_%s_%s_%s.Rds",
            savedir,
            __discmod_to_wstr(discrete_model),
            __contmod_to_wstr(continuous_model),
            conttrait,
            disctrait,
            nullmodel ? L"CID" : L"CD",
            suffix
        );
        return buffer;
    }

    [[nodiscard]] static inline std::wstring __stdcall generate_rscript( // NOLINT(readability-redundant-inline-specifier)
        _In_ const wchar_t* const       phylogeny,
        _In_ const wchar_t* const       traitdata,
        _In_ const DISCRETE_MODELS&    discrete_model,
        _In_ const CONTINUOUS_MODELS&  continuous_model,
        _In_ const wchar_t* const       savedir,
        _In_ const wchar_t* const       conttrait,
        _In_ const wchar_t* const       disctrait,
        _In_ const wchar_t* const       suffix,
        _In_ const bool&               null_model,
        _In_ const unsigned long long& nsims = 30
    ) noexcept {
        static constexpr unsigned long long BUFFSIZE { 0xFFF };
        std::wstring                        buffer {};
        buffer.resize(BUFFSIZE);

        ::swprintf_s(
            buffer.data(),
            BUFFSIZE,
            // who gives a damn when warnings are emiited during package loading during automation
            // also using ; instead of new lines to delineate expressions (expressions separated by \n s did not work)
            // and when passed as expressions, all the double quotes get stripped away for some reason?????, using single quotes for string literals
            L"library('ape');"
            L"library('corHMM');"
            L"library('OUwie');"
            L"phylogeny <- ape::read.tree('%s');"
            L"data <- read.csv('%s');"
            L"stopifnot(all(phylogeny$tip.label == data$binominal));"
            L"model <- OUwie::hOUwie(phy = phylogeny, data = data, rate.cat = %1u, discrete_model = '%s', continuous_model = '%s', nSim = %u, null.model = %s);"
            L"saveRDS(object = model, file = '%s');",
            phylogeny,
            traitdata,
            // if null_model is true, then it's a CID model with 2 rate categories, else it's a CD model with just 1 rate category
            null_model ? 2 : 1,
            __discmod_to_wstr(discrete_model),
            __contmod_to_wstr(continuous_model),
            nsims,
            null_model ? L"TRUE" : L"FALSE",
            __path_to_serialize(discrete_model, continuous_model, savedir, conttrait, disctrait, null_model, suffix)
        );
        return buffer;
    }
} // namespace houwie

// typically, each R process (inside Jupyter) only takes up about ~9% of the CPU, so this could absolutely benefit from paralellization
// wait 5 seconds between launching new processes, so we don't run into (possible???) file I/O issues inside the R instances

// the issue is that when the R interpreter gets called, the expression gets passed with all the quotes stripped away - leads to syntax errors
// figure out why the quotes get stripped away and how to preserve them when they are loaded into the R interpreter
// TURNS OUT THAT THE EXPRESSION ARGUMENT (-e) MUST BE ENCLOSED IN DOUBLE QUOTES NOT SINGLE QUOTES!!

// R also seems to skip the assertion like expressions e.g. stopifnot() and the likes when non-interactively invoked with expressions (using -e)

int wmain(_In_ [[maybe_unused]] int argc, [[maybe_unused]] _In_ wchar_t* argv[]) {
    static const wchar_t* const         R_INTERPRETER_PATH { L"C:/R-4.5.2/bin/R.exe" }; // the install directory of the R.exe binary
    static constexpr unsigned long long CMDLINE_BUFFSIZE { 0x2FFF };                    // being a bit too generous here
    SYSTEM_INFO                         sysinf { 0 };

    ::GetSystemInfo(&sysinf);
    ::wprintf_s(L"Number of processors: %lu\n", sysinf.dwNumberOfProcessors); // this machine has 18 cores, which is quite suprising
    // https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex

    PROCESS_INFORMATION childprocinfo {};

    STARTUPINFOW childstarupinfo = { .cb          = sizeof(STARTUPINFOW),
                                     .dwFlags     = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES | STARTF_FORCEONFEEDBACK,
                                     .wShowWindow = SW_SHOW };

    const std::wstring script    = houwie::generate_rscript(
        LR"(./ouwie_64sp_example.tre)",
        LR"(./ouwie_64sp_trait_example.csv)",
        houwie::DISCRETE_MODELS::ER,
        houwie::CONTINUOUS_MODELS::OUMA,
        LR"(./rdata/)",
        L"X",
        L"REGIME",
        L"s",
        false,
        100
    );

    std::wstring cmdline {};
    cmdline.resize(CMDLINE_BUFFSIZE);
    // the double quotation marks enclosing the expression (-e) argument are absolutely critical
    ::swprintf_s(cmdline.data(), CMDLINE_BUFFSIZE, L"%s --no-save -e \"%s\"", R_INTERPRETER_PATH, script.c_str());
    ::_putws(cmdline.c_str());

    // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw
    // https://learn.microsoft.com/en-us/windows/win32/procthread/creating-processes
    if (!::CreateProcessW(
            R_INTERPRETER_PATH, // DO NOT LEAVE THIS EMPTY i.e. nullptr
            cmdline.data(),
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

    /*
    for (unsigned dmod = 0; dmod < 3; ++dmod) {     // discrete models
        for (unsigned cmod = 0; cmod < 4; ++cmod) { // continuous models
            for (unsigned nm = 0; nm < 2; ++nm) {   // null model (0, 1) i.e true or false
                const std::wstring script = houwie::generate_rscript(
                    LR"(../data/chapter2/uphylomaker/FRED_subset_collab_1005sp.tre)",
                    LR"(../data/chapter2/FREDv3subset/collab_rdlteq1_rd1ornan_log_RD_SRL_species_avgd.csv)",
                    static_cast<houwie::DISCRETE_MODELS>(dmod),
                    static_cast<houwie::CONTINUOUS_MODELS>(cmod),
                    LR"(./rdata/)",
                    L"SRL",
                    L"STATES",
                    L"1005sp",
                    nm,
                    100
                );
                ::_putws(script.c_str());
            }
        }
    }
    */

    return EXIT_SUCCESS;
}

// NOLINTEND(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)
