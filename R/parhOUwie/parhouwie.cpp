#if !defined(_WIN32) && !defined(_WIN64) && (!defined(_MSC_VER) || !defined(_MSC_FULL_VER))
    #error This is a Windows only implementation, not meant to be used on other platforms!.
#endif

// clang .\parhouwie.cpp -Wall -Wextra -static -march=native -DNDEBUG -D_NDEBUG -O3 -std=c++20 -o .\parhouwie.exe
// cl .\parhouwie.cpp /Wall /std:c++20 /O2 /MT /EHsc

#if defined(_MSC_FULL_VER) && !defined(__llvm__) // MSVC specific warnings
    #pragma warning(disable : 4710 4711 4820)
#endif

// clang-format off
#define _AMD64_ // architecture
#define WIN32_LEAN_AND_MEAN
#include <errhandlingapi.h>
#include <libloaderapi.h>
#include <processthreadsapi.h>
#include <synchapi.h>
#include <sysinfoapi.h>
#include <WinDef.h>
#include <WinBase.h>
#include <WinUser.h>
// clang-format on

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HOUWIE_VARIABLE_ALPHA_WARNING                                                                            \
    "Warning: as of OUwie version 2.16, users are temporarily discouraged from using the variable alpha models!"
static const wchar_t* const         R_INTERPRETER_PATH { L"C:/R-4.5.2/bin/R.exe" }; // the install directory of the R.exe binary
static constexpr unsigned long long CMDLINE_BUFFSIZE { 0x2FFF };                    // being a bit too generous here
static constexpr unsigned long long TOTAL_PROCESSES { 24 };               // 4 continuous models x 3 discrete models x 2 rate categories
static constexpr unsigned long long ERROR_MSG_BUFFSIZE { 512 };           // length of the error message buffer in number of wchar_t s
static constexpr unsigned long long MAX_SAVERDS_NAME_LENGTH { MAX_PATH }; // 260
static constexpr unsigned long long RSCRIPT_BUFFSIZE { 0xFFF };
static constexpr unsigned long long MAX_PARALLEL_PROCESSES { 10 }; // a decent number with enough CPU space for other essential processes
static HINSTANCE                    handle_ntdsbmsg {}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables) handle to Ntdsbmsg.dll

extern "C" inline void __cdecl __release_ntdbsdll() noexcept {
    if (handle_ntdsbmsg) ::FreeLibrary(handle_ntdsbmsg);
}

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)

namespace houwie {
    enum class DISCRETE_MODELS : unsigned char {
        ER,  // all rates are identical
        SYM, // symmetrically identical rates
        ARD  // all rates are allowed to be different (asymmetrically)
    };

    enum class CONTINUOUS_MODELS : unsigned char {
        OUM, // only the continuous trait optimum varies depending on the discrete state regimes
        OUMA [[deprecated(
            HOUWIE_VARIABLE_ALPHA_WARNING
        )]],  // the continuous trait optimum and the pull towards the optimum vary depending on the discrete state regimes
        OUMV, // the continuous trait optimum and the rate of continuous trait evolution vary depending on the discrete state regimes
        OUMVA [[deprecated(
            HOUWIE_VARIABLE_ALPHA_WARNING
        )]] // optima, rate of evolution and the pull towards the optima of the continuous trait vary depending on the discrete state regimes
    };

    // NOLINTNEXTLINE(readability-redundant-inline-specifier)
    [[clang::always_inline, nodiscard]] static inline constexpr const wchar_t* __stdcall __discmod_to_wstr(
        _In_ const DISCRETE_MODELS& model
    ) noexcept {
        switch (model) {
            case DISCRETE_MODELS::ER  : return L"ER";
            case DISCRETE_MODELS::SYM : return L"SYM";
            case DISCRETE_MODELS::ARD : return L"ARD";
        }
    }

    // NOLINTNEXTLINE(readability-redundant-inline-specifier)
    [[clang::always_inline, nodiscard]] static inline constexpr const wchar_t* __stdcall __contmod_to_wstr(
        _In_ const CONTINUOUS_MODELS& model
    ) noexcept {
        switch (model) {
            case CONTINUOUS_MODELS::OUM   : return L"OUM";
            case CONTINUOUS_MODELS::OUMA  : return L"OUMA";
            case CONTINUOUS_MODELS::OUMV  : return L"OUMV";
            case CONTINUOUS_MODELS::OUMVA : return L"OUMVA";
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
        static wchar_t buffer[MAX_SAVERDS_NAME_LENGTH] {};
        ::memset(buffer, 0, sizeof(buffer)); // we don't want buffer contents from previous writes intefereing with new writes
        // e.g. ARD_OUMV_RD_MYCO_CD_395sp.Rds
        ::swprintf_s(
            buffer,
            MAX_SAVERDS_NAME_LENGTH,
            L"%s%s_%s_%s_%s_%s_%s.Rds", // we expect the directory path to end with a forward slash
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

    [[clang::always_inline]] static inline void __stdcall generate_rscript( // NOLINT(readability-redundant-inline-specifier)
        _Inout_ std::wstring& buffer,
        _In_ const wchar_t* const      phylogeny,
        _In_ const wchar_t* const      traitdata,
        _In_ const DISCRETE_MODELS&    discrete_model,
        _In_ const CONTINUOUS_MODELS&  continuous_model,
        _In_ const wchar_t* const      savedir,
        _In_ const wchar_t* const      conttrait,
        _In_ const wchar_t* const      disctrait,
        _In_ const wchar_t* const      suffix,
        _In_ const bool&               null_model,
        _In_ const unsigned long long& nsims = 30
    ) noexcept {
        if (buffer.size() < RSCRIPT_BUFFSIZE) buffer.resize(RSCRIPT_BUFFSIZE);
        ::memset(buffer.data(), 0, buffer.size() * sizeof(wchar_t)); // clean up the buffer before every new write

        ::swprintf_s(
            buffer.data(),
            buffer.size(),
            // who gives a damn when warnings are emiited during package loading in automation
            // also using ; instead of new lines to delineate expressions (expressions separated by \n s did not work for some reason???)
            // and when passed as expressions, all the double quotes get stripped away for some reason?????, using single quotes instead for string literals
            L"library('ape');"
            L"library('corHMM');"
            L"library('OUwie');"
            L"phylogeny <- ape::read.tree('%s');"
            L"data <- read.csv('%s');"
            L"stopifnot(all(phylogeny$tip.label == data$binominal));"
            L"model <- OUwie::hOUwie(phy = phylogeny, data = data, rate.cat = %1u, discrete_model = '%s', continuous_model = '%s', nSim = %llu, null.model = %s);"
            L"saveRDS(object = model, file = '%s');",
            phylogeny,
            traitdata,
            // if null_model is true, then it's a CID model with 2 rate categories, else it's a CD model with just 1 rate category
            null_model ? 2U : 1U,
            __discmod_to_wstr(discrete_model),
            __contmod_to_wstr(continuous_model),
            nsims,
            null_model ? L"TRUE" : L"FALSE",
            __path_to_serialize(discrete_model, continuous_model, savedir, conttrait, disctrait, null_model, suffix)
        );
    }

} // namespace houwie

namespace utils {

    // get the string representation of a _WIN32 error code
    // NOLINTNEXTLINE(readability-redundant-inline-specifier)
    [[nodiscard, clang::always_inline]] static inline const wchar_t* __stdcall error_code_to_wstring(
        _In_ const unsigned long& errcode
    ) noexcept {
        static wchar_t errmsgbuffer[ERROR_MSG_BUFFSIZE] = { 0 }; // needs to be in static memory
        // without this the previously written buffer can get partially overwritten and returned in subsequent function invocations
        ::memset(errmsgbuffer, 0, sizeof(errmsgbuffer));

        unsigned long nbyteswritten = ::FormatMessageW(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, errcode, 0, errmsgbuffer, ERROR_MSG_BUFFSIZE, nullptr
        );

        if (!nbyteswritten) { // will be 0 if the call above to FormatMessageW failed; if that, the error string is not found in the system, try Ntdsbmsg.dll
            // if the library hasn't already been loaded by previous calls to this function
            if (!handle_ntdsbmsg) handle_ntdsbmsg = ::LoadLibraryW(L"Ntdsbmsg.dll");
            if (!handle_ntdsbmsg) { // will be NULL if the DLL failed to load
                ::fputws(L"Failed to load Ntdsbmsg.dll", stderr);
                return errmsgbuffer; // must be an empty buffer here
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
            // ::FreeLibrary(handle_ntdsbmsg); // detach the DLL from the process - atexit() will handle this for us
        }
        return errmsgbuffer;
    }

    [[nodiscard, clang::always_inline]] static inline bool __stdcall handle_parallel_waits( // NOLINT(readability-redundant-inline-specifier)
        _In_ const unsigned long& retval,  _Inout_ std::vector<HANDLE64>& active_proc_handles, _Inout_ std::vector<HANDLE64>& active_thread_handles
    ) noexcept {
        // WAIT_OBJECT_0 is defined as 0 and WAIT_ABANDONED_0 is defined as 0x00000080L
        // so retval is practically equal to the offset of the signalled handle WHEN WAIT SUCCEEDS
        unsigned long pop_offset { 0xFF };

        // range WAIT_OBJECT_0 to (WAIT_OBJECT_0 + nCount - 1) indicates success
        if (retval < active_proc_handles.size()) pop_offset = retval - WAIT_OBJECT_0;
        // range WAIT_ABANDONED_0 to (WAIT_ABANDONED_0 + nCount - 1) indiacate an abandoned mutex
        else if ((retval >= WAIT_ABANDONED_0) && (retval < WAIT_TIMEOUT)) {
            pop_offset = retval - WAIT_ABANDONED_0;
            ::fputws(L"WaitForMultipleObjects signalled WAIT_ABANDONED", stderr);
        } else if (retval == WAIT_TIMEOUT) // cannot (should not) happen as we specified the time limit to be INFINITE
            ::fputws(L"WaitForMultipleObjects signalled WAIT_TIMEOUT", stderr);
        else if (retval == WAIT_FAILED)
            ::fwprintf_s(stderr, L"WaitForMultipleObjects signalled WAIT_FAILED, %s\n", error_code_to_wstring(::GetLastError()));

        if (pop_offset != 0xFF) { // no matter what, we have one less active process now
            // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic) - close the process and thread handles of the signalled process
            ::CloseHandle(*(active_proc_handles.data() + pop_offset));
            ::CloseHandle(*(active_thread_handles.data() + pop_offset));
            // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic) - close that handle
            active_proc_handles.erase(active_proc_handles.begin() + pop_offset); // remove the signalled handle
            active_thread_handles.erase(active_thread_handles.begin() + pop_offset);
            // active_proc_handles.shrink_to_fit();
            return true;
        }

        return false; // WAIT_TIMEOUT or WAIT_FAILED
    }

} // namespace utils

// typically, each R process (inside Jupyter) only takes up about ~9% of the CPU, so this could absolutely benefit from paralellization
// wait 5 seconds between launching new processes, so we don't run into (possible???) file I/O issues inside the R instances

// the issue is that when the R interpreter gets called, the expression gets passed with all the quotes stripped away - leads to syntax errors
// figure out why the quotes get stripped away and how to preserve them when they are loaded into the R interpreter
// TURNS OUT THAT THE EXPRESSION ARGUMENT (-e) MUST BE ENCLOSED IN DOUBLE QUOTES NOT SINGLE QUOTES!!

// R also seems to skip the assertion like expressions e.g. stopifnot() and the likes when non-interactively invoked with expressions (using -e)?????

int wmain(_In_ [[maybe_unused]] int argc, [[maybe_unused]] _In_ wchar_t* argv[]) {
    ::atexit(::__release_ntdbsdll); // to release the Ntdsbmsg.DLL at the parent process exit

    // https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex
    // SYSTEM_INFO sysinf {};
    // ::GetSystemInfo(&sysinf);
    // sysinf.dwNumberOfProcessors - this machine has 18 cores, which is quite suprising

    // for ::WaitForMultipleObjects, we need an array of active process handles
    std::vector<HANDLE64> active_process_handles {}, active_thread_handles {}; // NOLINT(readability-isolate-declaration)
    unsigned long         wfmo_result {}; // the offset of the signalled handle in active_process_handles will be this value - WAIT_OBJECT_0

    unsigned long long nsucceeded_launches {};
    std::wstring       rscript {}, cmdline {}; // NOLINT(readability-isolate-declaration)
    rscript.resize(RSCRIPT_BUFFSIZE);
    cmdline.resize(CMDLINE_BUFFSIZE);
    bool is_loop_broken_prematurely {};

    for (unsigned dmod = 0; dmod < 3; ++dmod) {     // discrete models
        for (unsigned cmod = 0; cmod < 4; ++cmod) { // continuous models
            for (unsigned nm = 0; nm < 2; ++nm) {   // null model (0, 1) i.e true or false

                // THIS TWO STRUCTS ARE INTENTIONALLY FRESHLY CREATED IN EVERY ITERATION!!!!
                STARTUPINFOW        starupinfo { .cb          = sizeof(STARTUPINFOW),
                                                 .dwFlags     = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES | STARTF_FORCEONFEEDBACK,
                                                 .wShowWindow = SW_HIDE }; // DON'T WANT TO SEE 9 INTERPRETER SESSIONS ON SCREEN
                PROCESS_INFORMATION procinfo {};

                houwie::generate_rscript(
                    rscript, // the launch directory of this programme will have all the needed files
                    LR"(./FRED_subset_collab_1005sp.tre)",
                    LR"(./genus_state_rec_logged_species_avgd_RD_1005sp.csv)",
                    static_cast<houwie::DISCRETE_MODELS>(dmod),
                    static_cast<houwie::CONTINUOUS_MODELS>(cmod),
                    LR"(../rdata/parallel/)",
                    L"RD",
                    L"MYCO",
                    L"1005SP",
                    nm,
                    30
                );

                ::memset(cmdline.data(), 0, cmdline.size() * sizeof(wchar_t));
                // the double quotation marks enclosing the expression (-e) argument are absolutely critical
                ::swprintf_s(cmdline.data(), cmdline.size(), L"%s --no-save -e \"%s\"", R_INTERPRETER_PATH, rscript.c_str());
                // ::_putws(cmdline.c_str());

                // if we are at (or above) capacity, halt the launch of new processes and wait for one to finish before laucning a new one
                if (active_process_handles.size() >= MAX_PARALLEL_PROCESSES) {
                    // https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitformultipleobjects
                    // https://learn.microsoft.com/en-us/windows/win32/sync/waiting-for-multiple-objects
                    wfmo_result = ::WaitForMultipleObjects( // make sure that the return value is valid
                        active_process_handles.size(),
                        active_process_handles.data(),
                        false, // return when at least one process is signalled
                        INFINITE // INFINITE milliseconds is about 1193 hours, so we can just use that
                    );

                    // if we are at capacity and wait failed, break out the loop and focus on the already active processes
                    if (!utils::handle_parallel_waits(wfmo_result, active_process_handles, active_thread_handles)) {
                        is_loop_broken_prematurely = true;
                        break; // no more new process launches
                    }
                }

                //------------------------------------------------------------------------
                // EVERYTHING BELOW WILL ONLY BE EXECUTED WHEN WE ARE BELOW CAPACITY
                //------------------------------------------------------------------------

                // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw
                // https://learn.microsoft.com/en-us/windows/win32/procthread/creating-processes
                if (!::CreateProcessW(
                        R_INTERPRETER_PATH, // DO NOT LEAVE THIS EMPTY!!! i.e. nullptr
                        cmdline.data(),
                        nullptr,
                        nullptr,
                        TRUE,
                        HIGH_PRIORITY_CLASS | CREATE_NEW_CONSOLE,
                        // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getpriorityclass
                        // look up the above for process priorities and scheduling
                        nullptr,
                        nullptr,
                        &starupinfo,
                        &procinfo
                    )) {
                    ::fwprintf_s( // log where the launch failed
                        stderr,
                        L"Failed to launch %s-%s-%s fit, %s error in call to CreateProcessW!\n",
                        houwie::__discmod_to_wstr(static_cast<houwie::DISCRETE_MODELS>(dmod)),
                        houwie::__contmod_to_wstr(static_cast<houwie::CONTINUOUS_MODELS>(cmod)),
                        nm ? L"CID" : L"CD",
                        utils::error_code_to_wstring(::GetLastError())
                    );
                    continue; // move on to the next launch
                }

                // if the lauch succeeded,
                nsucceeded_launches++;
                // update active process and thread handles
                active_process_handles.push_back(procinfo.hProcess);
                active_thread_handles.push_back(procinfo.hThread);
            }
        }
    }

    //--------------------------------------
    // ONCE WE HAVE EXITED THE LOOP
    //--------------------------------------

    if (is_loop_broken_prematurely)
        ::fwprintf_s(
            stderr, L"Process launch terminated prematurely, %llu active processes running at termination!\n", nsucceeded_launches
        );

    // return only when all the processs are signalled (ON PREMATURE LOOP BREAK OR SUCCESSFUL COMPLETION)
    wfmo_result = ::WaitForMultipleObjects(active_process_handles.size(), active_process_handles.data(), true, INFINITE);

    // https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitformultipleobjects
    if (wfmo_result < WAIT_OBJECT_0 + active_process_handles.size())
        ::fputws(L"All the processes have signalled successfully!\n", stderr);
    else if (wfmo_result < WAIT_ABANDONED_0 + active_process_handles.size())
        ::fputws(L"All the processes have signalled with at least one abandoned mutex!\n", stderr);

    // close all the leftover process handles and thread handles
    std::for_each(active_process_handles.begin(), active_process_handles.end(), ::CloseHandle);
    std::for_each(active_thread_handles.begin(), active_thread_handles.end(), ::CloseHandle);

    ::wprintf_s(L"Done, %llu out of %llu launches succeeded!\n", nsucceeded_launches, TOTAL_PROCESSES);

    return EXIT_SUCCESS;
}

// NOLINTEND(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)
