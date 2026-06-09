// this thing has been a lifesaver :)

#if !(defined(_WIN32) || defined(_WIN64)) && !(defined(_MSC_VER) || defined(_MSC_FULL_VER))
    #error This is a Windows only implementation that liberally uses the Win32 API, not meant to be used on other platforms!.
#endif

// clang .\parhouwie.cpp -Wall -Wextra -Wpedantic -static -march=native -DNDEBUG -D_NDEBUG -O3 -std=c++20 -o .\parhouwie.exe
// cl .\parhouwie.cpp /Wall /std:c++20 /O2 /MT /EHsc /DNDEBUG /D_NDEBUG

#if defined(_MSC_FULL_VER) && !defined(__llvm__) // MSVC specific warnings
    #pragma warning(disable : 4267 4710 4711 4774 4800 4820)
#endif

#ifdef __llvm__

    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wmicrosoft-string-literal-from-predefined"
    #pragma clang diagnostic ignored "-Wunused-function"
    #pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"

#endif

// clang-format off
#define _AMD64_ // architecture
#define WIN32_LEAN_AND_MEAN
#include <errhandlingapi.h>
#include <libloaderapi.h>
#include <processthreadsapi.h>
#include <profileapi.h>
#include <Shlwapi.h>
#include <synchapi.h>
#include <sysinfoapi.h>
#include <WinDef.h>
#include <WinBase.h>
#include <WinUser.h>
// clang-format on

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#pragma comment(lib, "Shlwapi.lib") // for ::PathFileExistsW

namespace fred {
    [[maybe_unused]] static constexpr wchar_t ROOT_DIAMETER[] { L"F00679" };
    [[maybe_unused]] static constexpr wchar_t SPECIFIC_ROOT_LENGTH[] { L"F00727" };
    [[maybe_unused]] static constexpr wchar_t ROOT_TISSUE_DENSITY[] { L"F00709" };
}

namespace paths {
    static constexpr wchar_t RINTERPRETER[] { L"C:/Program Files/R/R-4.6.0/bin/R.exe" }; // the install directory of the R.exe binary
    // al the below are relative paths, used assuming the executable will be launched from this directory
    static constexpr wchar_t PHYLOGENY[] { L"./../../data/chapter2/uphylomaker/FRED4_1301_species.tre" };
    static constexpr wchar_t TRAIT_DATA[] { L"./../../data/chapter2/FRED/subsets/final.csv" };
    static constexpr wchar_t SAVE_RDS[] { L"./../../data/chapter2/rdata/parallel/srl_1301_100sims/" }; // must end with a foward slash
}

// pick a decent number with enough CPU space for other essential processes - uni laptop has 14 cores and 18 logical processors
static constexpr unsigned long long NPARALLEL_PROCESSES { 0xC }; // with 12 the CPU gets very close maxxing out
static constexpr unsigned long long NTOTAL_PROCESSES { 0x18 };   // 4 continuous models x 3 discrete models x 2 rate categories

static constexpr unsigned long long RSCRIPT_BUFFSIZE { 0x4F0 };
static constexpr unsigned long long CMDLINE_BUFFSIZE { 0x6F0 }; // being a bit too generous here
static_assert(RSCRIPT_BUFFSIZE < CMDLINE_BUFFSIZE);

static HINSTANCE handle_ntdsbmsg {}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables) handle to Ntdsbmsg.dll

namespace utils {

    extern "C" inline void __cdecl release_ntdbsdll() noexcept {
        if (handle_ntdsbmsg) ::FreeLibrary(handle_ntdsbmsg);
    }

    // get the string representation of a _WIN32 error code

    [[nodiscard, clang::always_inline]] static inline const wchar_t* __stdcall error_code_to_wstring(_In_ const unsigned long& errcode) noexcept {
        static constexpr unsigned long long ERRORMSG_BUFFSIZE { 0x2EE };     // length of the error message buffer in number of wchar_t s
        static wchar_t                      buffer[ERRORMSG_BUFFSIZE] { 0 }; // needs to be in static memory for returning
        // without this the previously written buffer can get partially overwritten and returned in subsequent function invocations
        ::memset(buffer, 0, sizeof(buffer));

        // https://devblogs.microsoft.com/oldnewthing/20191025-00/?p=103025
        // https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-formatmessage
        unsigned long nbyteswritten =
            ::FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK, nullptr, errcode, 0, buffer, ERRORMSG_BUFFSIZE, nullptr);

        if (!nbyteswritten) { // will be 0 if the call above to FormatMessageW failed; if that, the error string is not found in the system, try Ntdsbmsg.dll
            // if the library hasn't already been loaded by previous calls to this function
            if (!handle_ntdsbmsg) handle_ntdsbmsg = ::LoadLibraryW(L"Ntdsbmsg.DLL");
            if (!handle_ntdsbmsg) { // will be NULL if the DLL failed to load
                ::fputws(L"Failed to load Ntdsbmsg.DLL", stderr);
                return buffer; // must be an empty buffer here
            }

            nbyteswritten = ::FormatMessageW(
                FORMAT_MESSAGE_FROM_HMODULE | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK, handle_ntdsbmsg, errcode, 0, buffer, ERRORMSG_BUFFSIZE, nullptr
            );
            // ::FreeLibrary(handle_ntdsbmsg); // detach the DLL from the process - atexit() will handle this for us
        }
        return buffer;
    }

    // will only return true when the wait is signalled success
    [[nodiscard]] static inline bool __stdcall handle_parallel_waits(
        _Inout_ std::vector<HANDLE64>& phandles, _Inout_ std::vector<HANDLE64>& thandles, _In_ const bool& all, _In_ const unsigned long& duration
    ) noexcept {
        // made the function more customizeable
        // WAIT_OBJECT_0 is defined as 0 and WAIT_ABANDONED_0 is defined as 0x00000080L
        // so the returned value is practically equal to the offset of the signalled handle WHEN WAIT SUCCEEDS
        unsigned long handle_offset { 0xFF };
        bool          exitval {};
        // https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitformultipleobjects
        // https://learn.microsoft.com/en-us/windows/win32/sync/waiting-for-multiple-objects
        const unsigned long waitstatus =        ::WaitForMultipleObjects( // make sure that the return value is valid
                        phandles.size(),
                        phandles.data(),
                        all, // return when at least one process is signalled
                        duration // INFINITE milliseconds is about 1193 hours, so we can just use that
                    );

        // WAIT_FAILED or WAIT_TIMEOUT
        switch (waitstatus) {
            case WAIT_FAILED :
                ::fwprintf_s(stderr, L"WaitForMultipleObjects signalled WAIT_FAILED, %s\n", error_code_to_wstring(::GetLastError()));
                if (all) goto CLOSE_ACTIVE_HANDLES_AND_EXIT; // if the wait was for all processes, close all the handles and return false
                return false;                                // else just return false

            case WAIT_TIMEOUT :
                ::fputws(L"WaitForMultipleObjects signalled WAIT_TIMEOUT\n", stderr);
                if (all) goto CLOSE_ACTIVE_HANDLES_AND_EXIT;
                return false;

            default : break;
        }

        // WAIT_OBJECT_0 to (WAIT_OBJECT_0 + nCount - 1)
        if ( // (waitstatus >= WAIT_OBJECT_0) && // will always be true because the return value of ::WaitForMultipleObjects() is unsigned
            waitstatus < (WAIT_OBJECT_0 + phandles.size())
        ) {
            exitval = true;
            if (all) goto CLOSE_ACTIVE_HANDLES_AND_EXIT;
            // bWaitAll == FALSE
            handle_offset = waitstatus - WAIT_OBJECT_0;
            goto CLOSE_SELECTED_HANDLE_AND_EXIT;
        }

        // WAIT_ABANDONED_0 to (WAIT_ABANDONED_0 + nCount - 1)
        else if ((waitstatus >= WAIT_ABANDONED_0) && (waitstatus < (WAIT_ABANDONED_0 + phandles.size()))) {
            if (all) {
                ::fputws(L"WaitForMultipleObjects signalled WAIT_ABANDONED with bWaitAll set to TRUE, 1 or more probable abandoned mutexes!\n", stderr);
                goto CLOSE_ACTIVE_HANDLES_AND_EXIT;
            }
            // bWaitAll == FALSE
            ::fputws(L"WaitForMultipleObjects signalled WAIT_ABANDONED\n", stderr);
            handle_offset = waitstatus - WAIT_ABANDONED_0;
            goto CLOSE_SELECTED_HANDLE_AND_EXIT;
        }

CLOSE_ACTIVE_HANDLES_AND_EXIT:                                // close all the leftover process handles and thread handles
        for (unsigned long i = 0; i < phandles.size(); ++i) { // taking it for granted that phandles.size()==thandles.size()
            // if (!::GetExitCodeProcess(phandles.at(i), &exitcode)) // 0 is failed
            //     ::fwprintf_s(stderr, L"GetExitCodeProcess returned 0, %s\n", __error_code_to_wstring(::GetLastError()));
            // else
            //     excodes.push_back(exitcode);
            ::CloseHandle(phandles.at(i));
            ::CloseHandle(thandles.at(i));
        }

        phandles.clear();
        thandles.clear();
        return exitval;

CLOSE_SELECTED_HANDLE_AND_EXIT:
        // capture the exit code
        // if (!::GetExitCodeProcess(*(phandles.data() + handle_offset), &exitcode))
        //     ::fwprintf_s(stderr, L"GetExitCodeProcess returned 0, %s\n", __error_code_to_wstring(::GetLastError()));
        // else
        //     excodes.push_back(exitcode);
        ::CloseHandle(*(phandles.data() + handle_offset));
        phandles.erase(phandles.begin() + handle_offset); // remove the signalled process
        ::CloseHandle(*(thandles.data() + handle_offset));
        thandles.erase(thandles.begin() + handle_offset); // remove the signalled thread handle
        return exitval;
    }

} // namespace utils

namespace houwie {
    enum class DISCRETE_MODEL : unsigned char {
        ER,  // all rates are identical
        SYM, // symmetrically identical rates
        ARD  // all rates are allowed to be different (asymmetrically)
    };

    enum class CONTINUOUS_MODEL : unsigned char {
        OUM,  // only the continuous trait optimum varies depending on the discrete state regimes
        OUMA, // the continuous trait optimum and the pull towards the optimum vary depending on the discrete state regimes
        OUMV, // the continuous trait optimum and the rate of continuous trait evolution vary depending on the discrete state regimes
        OUMVA // optima, rate of evolution and the pull towards the optima of the continuous trait vary depending on the discrete state regimes
    };

    [[clang::always_inline, nodiscard]] static inline constexpr const wchar_t* __stdcall dmod_tostr(_In_ const DISCRETE_MODEL& model) noexcept {
        switch (model) {
            case DISCRETE_MODEL::ER  : return L"ER";
            case DISCRETE_MODEL::SYM : return L"SYM";
            case DISCRETE_MODEL::ARD : return L"ARD";
        }
        // MSVC bitches about "not all control paths return a value"
    }

    [[clang::always_inline, nodiscard]] static inline constexpr const wchar_t* __stdcall cmod_tostr(_In_ const CONTINUOUS_MODEL& model) noexcept {
        switch (model) {
            case CONTINUOUS_MODEL::OUM   : return L"OUM";
            case CONTINUOUS_MODEL::OUMA  : return L"OUMA";
            case CONTINUOUS_MODEL::OUMV  : return L"OUMV";
            case CONTINUOUS_MODEL::OUMVA : return L"OUMVA";
        }
        // MSVC bitches about "not all control paths return a value"
    }

    [[clang::always_inline, nodiscard]] static inline const wchar_t* __stdcall rds_path(
        _In_ const DISCRETE_MODEL&   dmodel,
        _In_ const CONTINUOUS_MODEL& cmodel,
        _In_ const wchar_t* const    contrait, // e.g. F00727
        _In_ const wchar_t* const    savedir,  // assumed ends with a forward slash, e.g. "C:/Users/Documents/"
        _In_ const bool&             nullmodel,
        _In_ const wchar_t* const    suffix // e.g. _1006sp
    ) noexcept {
        static constexpr unsigned long long SAVERDS_NAME_LENGTH { MAX_PATH };
        static wchar_t                      buffer[SAVERDS_NAME_LENGTH] {};
        ::memset(buffer, 0, sizeof(buffer)); // we don't want buffer contents from previous writes intefereing with new writes

        if (::wcsrchr(savedir, L'/') != (savedir + ::wcslen(savedir) - 1)) { // exit if savedir does not end with a foward slash
                                                                             // ::fputws(::wcsrchr(savedir, L'/'), stderr);
            // ::fwprintf_s(stderr, L"%s does not end with a forward slash but ends with %d!\n", savedir, *(savedir + ::wcslen(savedir)));
            ::fputws(L"Invalid argument savedir in call to " __FUNCTIONW__ "; it must end with a foward slash!\n", stderr);
            return nullptr;
        }

        if (suffix) // e.g. C:/Users/Documents/ARDOUMV_F00679_CD_395sp.Rds
            ::swprintf_s(
                buffer,
                SAVERDS_NAME_LENGTH,
                L"%s%s%s_%s_%s_%s.Rds", // we expect the directory path to end with a forward slash
                savedir,
                dmod_tostr(dmodel),
                cmod_tostr(cmodel),
                contrait,
                nullmodel ? L"CID" : L"CD",
                suffix
            );
        else // e.g. C:/Users/Documents/ARDOUMV_F00679_CD.Rds
            ::swprintf_s(buffer, SAVERDS_NAME_LENGTH, L"%s%s%s_%s_%s.Rds", savedir, dmod_tostr(dmodel), cmod_tostr(cmodel), contrait, nullmodel ? L"CID" : L"CD");

        return buffer;
    }

    [[clang::always_inline]] static inline bool __stdcall generate_rscript(
        _Inout_ std::wstring&          buffer,
        _In_ const wchar_t* const      phylogeny,
        _In_ const wchar_t* const      traitdata,
        _In_ const wchar_t* const      contrait,
        _In_ const DISCRETE_MODEL&     dmodel,
        _In_ const CONTINUOUS_MODEL&   cmodel,
        _In_ const wchar_t* const      savedir,
        _In_ const wchar_t* const      suffix,
        _In_ const bool&               nullmodel,
        _In_ const unsigned long long& nsims
    ) noexcept {
        // make sure that all the paths are valid, using separate conditional for detailed error reporting
        if (!::PathFileExistsW(savedir)) {
            ::fwprintf_s(stderr, L"Invalid argument savedir in call to " __FUNCTIONW__ ". %s\n", utils::error_code_to_wstring(::GetLastError()));
            return false;
        }

        if (!::PathFileExistsW(phylogeny)) {
            ::fwprintf_s(stderr, L"Invalid argument phylogeny in call to " __FUNCTIONW__ ". %s\n", utils::error_code_to_wstring(::GetLastError()));
            return false;
        }

        if (!::PathFileExistsW(traitdata)) {
            ::fwprintf_s(stderr, L"Invalid argument traitdata in call to " __FUNCTIONW__ ". %s\n", utils::error_code_to_wstring(::GetLastError()));
            return false;
        }

        if (buffer.size() < RSCRIPT_BUFFSIZE) buffer.resize(RSCRIPT_BUFFSIZE);
        ::memset(buffer.data(), 0, buffer.size() * sizeof(wchar_t)); // clean up the buffer before every new write
        const wchar_t* const rds_savepath = rds_path(dmodel, cmodel, contrait, savedir, nullmodel, suffix);
        if (!rds_savepath) return false;

        ::swprintf_s(
            buffer.data(),
            buffer.size(),
            // who gives a damn when warnings are emiited during package loading in automation
            // also using ; instead of new lines to delineate expressions (expressions separated by \n s did not work for some reason???)
            // and when passed as expressions, all the double quotes get stripped away for some reason?????, using single quotes instead for string literals
            L"library('ape');"
            L"library('OUwie');"
            L"set.seed(1, kind = 'Mersenne-Twister');" // make sure reruns don't give us inconsistent results, don't know how useful this is
            L"phylogeny <- ape::read.tree('%s');"
            L"data <- read.csv('%s')[, c('binominal', 'state', '%s')];" // only consider the columns we need - binominal name, mycorrhizal state and the continuous trait of choice
            L"stopifnot(all(phylogeny$tip.label == data$binominal));"
            L"model <- OUwie::hOUwie(phy = phylogeny, data = data, rate.cat = %1u, discrete_model = '%s', continuous_model = '%s', nSim = %llu, null.model = %s);"
            L"saveRDS(object = model, file = '%s');",
            phylogeny,
            traitdata,
            contrait,
            // if null_model is true, then it's a CID model with 2 rate categories, else it's a CD model with just 1 rate category
            nullmodel ? 2U : 1U,
            dmod_tostr(dmodel),
            cmod_tostr(cmodel),
            nsims,
            nullmodel ? L"TRUE" : L"FALSE",
            rds_savepath
        );

        return true;
    }

} // namespace houwie

// typically, each R process (inside Jupyter) only takes up about ~9% of the CPU, so this could absolutely benefit from paralellization
// the issue is that when the R interpreter gets called, the expression gets passed with all the quotes stripped away - leads to syntax errors
// figure out why the quotes get stripped away and how to preserve them when they are loaded into the R interpreter
// TURNS OUT THAT THE EXPRESSION ARGUMENT (-e) MUST BE ENCLOSED IN DOUBLE QUOTES NOT SINGLE QUOTES!!

int wmain() {
    ::atexit(utils::release_ntdbsdll); // to release the Ntdsbmsg.DLL at the parent process exit

    // https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex
    // SYSTEM_INFO sysinf {};
    // ::GetSystemInfo(&sysinf);
    // sysinf.dwNumberOfProcessors - this machine has 18 cores, (don't know how many P cores and E cores, through?????)

    // for ::WaitForMultipleObjects, we need an array of active process handles
    std::vector<HANDLE64> active_process_handles {}, active_thread_handles {};
    long                  exitcode { EXIT_SUCCESS };

    unsigned long long nsucceeded_launches {};
    std::wstring       rscript {}, cmdline {};

    rscript.resize(RSCRIPT_BUFFSIZE);
    cmdline.resize(CMDLINE_BUFFSIZE);
    bool is_broken_prematurely {};

    // timing runtime
    // https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps
    LARGE_INTEGER start {}, stop {}, freq {};
    ::QueryPerformanceFrequency(&freq); // number of ticks per second
    ::QueryPerformanceCounter(&start);

    for (unsigned dmod = 0; dmod < 3; ++dmod) {     // discrete models
        for (unsigned cmod = 0; cmod < 4; ++cmod) { // continuous models
            for (unsigned nm = 0; nm < 2; ++nm) {   // null model (0, 1) i.e true or false

                // THESE TWO STRUCTS ARE INTENTIONALLY PLACED HERE TO BE FRESHLY CREATED IN EVERY ITERATION!!!!
                STARTUPINFOW        starupinfo { .cb          = sizeof(STARTUPINFOW),
                                                 .dwFlags     = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES | STARTF_FORCEONFEEDBACK,
                                                 .wShowWindow = SW_HIDE }; // whether or not to display the terminal for the launched processes
                PROCESS_INFORMATION procinfo {};

                if (!houwie::generate_rscript(
                        rscript, // the launch directory of this programme will have all the needed files
                        paths::PHYLOGENY,
                        paths::TRAIT_DATA,
                        fred::SPECIFIC_ROOT_LENGTH,
                        static_cast<houwie::DISCRETE_MODEL>(dmod),
                        static_cast<houwie::CONTINUOUS_MODEL>(cmod),
                        paths::SAVE_RDS,
                        nullptr,
                        nm,
                        100
                    ))
                    ::exit(EXIT_FAILURE);

                ::memset(cmdline.data(), 0, cmdline.size() * sizeof(wchar_t));

                // the double quotation marks enclosing the expression (-e) argument are absolutely critical
                ::swprintf_s(cmdline.data(), cmdline.size(), L"%s --no-save -e \"%s\"", paths::RINTERPRETER, rscript.c_str());
                // ::_putws(cmdline.c_str());

                // if we are at (or above) capacity, halt the launch of new processes and wait for one to finish before laucning a new one
                if (active_process_handles.size() >= NPARALLEL_PROCESSES) {
                    // if we are at capacity and wait failed, break out the loop and focus on the already active processes
                    if (!utils::handle_parallel_waits(active_process_handles, active_thread_handles, false, INFINITE)) {
                        // can happen when WAIT_FAILED, WAIT_ABANDONED_0 or WAIT_TIMEOUT
                        is_broken_prematurely = true;
                        break; // no more new process launches
                    }
                }

                //------------------------------------------------------------------------
                // EVERYTHING BELOW WILL ONLY BE EXECUTED WHEN WE ARE BELOW CAPACITY
                //------------------------------------------------------------------------

                // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw
                // https://learn.microsoft.com/en-us/windows/win32/procthread/creating-processes
                if (!::CreateProcessW(
                        paths::RINTERPRETER, // DO NOT LEAVE THIS EMPTY!!! i.e. nullptr
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
                        L"Failed to launch %s-%s-%s fit, Error in call to ::CreateProcessW: %s\n",
                        houwie::dmod_tostr(static_cast<houwie::DISCRETE_MODEL>(dmod)),
                        houwie::cmod_tostr(static_cast<houwie::CONTINUOUS_MODEL>(cmod)),
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

    if (is_broken_prematurely) {
        exitcode = EXIT_FAILURE;
        ::fwprintf_s(stderr, L"Process launch terminated prematurely, %llu active processes running at termination!\n", nsucceeded_launches);
    }

    // return only when all the processs are signalled (ON PREMATURE LOOP BREAK OR SUCCESSFUL COMPLETION)
    if (!utils::handle_parallel_waits(active_process_handles, active_thread_handles, true, INFINITE)) exitcode = EXIT_FAILURE;

    ::QueryPerformanceCounter(&stop);

    ::wprintf_s(
        L"Done, %llu out of %llu launches completed within %.3Lf hours!\n",
        nsucceeded_launches,
        NTOTAL_PROCESSES,
        (stop.QuadPart - start.QuadPart) / (freq.QuadPart * 3600.00L) // NOLINT(cppcoreguidelines-narrowing-conversions)
    );

    return exitcode;
}

#ifdef __llvm__
    #pragma clang diagnostic pop
#endif
