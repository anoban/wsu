#if !defined(_WIN32) && !defined(_WIN64) && (!defined(_MSC_VER) || !defined(_MSC_FULL_VER))
    #error This is a Windows only implementation, not meant to be used on other platforms!.
#endif

// clang .\parhouwie.c -Wall -Wextra -static -march=native -DNDEBUG -O3 -std=c++20
// launch the R interpretor in parallel for the hOUwie model fits

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

#include <array>
#include <cassert>
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
static constexpr unsigned long long MAX_PARALLEL_PROCESSES { 9 }; // half the number of cores

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)

namespace houwie {

    enum class DISCRETE_MODELS : unsigned char {
        ER,  // all rates are identical
        SYM, // symmetrically identical rates
        ARD  // all rates are allowed to be different (asymmetrically)
    };

    enum class CONTINUOUS_MODELS : unsigned char {
        OUM,
        OUMA [[deprecated(HOUWIE_VARIABLE_ALPHA_WARNING)]],
        OUMV,
        OUMVA [[deprecated(HOUWIE_VARIABLE_ALPHA_WARNING)]]
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
        ::memset(buffer, 0, sizeof(buffer));
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
        ::memset(buffer.data(), 0, buffer.size() * sizeof(wchar_t));

        ::swprintf_s(
            buffer.data(),
            buffer.size(),
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
    }
} // namespace houwie

namespace utils {

    // get the string representation of a _WIN32 error code
    // NOLINTNEXTLINE(readability-redundant-inline-specifier)
    [[nodiscard]] static inline const wchar_t* __stdcall error_code_to_string(_In_ const unsigned long& errcode) noexcept {
        static wchar_t errmsgbuffer[ERROR_MSG_BUFFSIZE] = { 0 }; // needs to be in static memory
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

    static inline bool handle_parallel_wait_signals( // NOLINT(readability-redundant-inline-specifier)
        _In_ const unsigned long& retval, _Inout_ unsigned long long& nactiveprocs, _Inout_ std::vector<HANDLE64>& active_proc_handles
    ) noexcept {
        // WAIT_OBJECT_0 is defined as 0 and WAIT_ABANDONED_0 is defined as 0x00000080L
        // so retval is practically equal to the offset of the signalled handle WHEN WAIT SUCCEEDS
        unsigned long pop_offset {};
        if ((retval < nactiveprocs) && (retval >= WAIT_OBJECT_0)) { // range WAIT_OBJECT_0 to (WAIT_OBJECT_0 + nCount - 1) indicates success
            pop_offset = retval - WAIT_OBJECT_0;
            active_proc_handles.erase(active_proc_handles.begin() + pop_offset); // remove the signalled handle
            nactiveprocs--;                                                      // no matter what, we have one less active process now
        } else if ((retval >= WAIT_ABANDONED_0) && (retval < WAIT_TIMEOUT)) { // range WAIT_ABANDONED_0 to (WAIT_ABANDONED_0 + nCount - 1)
            pop_offset = retval - WAIT_ABANDONED_0;
            active_proc_handles.erase(active_proc_handles.begin() + pop_offset);
        } else if (retval == WAIT_TIMEOUT) { // cannot (should not) happen as we specified the time limit to be INFINITE
            //
            ::fputws(L"WaitForMultipleObjects signalled WAIT_TIMEOUT", stderr);
        } else if (retval == WAIT_FAILED) {
            //
            ::fwprintf_s(stderr, L"WaitForMultipleObjects signalled WAIT_FAILED, %s\n", error_code_to_string(::GetLastError()));
        }
    }

} // namespace utils

// typically, each R process (inside Jupyter) only takes up about ~9% of the CPU, so this could absolutely benefit from paralellization
// wait 5 seconds between launching new processes, so we don't run into (possible???) file I/O issues inside the R instances

// the issue is that when the R interpreter gets called, the expression gets passed with all the quotes stripped away - leads to syntax errors
// figure out why the quotes get stripped away and how to preserve them when they are loaded into the R interpreter
// TURNS OUT THAT THE EXPRESSION ARGUMENT (-e) MUST BE ENCLOSED IN DOUBLE QUOTES NOT SINGLE QUOTES!!

// R also seems to skip the assertion like expressions e.g. stopifnot() and the likes when non-interactively invoked with expressions (using -e)?????

int wmain(_In_ [[maybe_unused]] int argc, [[maybe_unused]] _In_ wchar_t* argv[]) {
    // https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex
    // SYSTEM_INFO sysinf {};
    // ::GetSystemInfo(&sysinf);
    // sysinf.dwNumberOfProcessors - this machine has 18 cores, which is quite suprising

    switch (::WaitForSingleObject(childprocinfo.hProcess, INFINITE)) { // wait for the child process to finish
        case WAIT_ABANDONED :
            ::fputws(L"Mutex object was not released by the child thread before the caller thread terminated.\n", stderr);
            break;
        case WAIT_TIMEOUT  : ::fputws(L"The time-out interval has elapsed, and the object's state is nonsignaled.\n", stderr); break;
        case WAIT_FAILED   : ::fwprintf_s(stderr, L"Error %lu: Wait failed.\n", ::GetLastError()); break;
        case WAIT_OBJECT_0 : ::_putws(L"Wait success!"); break; // The state of the specified object is signaled, wait success
        default            : break;
    }

    // since this is not used to close handles, we can just resuse the same struct, who cares
    STARTUPINFOW                                     childstarupinfo = { .cb = sizeof(STARTUPINFOW),
                                                                         .dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES | STARTF_FORCEONFEEDBACK,
                                                                         .wShowWindow = SW_SHOW };
    std::array<PROCESS_INFORMATION, TOTAL_PROCESSES> procinfos {}; // for all the lauched processes
    // unfortunately for ::WaitForMultipleObjects, we need an array of active process handles, cannot index into the above struct to access the process handles
    std::vector<HANDLE64>                            active_process_handles {};
    active_process_handles.reserve(TOTAL_PROCESSES * 2); // x 2 just because
    unsigned long multobjreturn {}; // the offset of the signalled handle in active_process_handles will be this value - WAIT_OBJECT_0

    unsigned long long nactive_processes {}, nsucceeded_launches {}; // NOLINT(readability-isolate-declaration)
    std::wstring       rscript {}, cmdline {};                       // NOLINT(readability-isolate-declaration)
    rscript.resize(RSCRIPT_BUFFSIZE);
    cmdline.resize(CMDLINE_BUFFSIZE);

    for (unsigned dmod = 0; dmod < 3; ++dmod) {     // discrete models
        for (unsigned cmod = 0; cmod < 4; ++cmod) { // continuous models
            for (unsigned nm = 0; nm < 2; ++nm) {   // null model (0, 1) i.e true or false

                houwie::generate_rscript(
                    rscript, // the launch directory of this programme will have all the needed files
                    LR"(./FRED_subset_collab_1005sp.tre)",
                    LR"(./genus_state_rec_logged_species_avgd_SRL_1005sp.csv)",
                    static_cast<houwie::DISCRETE_MODELS>(dmod),
                    static_cast<houwie::CONTINUOUS_MODELS>(cmod),
                    LR"(../rdata/parallel/)",
                    L"logSRL",
                    L"state",
                    L"1005sp",
                    nm,
                    30
                );

                ::memset(cmdline.data(), 0, cmdline.size() * sizeof(wchar_t));
                // the double quotation marks enclosing the expression (-e) argument are absolutely critical
                ::swprintf_s(cmdline.data(), cmdline.size(), L"%s --no-save -e \"%s\"", R_INTERPRETER_PATH, rscript.c_str());
                // ::_putws(cmdline.c_str());

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
                        procinfos.data() + nsucceeded_launches // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                        // the procinfos array might contain invalid or empty or uninitialized structs where launches failed!!!!!!
                    )) {
                    ::fwprintf_s( // log where the launch failed
                        stderr,
                        L"Failed to launch %s-%s-%s fit, %s error in call to CreateProcessW!\n",
                        houwie::__discmod_to_wstr(static_cast<houwie::DISCRETE_MODELS>(dmod)),
                        houwie::__contmod_to_wstr(static_cast<houwie::CONTINUOUS_MODELS>(cmod)),
                        nm ? L"CID" : L"CD",
                        utils::error_code_to_string(::GetLastError())
                    );
                    continue; // move on to the next launch
                }

                // if the lauch succeeded,
                nsucceeded_launches++;
                nactive_processes++;
                // record the process handle
                active_process_handles.push_back(procinfos[nsucceeded_launches].hProcess);

                // if we are at capacity, halt the launch of new processes and wait for one to finish before laucning a new one
                if (nactive_processes == MAX_PARALLEL_PROCESSES) {
                    // https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitformultipleobjects
                    // INFINITE milliseconds is about 1193 hours, so we can just use that
                    // https://learn.microsoft.com/en-us/windows/win32/sync/waiting-for-multiple-objects
                    multobjreturn = ::WaitForMultipleObjects(
                        nactive_processes,
                        active_process_handles.data(),
                        false, // return when at least one process is signalled
                        INFINITE
                    );
                    // make sure that the return value is valid
                    utils::handle_parallel_wait_signals(multobjreturn, nactive_processes, active_process_handles);
                    // https://learn.microsoft.com/en-us/windows/win32/procthread/creating-processes
                    // close the signalled process' handles
                }
            }
        }
    }

    ::wprintf_s(L"%llu out of %llu launches succeeded!", nsucceeded_launches, TOTAL_PROCESSES);

    return EXIT_SUCCESS;
}

// NOLINTEND(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)
