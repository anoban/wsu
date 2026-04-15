#if !(defined(_WIN32) || defined(_WIN64)) && !(defined(_MSC_VER) || defined(_MSC_FULL_VER))
    #error This is a Windows only implementation that liberally uses the Win32 API, not meant to be used on other platforms!.
#endif

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
#include <ctime>
#include <string>
#include <vector>

static constexpr unsigned long long       NPARALLEL_PROCESSES { 0xA }, NTOTAL_PROCESSES { 0x1E };
static constexpr unsigned long long       CMDLINE_BUFFSIZE { 0xFF };
[[maybe_unused]] static constexpr wchar_t EXECUTABLE_PATH[] { LR"(C:\Program Files\LLVM\bin\clang.exe)" };
[[maybe_unused]] static constexpr wchar_t PYTHON_FULLPATH[] { LR"(C:\Program Files\Python314\python.exe)" };

#define PATH LR"(./dummy.exe)"

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
        _Inout_ std::vector<HANDLE64>& phandles,
        _Inout_ std::vector<HANDLE64>& thandles,
        _Inout_ std::vector<unsigned long>& exitcodes,
        _In_ const bool&                    all,
        _In_ const unsigned long&           duration
    ) noexcept {
        // made the function more customizeable
        // WAIT_OBJECT_0 is defined as 0 and WAIT_ABANDONED_0 is defined as 0x00000080L
        // so the returned value is practically equal to the offset of the signalled handle WHEN WAIT SUCCEEDS
        unsigned long handle_offset { 0xFF }, exitcode {};
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

CLOSE_ACTIVE_HANDLES_AND_EXIT:                                    // close all the leftover process handles and thread handles
        for (unsigned long i = 0; i < phandles.size(); ++i) {     // taking it for granted that phandles.size()==thandles.size()
            if (!::GetExitCodeProcess(phandles.at(i), &exitcode)) // 0 is failed
                ::fwprintf_s(stderr, L"GetExitCodeProcess returned 0, %s\n", error_code_to_wstring(::GetLastError()));
            else
                exitcodes.push_back(exitcode);
            ::CloseHandle(phandles.at(i));
            ::CloseHandle(thandles.at(i));
        }

        phandles.clear();
        thandles.clear();
        return exitval;

CLOSE_SELECTED_HANDLE_AND_EXIT:
        // capture the exit code
        if (!::GetExitCodeProcess(*(phandles.data() + handle_offset), &exitcode))
            ::fwprintf_s(stderr, L"GetExitCodeProcess returned 0, %s\n", error_code_to_wstring(::GetLastError()));
        else
            exitcodes.push_back(exitcode);
        ::CloseHandle(*(phandles.data() + handle_offset));
        phandles.erase(phandles.begin() + handle_offset); // remove the signalled process
        ::CloseHandle(*(thandles.data() + handle_offset));
        thandles.erase(thandles.begin() + handle_offset); // remove the signalled thread handle
        return exitval;
    }

} // namespace utils

int wmain(_In_ [[maybe_unused]] int argc, [[maybe_unused]] _In_ wchar_t* wargv[]) {
    ::atexit(utils::release_ntdbsdll); // to release the Ntdsbmsg.DLL at the parent process exit
    ::srand(::time(nullptr));

    // for ::WaitForMultipleObjects, we need an array of active process handles
    std::vector<HANDLE64>      active_process_handles {}, active_thread_handles {};
    std::vector<unsigned long> proc_exitcodes {};
    proc_exitcodes.reserve(NTOTAL_PROCESSES);

    long               exitcode { EXIT_SUCCESS };
    unsigned long long nsucceeded_launches {};

    std::wstring cmdline {};
    cmdline.resize(CMDLINE_BUFFSIZE);

    bool is_broken_prematurely {};

    for (unsigned i = 0; i < NTOTAL_PROCESSES; ++i) {
        // THESE TWO STRUCTS ARE INTENTIONALLY PLACED HERE TO BE FRESHLY CREATED IN EVERY ITERATION!!!!
        STARTUPINFOW        starupinfo { .cb          = sizeof(STARTUPINFOW),
                                         .dwFlags     = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES | STARTF_FORCEONFEEDBACK,
                                         .wShowWindow = SW_HIDE }; // whether or not to display the terminal for the launched processes
        PROCESS_INFORMATION procinfo {};

        ::memset(cmdline.data(), 0, cmdline.size() * sizeof(wchar_t));
        ::swprintf_s(cmdline.data(), cmdline.size(), L"%s --hello", PATH);

        // for python.exe
        // ::swprintf_s(cmdline.data(), cmdline.size(), L"%s -c \"import sys; sys.exit(%d)\"", PYTHON_FULLPATH, ::rand());

        // ::_putws(cmdline.c_str());
        // if we are at (or above) capacity, halt the launch of new processes and wait for one to finish before laucning a new one
        if (active_process_handles.size() >= NPARALLEL_PROCESSES) {
            // if we are at capacity and wait failed, break out the loop and focus on the already active processes
            if (!utils::handle_parallel_waits(active_process_handles, active_thread_handles, proc_exitcodes, false, INFINITE)) {
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
                PATH, // DO NOT LEAVE THIS EMPTY!!! i.e. nullptr
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
                        L"Failed to launch %s, Error in call to ::CreateProcessW: %s\n",
                        PATH,
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

    //--------------------------------------
    // ONCE WE HAVE EXITED THE LOOP
    //--------------------------------------

    if (is_broken_prematurely) {
        exitcode = EXIT_FAILURE;
        ::fwprintf_s(stderr, L"Process launch terminated prematurely, %llu active processes running at termination!\n", nsucceeded_launches);
    }

    // return only when all the processs are signalled (ON PREMATURE LOOP BREAK OR SUCCESSFUL COMPLETION)
    if (!utils::handle_parallel_waits(active_process_handles, active_thread_handles, proc_exitcodes, true, INFINITE)) exitcode = EXIT_FAILURE;

    for (unsigned long i = 0; i < proc_exitcodes.size(); ++i) ::wprintf_s(L"Proc No: %3lu - exitcode: %lu\n", i, proc_exitcodes[i]);

    return exitcode;
}

#ifdef __llvm__
    #pragma clang diagnostic pop
#endif
