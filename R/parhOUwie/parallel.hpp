#pragma once
#ifndef __PARALLEL_HPP
    #define __PARALLEL_HPP 1
#endif

#if !(defined(_WIN32) || defined(_WIN64)) && !(defined(_MSC_VER) || defined(_MSC_FULL_VER))
    #error This is a Windows only implementation that liberally uses the Win32 API, not meant to be used on other platforms!.
#endif

#if defined(_MSC_FULL_VER) && !defined(__llvm__) // MSVC specific warnings
    #pragma warning(disable : 4267 4710 4711 4774 4800 4820)
#endif

#ifdef __llvm__ // only for LLVM

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

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>


// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)

#pragma comment(lib, "Shlwapi.lib") // for ::PathFileExistsW

namespace parallel {

    static constexpr unsigned long long ERRORMSG_BUFFSIZE { 0x2EE }; // length of the error message buffer in number of wchar_t s

    // get the string representation of a _WIN32 error code
    // NOLINTNEXTLINE(readability-redundant-inline-specifier)
    [[nodiscard, clang::always_inline]] static inline const wchar_t* __stdcall __error_code_to_wstring(
        _In_ const unsigned long& errcode, _Inout_ HANDLE64 hntdsbmsg
    ) noexcept {
        static wchar_t buffer[ERRORMSG_BUFFSIZE] { 0 }; // needs to be in static memory for returning
        // without this the previously written buffer can get partially overwritten and returned in subsequent function invocations
        ::memset(buffer, 0, sizeof(buffer));

        // https://devblogs.microsoft.com/oldnewthing/20191025-00/?p=103025
        // https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-formatmessage
        unsigned long nbyteswritten = ::FormatMessageW(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK,
            nullptr,
            errcode,
            0,
            buffer,
            ERRORMSG_BUFFSIZE,
            nullptr
        );

        if (!nbyteswritten) { // will be 0 if the call above to FormatMessageW failed; if that, the error string is not found in the system, try Ntdsbmsg.dll
            // if the library hasn't already been loaded by previous calls to this function
            if (!hntdsbmsg) hntdsbmsg = ::LoadLibraryW(L"Ntdsbmsg.DLL");
            if (!hntdsbmsg) { // will be NULL if the DLL failed to load
                ::fputws(L"Failed to load Ntdsbmsg.DLL", stderr);
                return buffer; // must be an empty buffer here
            }

            nbyteswritten = ::FormatMessageW(
                FORMAT_MESSAGE_FROM_HMODULE | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK,
                hntdsbmsg,
                errcode,
                0,
                buffer,
                ERRORMSG_BUFFSIZE,
                nullptr
            );
            // ::FreeLibrary(handle_ntdsbmsg); // detach the DLL from the process - atexit() will handle this for us
        }
        return buffer;
    }

    // will only return true when the wait is signalled success
    [[nodiscard]] static inline bool __stdcall handle_parallel_waits( // NOLINT(readability-redundant-inline-specifier)
        _Inout_ std::vector<HANDLE64>& phandles, _Inout_ std::vector<HANDLE64>& thandles, _Inout_ std::vector<unsigned long>& excodes, _In_ const bool& all, _In_ const unsigned long& duration
    ) noexcept {
        // made the function more customizeable
        // WAIT_OBJECT_0 is defined as 0 and WAIT_ABANDONED_0 is defined as 0x00000080L
        // so the returned value is practically equal to the offset of the signalled handle WHEN WAIT SUCCEEDS
        unsigned long handle_offset { 0xFF };
        unsigned long exitcode { 0xAABBCC };
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
                ::fwprintf_s(stderr, L"WaitForMultipleObjects signalled WAIT_FAILED, %s\n", __error_code_to_wstring(::GetLastError()));
                if (all) goto CLOSE_ACTIVE_HANDLES_AND_EXIT; // if the wait was for all processes, close all the handles and return false
                return false;                                // else just return false

            case WAIT_TIMEOUT :
                ::fputws(L"WaitForMultipleObjects signalled WAIT_TIMEOUT\n", stderr);
                if (all) goto CLOSE_ACTIVE_HANDLES_AND_EXIT;
                return false;

            default : break;
        }

        // WAIT_OBJECT_0 to (WAIT_OBJECT_0 + nCount - 1)
        if (// (waitstatus >= WAIT_OBJECT_0) && // will always be true because the return value of ::WaitForMultipleObjects() is unsigned
             waitstatus < (WAIT_OBJECT_0 + phandles.size())) {
            exitval = true;
            if (all) goto CLOSE_ACTIVE_HANDLES_AND_EXIT;
            // bWaitAll == FALSE
            handle_offset = waitstatus - WAIT_OBJECT_0;
            goto CLOSE_SELECTED_HANDLE_AND_EXIT;
        }

        // WAIT_ABANDONED_0 to (WAIT_ABANDONED_0 + nCount - 1)
        else if ((waitstatus >= WAIT_ABANDONED_0) && (waitstatus < (WAIT_ABANDONED_0 + phandles.size()))) {
            if (all) {
                ::fputws(
                    L"WaitForMultipleObjects signalled WAIT_ABANDONED with bWaitAll set to TRUE, 1 or more probable abandoned mutexes!\n",
                    stderr
                );
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
                ::fwprintf_s(stderr, L"GetExitCodeProcess returned 0, %s\n", __error_code_to_wstring(::GetLastError()));
            else
                excodes.push_back(exitcode);
            ::CloseHandle(phandles.at(i));
            ::CloseHandle(thandles.at(i));
        }

        phandles.clear();
        thandles.clear();
        return exitval;

CLOSE_SELECTED_HANDLE_AND_EXIT:
        // capture the exit code
        if (!::GetExitCodeProcess(*(phandles.data() + handle_offset), &exitcode))
            ::fwprintf_s(stderr, L"GetExitCodeProcess returned 0, %s\n", __error_code_to_wstring(::GetLastError()));
        else
            excodes.push_back(exitcode);
        ::CloseHandle(*(phandles.data() + handle_offset)); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        phandles.erase(phandles.begin() + handle_offset);  // remove the signalled process
        ::CloseHandle(*(thandles.data() + handle_offset)); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        thandles.erase(thandles.begin() + handle_offset);  // remove the signalled thread handle
        return exitval;
    }

} // namespace parallel

// NOLINTEND(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)

#ifdef __llvm__
    #pragma clang diagnostic pop
#endif
