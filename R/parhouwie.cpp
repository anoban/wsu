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
#include <string>

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)

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

// typically, each R process (inside Jupyter) only takes up about ~9% of the CPU, so this could absolutely benefit from paralellization
// wait 5 seconds between launching new processes, so we don't run into (possible???) file I/O issues inside the R instances

// this is a constant string for loading in the necessary libraries and checking their versions
static const wchar_t* const LIBRARY_LOADING__AND_SANITATION = L"suppressPackageStartupMessages({"
                                                              "    library(\"ape\")"
                                                              "    library(\"corHMM\")"
                                                              "    library(\"OUwie\")"
                                                              "})"
                                                              "stopifnot(packageVersion(\"OUwie\") == \"2.16\")"
                                                              "stopifnot(packageVersion(\"corHMM\") == \"2.8\")";

// template string for leading in the phylogeny data and the trait data
static const wchar_t* const DATA_LOADING_TEMPLATE = L"phylogeny <- ape::multi2di(ape::read.tree(file = \"%s\"))"
                                                    "trait_data <- read.csv(\"%s\", stringsAsFactors = TRUE)";

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

static wchar_t DUMMY_R_COMMANDLINE[512] {
    // the readline() at the end will keep the console window open until a user input is provided
    L"C:/R-4.5.2/bin/Rscript.exe --no-save -e 'write.csv(installed.packages()[, c('Package', 'Version')], file = paste0('./packages_', sample(1:100, 1), '.csv'));readline();'"
};

int wmain(_In_ [[maybe_unused]] int argc, [[maybe_unused]] _In_ wchar_t* argv[]) {
    static const wchar_t* const         R_INTERPRETER_PATH { L"C:/R-4.5.2/bin/Rscript.exe" }; // the install directory of the R.exe binary
    static constexpr unsigned long long INTERPRETER_CMDLINE_BUFFSIZE { (1024 * 64) };         // being a bit too generous here

    static constexpr unsigned long long CHILDPROC_STDOUT_PIPE_BUFFSIZE { 0xFFFFF };
    std::string                         childproc_stdout_buffer;
    childproc_stdout_buffer.reserve(CHILDPROC_STDOUT_PIPE_BUFFSIZE);

    SYSTEM_INFO   sysinf { 0 };
    unsigned long bfsize {};

    ::GetSystemInfo(&sysinf);
    ::wprintf_s(L"Number of processors: %lu\n", sysinf.dwNumberOfProcessors); // this machine has 18 cores, which is quite suprising
    // https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex

    wchar_t interpreter_invocation_cmdline[INTERPRETER_CMDLINE_BUFFSIZE] = { 0 };
    // to invoke the R interpreter with strings passed as code, use the -e flag
    // e.g. R.exe -e "write.csv(installed.packages(), file=\"name.csv\")"
    // ::swprintf_s(interpreter_invocation_cmdline, INTERPRETER_CMDLINE_BUFFSIZE, R_INTERPRETER_PATH " -e \"\"");

    ::_putws(DUMMY_R_COMMANDLINE);

    PROCESS_INFORMATION childprocinfo {};
    HANDLE64            childproc_stdout_write_end {} /* child's end */, childproc_stdout_read_end {} /* parent's end */;
    SECURITY_ATTRIBUTES childsecattrs { .nLength = sizeof(SECURITY_ATTRIBUTES), .lpSecurityDescriptor = nullptr, .bInheritHandle = true };
    STARTUPINFOW        childstarupinfo = {
               .cb          = sizeof(STARTUPINFOW),
               .lpTitle     = L"ChildProc",
               .dwFlags     = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES | STARTF_FORCEONFEEDBACK,
               .wShowWindow = SW_SHOW,
               .hStdOutput  = childproc_stdout_write_end, // we are only interested in the stdouts of the child process
               .hStdError   = childproc_stdout_write_end,
    };

    // if (!::CreatePipe(&childproc_stdout_write_end, &childproc_stdout_read_end, &childsecattrs, CHILDPROC_STDOUT_PIPE_BUFFSIZE))
    //     ::fwprintf_s(stderr, L"%s error in call to CreatePipe!\n", ::error_code_to_string(::GetLastError()));
    // else
    //     // since we launch the child process after creating the pipe, make sure that the read end of the child process' stdout that the parent process has isn't inherited by the child process upon launch
    //     if (!::SetHandleInformation(childproc_stdout_read_end, HANDLE_FLAG_INHERIT, false))
    //         ::fwprintf_s(stderr, L"%s error in call to SetHandleInformation!\n", ::error_code_to_string(::GetLastError()));

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
        ::fwprintf_s(stderr, L"%s error in call to CreateProcessW!\n", ::error_code_to_string(::GetLastError()));
        return EXIT_FAILURE;
    }

    // unsigned long readbytes_pipe {};
    // long          read_status { true };
    // for (;;) {
    //     read_status = ::ReadFile(
    //         childproc_stdout_read_end,
    //         childproc_stdout_buffer.data() + readbytes_pipe, // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    //         CHILDPROC_STDOUT_PIPE_BUFFSIZE,
    //         &readbytes_pipe,
    //         nullptr
    //     );
    //     if (!readbytes_pipe || !read_status) break;
    // }
    // ::WaitForSingleObject(childprocinfo.hProcess, INFINITE);

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

    ::wprintf_s(L"%S\n", childproc_stdout_buffer.c_str());
    ::GetExitCodeProcess(childprocinfo.hProcess, &childproc_exitcode);
    ::wprintf_s(L"Exit code of the child process is %lu\n", childproc_exitcode);

    // https://learn.microsoft.com/en-us/windows/win32/procthread/creating-processes
    ::CloseHandle(childprocinfo.hProcess); // close the child process
    ::CloseHandle(childprocinfo.hThread);  // close the child thread

    return EXIT_SUCCESS;
}

// NOLINTEND(cppcoreguidelines-pro-type-vararg,modernize-avoid-c-arrays)
