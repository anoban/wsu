// clang .\parhouwie.c -Wall -Wextra -static -march=native -DNDEBUG -O3 -std=c23
// launch the R interpretor in parallel for the hOUwie model fits

// clang-format off
#define _AMD64_ // architecture
#define WIN32_LEAN_AND_MEAN
#include <WinDef.h>
#include <processthreadsapi.h>
#include <sysinfoapi.h>
// clang-format on

#include <stdio.h>
#include <stdlib.h>

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

static const wchar_t* const DISCRETE_MODELS[] = { L"OUM", L"OUMA", L"OUMV", L"OUMVA", nullptr };

typedef enum DISCRETE_MODELS_INDICES {
    ER,  // all rates are identical
    SYM, // symmetrically identical rates
    ARD  // all rates are allowed to be different (asymmetrically)
} DISCRETE_MODELS_INDICES;

static const wchar_t* const CONTINUOUS_MODELS[] = { L"OUM", L"OUMA", L"OUMV", L"OUMVA", nullptr };

typedef enum CONTINUOUS_MODELS_INDICES { OUM, OUMA, OUMV, OUMVA } CONTINUOUS_MODELS_INDICES;

static inline bool __stdcall houwie() {
    // template string for fitting an hOUwie model, with customizable parameters
    static const wchar_t* const HOUWIE_FIT_TEMPLATE =
        L"model <- OUwie::hOUwie(phy = %s, data = %s, rate.cat = %u, discrete_model = \"%s\", continuous_model = \"%s\" , nSim = %u, null.model = %s)";
}

int wmain(_In_ int argc, _In_ wchar_t* argv[]) {
    SYSTEM_INFO sysinf = { 0 };
    GetSystemInfo(&sysinf);
    wprintf_s(L"Number of processors: %lu\n", sysinf.dwNumberOfProcessors); // this machine has 18 cores, which is quite suprising

    return EXIT_SUCCESS;
}
