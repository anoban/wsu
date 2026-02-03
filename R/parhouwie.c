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
static const wchar_t* const DATA_LOADING_TEMPLATE =
    L"phylogeny <- ape::multi2di(ape::read.tree(file = \"../../data/chapter2/uphylomaker/%s\"))" // assume that the directories won't need changing
    "trait_data <- read.csv(\"../../data/chapter2/FREDv3subset/%s\", stringsAsFactors = TRUE)"; // same here

typedef enum DISCRETE_MODELS { ER, SYM, ARD } DISCRETE_MODELS; // DISCRETE MODELS

#define ER  L"ER"  // all rates are identical
#define SYM L"SYM" // symmetrically identical rates
#define ARD L"ARD" // all rates are allowed to be different (asymmetrically)

typedef enum CONTINUOUS_MODELS { OUM, OUMA, OUMV, OUMVA } CONTINUOUS_MODELS; // CONTINUOUS MODELS

#define OUM   L"OUM"   //
#define OUMA  L"OUMA"  //
#define OUMV  L"OUMV"  //
#define OUMVA L"OUMVA" //

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
