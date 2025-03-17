$cfiles = [System.Collections.ArrayList]::new()
$unrecognized = [System.Collections.ArrayList]::new()

foreach ($arg in $args) {
    if ($arg -clike "*.cpp" -or $arg -clike "*.cc") { # .cc is here only for gtest sources :(
        [void]$cfiles.Add($arg.ToString().Replace(".\", ""))
    }
    else {
        [void]$unrecognized.Add($arg)
    }
}

if ($unrecognized.Count -ne 0) {
    Write-Error "Incompatible files passed for compilation: ${unrecognized}"
    Exit 1
}

$cflags = @(
    # "/arch:AVX512", # university laptop with Core ULTRA 5 doesn't have zmm lanes
    "/debug:none"
    "/diagnostics:caret",
    "/DNDEBUG",
    "/D_NDEBUG",
    "/EHa",
    "/F0x10485100",
    "-fcf-protection:full",
    "/fp:fast",
    "/Gd",
    "/GF",
    "/GR",  # RTTI
    "/GS",
    "/guard:cf",
    "/Gw",
    "/I./",
    "/I./googletest",
    "/I./googletest/include",
    "/J",
    "/MT",
    "/O3",
    "/Oi",
    "/Ot",
    # "/QaxCORE-AVX512", # university laptop :(
    "/Qbranches-within-32B-boundaries",
    "/Qftz",
    "/Qgather-",
    "/Qintrinsic-promote",
	"/Qimf-absolute-error:1E-10",
	"/Qimf-accuracy-bits:30/f",
	"/Qimf-accuracy-bits:60/",
	"/Qimf-accuracy-bits:70/l",
	"/Qimf-arch-consistency:false",
	"/Qimf-domain-exclusion:0",
	"/Qimf-max-error:1E-2",
	"/Qimf-use-svml:false",
	"/Qfma",
	"/Qfp-speculation:safe",
    "/Qipo",
	"/Qkeep-static-consts-",
	# "/Qlong-double", causes compiler errors with c++20 and above (80 bits wide doubles lack formatting support in UCRT I guess!!)
    "/Qm64",
    "/Qopt-assume-no-loop-carried-dep=2",
    "/Qopt-dynamic-align",
    "/Qopt-multiple-gather-scatter-by-shuffles",
    "/Qopt-prefetch:5",
    "/Qopt-prefetch-distance:10000",
	"/Qpc80",
    "/Qscatter-",
    "/Qstd:c++23",
    "/Qvec",
    "/Qvec-peel-loops",
    "/Qvec-threshold:0",
    "/Qvec-with-mask",
    "/Qunroll:10000",
    "/TP",
    "/vd3", # https://learn.microsoft.com/en-us/cpp/preprocessor/vtordisp?view=msvc-170
	"/Wabi",
    "/Wall",
	"/Wcomment",
	"/Wdeprecated",
	"/Wextra-tokens",
	"/Wformat",
	"/Wformat-security",
	"/Wmain",
	"/Wno-missing-prototypes",
	"/Wno-pointer-arith",
	"/Wreturn-type",
	"/Wshadow",
	"/Wsign-compare",
	"/Wstrict-aliasing",
	"/Wuninitialized",
	"/Wunknown-pragmas",
	"/Wunused-function",
	"/Wunused-variable",
	"/Wwrite-strings",
    "/wd4710",
    "/wd4711",
    "/wd4820",
    "/Zc:char8_t",
	"/Zc:strictStrings",
    "/Zc:twoPhase"
)

Write-Host "icx.exe ${cfiles} ${cflags}" -ForegroundColor Cyan
icx.exe $cfiles $cflags
