Whenever an R package needs to be installed from `GitHub`, clone the repo and build it using the following steps. 
Using `devtools::install_github()` fails to recognize the installed `Rtools` and results in installation errors.

------------------------------

Steps:

1. `$ git clone https://github.com/user/repo.git`

2. `&"<PATH_TO_R_EXE>" CMD build <PACKAGE_NAME> --no-build-vignettes --ignore-vignettes` (`PowerShell`)
    `PATH_TO_R_EXE` is the path to `R.exe` on your system - e.g. `C:\Program Files\R\R-4.6.0\bin\R.exe`. The commandline arguments passed to `R.exe` are case sensitive.
    `--no-build-vignettes --ignore-vignettes` help avoid building `R` vignettes which usually require a `LaTex` system and its dependencies.
    Make sure the package name matches the name of the cloned repo. This step will _build_ the library into a `.tar.gz` archive that's ready to be installed. 
    However, C/C++ sources (if included in the library) won't be built during this step.

3. `&"<PATH_TO_R_EXE>" CMD INSTALL .\PACKAGE_NAME.PACKAGE_VERSION.tar.gz` (`PowerShell`)
	This installs the _bulit_ library and this is actually where the C/C++ sources get compiled, linked and installed using `Rtools`.
	Make sure you have a matching `Rtools` installation on your machine.
