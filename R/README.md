Whenever a R package needs to be installed from GitHub, clone the repo and build it the following steps.
Using `devtools::install_github()` fails to recognize Rtools and results in an installation error.

Steps:

1. `git clone <REPO_URL>`   

2. `&"C:\R-4.5.2\bin\R.exe" CMD build <PACKAGE_NAME>`   
    Make sure the package name matches the name of the cloned repo.   
	This will "build" the library into a .tar.gz archive that's ready to be installed. However, C/C++ sources (if included in the library) won't be built during this step.   

3. `&"C:\R-4.5.2\bin\R.exe" CMD INSTALL .\PACKAGE_NAME.PACKAGE_VERSION.tar.gz`   
	This installs the "bulit" library and this is actually where the C/C++ sources get compiled, linked and installed using Rtools.    
	Make sure you have a matching Rtools installation on your machine.    

