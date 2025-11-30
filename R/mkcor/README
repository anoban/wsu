### ___A clone of https://github.com/blueraleigh/mkcor___
---------------------------------

To compile the mkcor R package on Windows (using Rtools):

1) ensure that R is installed
2) open PowerShell
3) cd into the directory containing this README file
4) execute the following commands (Note that the commands passed to R.exe are case sensitive!):
    - `& '<your install location>\R-<your installed version>\bin\R.exe' CMD build --no-build-vignettes .` not specifying --no-build-vignettes causes an error for some reason?
       e.g. `& 'C:\Program Files\R\R-4.5.2\bin\R.exe' CMD build --no-build-vignettes .`
       This will create an archive named `mkcor_1.0.tar.gz` in your current working directory. Next execute:
    - `& 'C:\Program Files\R\R-4.5.2\bin\R.exe' CMD INSTALL mkcor_1.0.tar.gz`
       This would install the package for your R installation.

To get started using the package, start a new R session and type

  > library(mkcor)
  > ?mkcor

A vignette is also available

  > vignette("CorrelatedMk")
