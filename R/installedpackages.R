write.csv(x = installed.packages()[, c("Package", "Version")], file = "./installed.packages.csv", row.names = FALSE, fileEncoding = "ascii")
