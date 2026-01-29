write.csv(x = installed.packages()[, c("Package", "Version")], file = "./packages.txt", row.names = FALSE, fileEncoding = "ascii")
