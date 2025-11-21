if(!require("rtry")){
    install.packages("rtry")
}
library("rtry")

try_photo <- read.delim(file = "../data/chapter2/TRY/photosynthetic_pathways.txt", header = TRUE, sep = '\t', encoding = "Latin 1")
photo <- rtry::rtry_import("../data/chapter2/TRY/43974.txt")

# NOPE, ENDED UP WITH THE SAME MESS
