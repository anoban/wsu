library("ape")
library("phytools")
library("readxl")

phylogeny <- ape::read.tree("../data/chapter2/uphylomaker/FRED4_1301.tre")
data <- readxl::read_excel("../data/chapter2/FRED/subsets/final.xlsx", sheet = "final")
data$binominal <- gsub(data$binominal, pattern = ' ', replacement = '_')
stopifnot(all(data$binominal %in% phylogeny$tip.label))

data <- data[match(phylogeny$tip.label, data$binominal), ] # name match the dataset with the phylogeny
stopifnot(all(data$binominal == phylogeny$tip.label))


#--------------------------
# phylogeny visualization
#--------------------------

png("../plots/phylogeny.png", width = 22000, height = 22000, units = "px", res = 400)#, bg = "transparent")
plot(x = phylogeny, ftype = "i", fsize = 0.80, lwd = 1, offset = 2, part = 1, type = "fan", show.tip.label = FALSE, show.node.label = FALSE)

# lets plot each mycorrhizal state in a separate line
ape::tiplabels(tip = data[data$state == "AM", "binominal"]$binominal, # paint only the given tips in the specified colour
               col = "red", pch=21, cex = 3)

dev.off()
