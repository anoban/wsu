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
ape::plot.phylo(x = phylogeny, ftype = "i", fsize = 0.80, lwd = 1, offset = 1, part = 1, type = "fan", show.tip.label = FALSE, show.node.label = FALSE)
ape::tiplabels(tip = seq_along(phylogeny$tip.label)[data$state == "AM"], col = "red", pch=19, cex = 1.5, offset = 2)
ape::tiplabels(tip = seq_along(phylogeny$tip.label)[data$state == "EcM"], col = "darkgreen", pch=19, cex = 1.5, offset = 4)
ape::tiplabels(tip = seq_along(phylogeny$tip.label)[data$state == "NM"], col = "blue", pch=19, cex = 1.5, offset = 4)
ape::tiplabels(tip = seq_along(phylogeny$tip.label)[data$state == "ErM"], col = "brown", pch=19, cex = 1.5, offset = 4)
ape::tiplabels(tip = seq_along(phylogeny$tip.label)[data$state == "EcMAM"], col = "orange", pch=19, cex = 1.5, offset = 4)
ape::tiplabels(tip = seq_along(phylogeny$tip.label)[data$state == "NMAM"], col = "purple", pch=19, cex = 1.5, offset = 4)
dev.off()

