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

# filter the edges that end at tips
all_edges <- data.frame(phylogeny$edge)
all_edges$colour <- NA # introduce a colour column
# update the colour column with states which can later be remapped to colours
all_edges[phylogeny$edge[, 2] %in% 1:length(phylogeny$tip.label), "colour"] <- data$state[all_edges[phylogeny$edge[, 2] %in% 1:length(phylogeny$tip.label), "X2"]]
STATE_COLOURS <- c(AM = "darkgreen", EcM = "brown", ErM = "#5A778E", NMAM = "#483023", EcMAM = "#FF00FF")
all_edges$colour <- STATE_COLOURS[all_edges$colour]
all_edges$colour[is.na(all_edges$colour)] <- "#CDCBCE"

png("../plots/phylogeny.png", width = 22000, height = 22000, units = "px", res = 400)#, bg = "transparent")
ape::plot.phylo(x = phylogeny, lwd = 1, type = "fan", show.tip.label = FALSE, show.node.label = FALSE, edge.color = all_edges$colour)
ape::tiplabels(bg = "red", pch=21, cex = scale(exp(data$F00679), center = FALSE), offset = 1.5) # RD
ape::tiplabels(bg = "blue", pch=21, cex = scale(exp(data$F00727), center = FALSE), offset = 5.5) # SRL
ape::tiplabels(bg = "orange", pch=21, cex = scale(exp(data$F00709), center = FALSE), offset = 9.5) # RTD
dev.off()



