library("ape")
library("phytools")
library("corHMM")
library("readxl")

phylogeny <- ape::read.tree("../data/chapter2/uphylomaker/FRED4_1292.tre") # this is the ErM removed phylogeny
data <- readxl::read_excel("../data/chapter2/FRED/subsets/final.xlsx", sheet = "final")
data$binominal_ <- gsub(data$binominal, pattern = ' ', replacement = '_')
stopifnot(all(phylogeny$tip.label %in% data$binominal_))

data <- data[data$binominal_ %in% phylogeny$tip.label, ] # subset the dataset to only include the species in the phylogeny
data <- data[match(phylogeny$tip.label, data$binominal_), ] # name match the dataset with the phylogeny
stopifnot(all(data$binominal_ %in% phylogeny$tip.label))


#--------------------------
# phylogeny visualization
#--------------------------

state_colors <- c("red", "blue", "yellow", "orange", "green")
tscale_max <- max(phytools::nodeHeights(phylogeny))
states <- setNames(data$state, nm = data$binominal_)

# par(bg = NA)
png("../plots/states_mapped_phylogeny_1282.png", width = 22000, height = 22000, units = "px", res = 400)#, bg = "transparent")
plot <- phytools::plotTree(tree = phylogeny, ftype = "i", fsize = 0.80, lwd = 1, offset = 2, part = 0.998, type = "fan")
tscale_ticks <- seq(0, tscale_max, length.out = 20)
tscale_axis <- axis(1, pos = -1, at = tscale_max - tscale_ticks, cex.axis = 1., labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-3, length(tscale_ticks)), labels = lapply(rev(tscale_ticks), sprintf, fmt = "%.2f"), cex = 1, col = "red")
text(x = tscale_max + 20, y = -2.5, labels = "Time (Million years)", cex = 1.00, col = "red")
ape::tiplabels(pie = to.matrix(states, sort(unique(states))), piecol = state_colors, cex = 0.04)
legend("topright", legend = sort(unique(states)), pt.bg = state_colors, cex = 3, pt.cex = 5, pch = 21, ncol = 1)
dev.off()
