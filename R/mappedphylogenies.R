library("ape")
library("phytools")
library("corHMM")
library("readxl")

phylogeny <- ape::read.tree("../data/chapter2/uphylomaker/FRED4_1292.tre")
data <- readxl::read_excel("../data/chapter2/FRED/subsets/final.xlsx", sheet = "final")
data$binominal_ <- gsub(data$binominal, pattern = ' ', replacement = '_')
stopifnot(all(phylogeny$tip.label %in% data$binominal_))

data <- data[data$binominal_ %in% phylogeny$tip.label, ] # subset the dataset to only include the species in the phylogeny
data <- data[match(phylogeny$tip.label, data$binominal_), ] # name match the dataset with the phylogeny
stopifnot(all(data$binominal_ %in% phylogeny$tip.label))

#------------------------------------------------------------------------
# PHYLOGENY OF THE 995 SPECIES 5 MYCORRHIZAL STATE FRED V3 SUBSET
# TO USE IN THE DRAFT FOR STEVEN SMITH & LUKE MCCORMACK
#------------------------------------------------------------------------

state_colors <- c("red", "blue", "yellow", "orange", "green")
phylogeny <- ape::read.tree("../data/chapter2/uphylomaker/collab_fineroots_log_995_species_means_5states.tre")
tscale_max <- max(phytools::nodeHeights(phylogeny))
states <- read.csv("../data/chapter2/FREDv3subset/collab_fineroots_log_995_species_means_5states.csv")[, c("binominal", "state")]

# par(bg = NA)
png("../plots/995_species_5states_mapped_phylogeny.png", width = 22000, height = 22000, units = "px", res = 400)#, bg = "transparent")
plot <- phytools::plotTree(tree = phylogeny, ftype = "i", fsize = 1.0, lwd = 1, offset = 2, part = 0.998, type = "fan")
tscale_ticks <- seq(0, tscale_max, length.out = 20)
tscale_axis <- axis(1, pos = -1, at = tscale_max - tscale_ticks, cex.axis = 1., labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-3, length(tscale_ticks)), labels = lapply(rev(tscale_ticks), sprintf, fmt = "%.2f"), cex = 1, col = "red")
text(x = tscale_max + 20, y = -2.5, labels = "Time (Million years)", cex = 1.00, col = "red")
ape::tiplabels(pie = to.matrix(states$state, sort(unique(states$state))), piecol = state_colors, cex = 0.05) # nodes at the tips
legend("topright", legend = sort(unique(states$state)), pt.bg = state_colors, cex = 3, pt.cex = 5, pch = 21, ncol = 1)
dev.off()

# the above tree with an outer circle added to show order delineation amongst tips
data <- read.csv("../data/chapter2/FREDv3subset/collab_fineroots_log_995_species_means_5states_name_matched_with_phylogeny.csv")
data
