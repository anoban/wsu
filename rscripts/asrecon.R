# https://jhudatascience.org/AnVIL_Phylogenetic-Techniques/ancestral-state-reconstruction.html
# http://www.phytools.org/eqg/Exercise_5.2/

library("ape")
library("phytools")


# "F00727" - SRL
# "F00709" - RTD

# load in the dataset
rtd_srl <- read.csv("../data/chapter2/FREDv3subset/RTD_SRL_species_means.csv", row.names = "binominal") # average RTD and SRL trait values for the 203 species
# did not do root order based trait normalizations :(
tree <- ape::read.tree("../data/chapter2/uphylomaker/fredv3subset.tre") # phylogenetic tree of the 203 species

# ancestral state reconstruction for RTD & SRL
# tip.labels have underscores in-between genus name and specific epithet :/
tip_labels <- tree$tip.label
named_rtd_vec <- setNames(rtd_srl[gsub(pattern = "_", replacement = " ", x = tip_labels), ]$F00709, tip_labels)
named_srl_vec <- setNames(rtd_srl[gsub(pattern = "_", replacement = " ", x = tip_labels), ]$F00727, tip_labels)
# these are mean RTD, SRL values in the same order as the tip labels of the phylogenetic tree

astate_rtd <- phytools::fastAnc(tree = tree, x = named_rtd_vec, CI = TRUE)
astate_srl <- phytools::fastAnc(tree = tree, x = named_srl_vec, CI = TRUE)

png("../plots/asrRTD.png", width = 8000, height = 8000, units = "px", res = 300)
rtd_map <- phytools::contMap(tree = tree, x = named_rtd_vec, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(rtd_map, type = "fan")
dev.off()

png("../plots/asrSRL.png", width = 8000, height = 8000, units = "px", res = 300)
rtd_map <- phytools::contMap(tree = tree, x = named_srl_vec, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(rtd_map, type = "fan")
dev.off()

# correlation between the evolution of these two traits
plot(named_rtd_vec, named_srl_vec)
