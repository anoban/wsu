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
rtd_vals <- rtd_srl[gsub(pattern = "_", replacement = " ", x = tree$tip.label), ]$F00709
srl_vals <- rtd_srl[gsub(pattern = "_", replacement = " ", x = tree$tip.label), ]$F00727
# these are mean RTD, SRL values in the same order as the tip labels of the phylogenetic tree

astate_rtd <- phytools::fastAnc(tree = tree, x = rtd_vals, CI = TRUE)
astate_srl <- phytools::fastAnc(tree = tree, x = srl_vals, CI = TRUE)

# GOT A WARNING
# x should be a vector with names corresponding to the taxon labels of the tree.
# Assuming x is in the order of tree$tip.label (this is seldom true).
