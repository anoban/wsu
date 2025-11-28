library("ape")
library("phytools")
library("nlme")
library("corHMM")
library("geiger")

data <- read.csv("../data/chapter2/FREDv3subset/FRED_subset_collab_states_n_species_avg_traits.csv")
tree <- ape::multi2di(ape::read.tree("../data/chapter2/uphylomaker/fredv3subset_collab_trait_n_states.tre"))
ape::is.binary(tree) # should be :)

