library("ape")
library("phytools")
library("nlme")
library("corHMM")
library("geiger")
library("mkcor")

data <- read.csv("../data/chapter2/FREDv3subset/FRED_subset_collab_states_n_species_avg_traits.csv", row.names = "binominal")
tree <- ape::multi2di(ape::read.tree("../data/chapter2/uphylomaker/fredv3subset_collab_trait_n_states.tre"))
ape::is.binary(tree) # should be :)

data <- data.frame(binominal = as.factor(gsub(rownames(data), pattern = ' ', replacement = '_')), rd = data$F00679, srl = data$F00727, myco = as.factor(data$F00645))
row_indices <- match(tree$tip.label, data$binominal)
all(data$binominal[row_indices] == tree$tip.label)
data <- data[row_indices, ]
all(data$binominal == tree$tip.label) # cool


