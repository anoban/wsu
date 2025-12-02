library("ape")
library("phytools")
library("OUwie")
library("corHMM")

# make sure the OUwie version you have is recent 2.16 NOT the old ones because they do not have the OUwie::hOUwie() function
# CRAN CDN delivers 2.10 by default
# to get the recent version, clone the github repo and use R CMD build --no-build-vignettes followed by R CMD INSTALL
# https://thej022214.github.io/OUwie/reference/hOUwie.html

data <- read.csv("../data/chapter2/FREDv3subset/FRED_subset_collab_states_n_species_avg_traits.csv", row.names = "binominal")
tree <- ape::multi2di(ape::read.tree("../data/chapter2/uphylomaker/fredv3subset_collab_trait_n_states.tre"))
ape::is.binary(tree) # should be :)

data <- data.frame(binominal = as.factor(gsub(rownames(data), pattern = ' ', replacement = '_')), rd = data$F00679, srl = data$F00727, myco = as.factor(data$F00645))
row_indices <- match(tree$tip.label, data$binominal)
all(data$binominal[row_indices] == tree$tip.label)
data <- data[row_indices, ]
all(data$binominal == tree$tip.label) # cool

# ouWIE::HOUwie() expects the data to have columns in the following order => species, categorical trait followed by continuous trait
data_sub <- data[, c("binominal", "myco", "srl")]
model <- OUwie::hOUwie(phy = tree, data = data_sub, rate.cat = 1, discrete_model = "ER", continuous_model = "OUM", nSim = 25)
