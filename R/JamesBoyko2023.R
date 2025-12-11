# Boyko, J.D., O’Meara, B.C. and Beaulieu, J.M. (2023) “A novel method for jointly modeling the evolution of discrete and continuous traits,” Evolution, 77(3), pp. 836–851.
# Available at: https://doi.org/10.1093/evolut/qpad002.

# code repo - https://github.com/tncvasconcelos/seed_dispersal
# https://github.com/jboyko/2020_houwie
# try and reproduce the analyses of the above paper to get comfortable with the OUwie stuff

suppressPackageStartupMessages({
    library("ape")
    library("phytools")
    library("nlme")
    library("corHMM")
    library("geiger")
    library("mkcor")
    library("OUwie")
    library("reshape2")
    library("ggplot2")
})

# pick family Ericaceae

# we don't want the other unnecessary columns
fruits <- read.csv("../R/thirdparty/seed_dispersal/trait_data/Ericaceae_trait_data.csv")[, c("Species", "Fruit_type")]
fruits <- fruits[fruits$Fruit_type == "Fleshy" | fruits$Fruit_type == "Dry", ] # drop the rows that do not have actual data for life form
# Fruit_type is a binary discrete categorical trait with values Fleshy and Dry
fruits$Fruit_type |> table()
dim(fruits ) # 444 species

# continuous traits
climate <- read.csv("../R/thirdparty/seed_dispersal/trait_data/Ericaceae_niche.csv")
# looks like this too has the fruit type info embedded
climate$Fruit_type |> table() # yup


# load in the phylogenetic tree for Ericaceae
tree <- ape::read.tree("../R/thirdparty/seed_dispersal/trees/Ericaceae_Schwery_etal_2015.tre")
ape::is.binary(tree) # TRUE :)

# look up https://github.com/jboyko/2020_houwie/blob/master/04_empirical-seed-dispersal.R for reference
