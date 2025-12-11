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
fruits <- read.csv("../R/thirdparty/seed_dispersal/trait_dataa/Ericaceae_trait_dataa.csv")[, c("Species", "Fruit_type")]
fruits <- fruits[fruits$Fruit_type == "Fleshy" | fruits$Fruit_type == "Dry", ] # drop the rows that do not have actual data for life form
# Fruit_type is a binary discrete categorical trait with values Fleshy and Dry
fruits$Fruit_type |> table()
dim(fruits ) # 444 species

# continuous traits
climate <- read.csv("../R/thirdparty/seed_dispersal/trait_dataa/Ericaceae_niche.csv")
# looks like this too has the fruit type info embedded
climate$Fruit_type |> table() # yup


# load in the phylogenetic tree for Ericaceae
tree <- ape::read.tree("../R/thirdparty/seed_dispersal/trees/Ericaceae_Schwery_etal_2015.tre")
ape::is.binary(tree) # TRUE :)

# look up https://github.com/jboyko/2020_houwie/blob/master/04_empirical-seed-dispersal.R for reference

# drop the species in the dataaset if they are not in the phylogenetic tree
species <- intersect(tree$tip.label, climate$species) # 309 species
indices <- match(species, climate$species)
data <- climate[indices, ]

# make sure that we only have species for which there's data in the phylogeny
stopifnot(length(intersect(data$species, tree$tip.label)) == length(indices))

# drop the unnecessary species from the phylogeny
species_to_drop_from_tree <- setdiff(tree$tip.label, species) # order of args matters here!!!
pruned_tree <- ape::drop.tip(phy = tree, tip = species_to_drop_from_tree)

# visualize the pruned phylogeny
Tmax <- max(ape::branching.times(pruned_tree))
plot(pruned_tree, show.tip.label = FALSE, x.lim = c(0, Tmax + 0.2 * Tmax))
start <- Tmax + (0.005 * Tmax)
data.prop <- (data[,3] - min(data[,3]))/max(data[,3] - min(data[,3]))
xadd <- Tmax * 0.2
jitter <- 0.1 * xadd
cols = c("brown", "purple")
for(i in 1:length(data.prop)){
    lines(list(x = c(start, start + (data.prop[i] * xadd)), y = c(i,i)),
          col = cols[ifelse(data$reg[i] == "Dry", 1, 2)],
          lwd = 1)
}
