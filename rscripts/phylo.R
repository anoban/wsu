library("ape")
library("U.PhyloMaker")
library("ggplot2")
library("maps")
library("phytools")
library("ggtree")

genus_family_relations <- read.csv("../data/UPhyloMaker/plant_genus_list.csv", sep = ",")
species_of_interest <- read.csv("../data/UPhyloMaker/fred_binom_genus.csv", sep = ",")
megatree <- ape::read.tree("../data/UPhyloMaker/plant_megatree.tre")


# runtime <- Sys.time()
# this took forfuckingever ~ 3 minutes
# phylogeny <- U.PhyloMaker::phylo.maker(sp.list = species_of_interest, tree = megatree, gen.list = genus_family_relations)
runtime <- Sys.time() - runtime

# serialize the new phylogenetic tree
# ape::write.tree(phy = phylogeny$phylo, file = "../data/UPhyloMaker/chapter03.tre")
# ggtree::ggtree(phylogeny$phylo, layout = "fan", open.angle = 120)

chap3 <- ape::read.tree("../data/UPhyloMaker/chapter03.tre")
plot <- ggtree::ggtree(chap3, layout = "circular") + ggtree::geom_tiplab(size = 2)
plot
