library("ape")
library("U.PhyloMaker")
library("ggplot2")
library("maps")
library("phytools")

genus_family_relations <- read.csv("../data/UPhyloMaker/plant_genus_list.csv", sep = ",")
species_of_interest <- read.csv("../data/UPhyloMaker/fred_binom_genus.csv", sep = ",")
megatree <- ape::read.tree("../data/UPhyloMaker/plant_megatree.tre")


runtime <- Sys.time()
phylogeny <- U.PhyloMaker::phylo.maker(sp.list = species_of_interest, tree = megatree, gen.list = genus_family_relations)
runtime <- sys.time() - clock

ape::write.tree(phy = phylogeny$phylo, file = "../data/UPhyloMaker/chapter03.tre")
