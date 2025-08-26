# install the package
# devtools::install_github("jinyizju/U.PhyloMaker")
# install.packages("ape")
# UPhyloMaker is already installed

genus_family_relations <- read.csv("../data/UPhyloMaker/plant_genus_list.csv", sep = ",")
species_of_interest <- read.csv("../data/UPhyloMaker/fred_binom_genus.csv", sep = ",")
megatree <- ape::read.tree("../data/UPhyloMaker/plant_megatree.tre")



phylogeny <- U.PhyloMaker::phylo.maker(sp.list = species_of_interest, tree = megatree, gen.list = genus_family_relations)
ape::write.tree(phy = phylogeny$phylo, file = "../data/UPhyloMaker/chapter03.tre")
