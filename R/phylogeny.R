library("ape")
library("U.PhyloMaker")
library("maps")
library("phytools")



#-------------------------------------
# first time phylogeny construction
#-------------------------------------

data <- read.csv("../data/chapter2/FRED/subsets/final.csv", sep = ",")
# sp.list needs to have the following columns => species, genus, family, species.relative, genus.relative
species_list <- data.frame(species = data$binominal, genus = data$F01286, family = data$F01289, species.relative = NA, genus.relative = NA)
genus_list <- data.frame(genus = data$F01286, family = data$F01289) # genus and family relationships
megatree <- ape::read.tree("../data/chapter2/uphylomaker/GBOTB.extended.TPL.tre") # TPL megatree

runtime <- Sys.time()
phylogeny <- U.PhyloMaker::phylo.maker(sp.list = species_list, tree = megatree, gen.list = genus_list)
runtime <- Sys.time() - runtime

# serialize the new phylogenetic tree
ape::write.tree(phy = phylogeny$phylo, file = "../data/chapter2/uphylomaker/FRED_subset_collab_395sp.tre")




COLLAB_395SP_TREE <- ape::read.tree("../data/chapter2/uphylomaker/FRED_subset_collab_395sp.tre")

stopifnot(length(COLLAB_395SP_TREE$tip.label) == 395) # must be 395!!

htree <- max(phytools::nodeHeights(COLLAB_395SP_TREE)) # timescale of the tree
png("../plots/", width = 8000, height = 8000, units = "px", res = 200)
plot <- phytools::plotTree(COLLAB_395SP_TREE, ftype = "i", fsize = 1.4, type = "fan", lwd = 1, part = 0.99)
# create a timescale axis that begins at the edge of the circle and increases towards the center
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1.75, labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-16, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1.5, col = "red")
text(x = 250, y = -35, labels = "Time (Million years)", cex = 1.5, col = "red")
dev.off()



#---------------------------------------------------------------
# name match the finalized root trait data with the phylogeny
#---------------------------------------------------------------

tree <- ape::read.tree("./../data/chapter2/uphylomaker/FRED4_1301_species.tre")
tree

# may be we should get rid of the ErM and OrM species before name matching

traits <- read.csv("./../data/chapter2/FRED/subsets/final_name_matched.csv")
traits$binominal <- gsub(traits$binominal, pattern = ' ', replacement = '_')

setdiff(tree$tip.label, traits$binominal) # some species failed to bind during the phylogeny creating and we used their synonyms
# these need to be revised in the trait data dataset, otherwise it will propagate errors during model fits!


idx <- match(tree$tip.label, traits$binominal)
idx
all(traits[idx, ]$binominal == tree$tip.label)
