library("ape")
library("U.PhyloMaker")
library("maps")
library("phytools")
# library("ggtree")

#############################################
# FIRST TIME PHYLOGENETIC TREE CONSTRUCTION #
#############################################

GENUS_FAMILY_RELATIONS <- read.csv("../data/chapter2/uphylomaker/plant_genus_list.csv", sep = ",") # this is from the U.PhyloMaker GitHub repo
FRED_COLLAB_SPECIES <- read.csv("../data/chapter2/FREDv3subset/FRED_subset_collab_unique_taxa.csv", sep = ",") # 395 collab axis species
FRED_COLLAB_SPECIES <- data.frame(species = FRED_COLLAB_SPECIES$binominal, genus = FRED_COLLAB_SPECIES$F01286, family = FRED_COLLAB_SPECIES$F01289,species.relative = NA, genus.relative = NA)
# the above needs to have the following columns => species,genus,family,species.relative,genus.relative
MEGATREE <- ape::read.tree("../data/chapter2/uphylomaker/GBOTB_extended_WP.tre")

runtime <- Sys.time()
phylogeny <- U.PhyloMaker::phylo.maker(sp.list = FRED_COLLAB_SPECIES, tree = MEGATREE, gen.list = GENUS_FAMILY_RELATIONS)
runtime <- Sys.time() - runtime # Time difference of 2.631325 mins

# serialize the new phylogenetic tree
ape::write.tree(phy = phylogeny$phylo, file = "../data/chapter2/uphylomaker/FRED_subset_collab_395sp.tre")
# ggtree::ggtree(phylogeny$phylo, layout = "fan", open.angle = 120)

##############################################################
# SUBSEQUENT ANALYSES USING THE SERIALIZED PHYLOGENETIC TREE #
##############################################################

COLLAB_395SP_TREE <- ape::read.tree("../data/chapter2/uphylomaker/fredv3subset.tre")
# opting to use phytools instead of ggtree because ggtree replaces spaces in scientific names with fugly underscores
# plot <- ggtree::ggtree(chapter2, layout = "circular", size = 0.5) + ggtree::geom_tiplab(size = 3)
# ggtree::ggsave(filename = "../plots/phyolo.png", plot = plot, device = png, width = 10, height = 10, units = "in", bg = "transparent", dpi = 500, scale = 1.5)
# ggtree::gheatmap(plot)

length(COLLAB_395SP_TREE$tip.label) # must be 395!!

htree <- max(phytools::nodeHeights(COLLAB_395SP_TREE)) # timescale of the tree
png("../plots/FRED_collab_395sp_phylogeny.png", width = 8000, height = 8000, units = "px", res = 300)
plot <- phytools::plotTree(COLLAB_395SP_TREE, ftype = "i", fsize = 1.4, type = "fan", lwd = 1, part = 0.99)
# create a timescale axis that begins at the edge of the circle and increases towards the center
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1.75, labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-16, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1.5, col = "red")
text(x = 250, y = -35, labels = "Time (Million years)", cex = 1.5, col = "red")
dev.off()

