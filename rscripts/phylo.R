library("ape")
library("U.PhyloMaker")
library("ggplot2")
library("maps")
library("phytools")
# library("ggtree")

#############################################
# FIRST TIME PHYLOGENETIC TREE CONSTRUCTION #
#############################################

genus_family_relations <- read.csv("../data/chapter2/uphylomaker/plant_genus_list.csv", sep = ",")
species_of_interest <- read.csv("../data/chapter2/fred_binom_genus.csv", sep = ",")
megatree <- ape::read.tree("../data/chapter2/uphylomaker/GBOTB_extended_WP.tre")
runtime <- Sys.time()
phylogeny <- U.PhyloMaker::phylo.maker(sp.list = species_of_interest, tree = megatree, gen.list = genus_family_relations) # this took forfuckingever ~ 3 minutes
runtime <- Sys.time() - runtime # Time difference of 2.631325 mins

# serialize the new phylogenetic tree
ape::write.tree(phy = phylogeny$phylo, file = "../data/chapter2/uphylomaker/fredv3subset.tre")
# ggtree::ggtree(phylogeny$phylo, layout = "fan", open.angle = 120)

##############################################################
# SUBSEQUENT ANALYSES USING THE SERIALIZED PHYLOGENETIC TREE #
##############################################################

fredv3tree <- ape::read.tree("../data/chapter2/uphylomaker/fredv3subset.tre")
# opting to use phytools instead of ggtree because ggtree replaces spaces in scientific names with fugly underscores
# plot <- ggtree::ggtree(chapter2, layout = "circular", size = 0.5) + ggtree::geom_tiplab(size = 3)
# ggtree::ggsave(filename = "../plots/phyolo.png", plot = plot, device = png, width = 10, height = 10, units = "in", bg = "transparent", dpi = 500, scale = 1.5)
# ggtree::gheatmap(plot)

htree <- max(phytools::nodeHeights(fredv3tree)) # timescale of the tree
png("../plots/phyolo-phytools.png", width = 8000, height = 8000, units = "px", res = 300)
plot <- phytools::plotTree(fredv3tree, ftype = "i", fsize = 1.4, type = "fan", lwd = 1, part = 0.99)
# create a timescale axis that begins at the edge of the circle and increases towards the center
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1.75, labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-16, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1.5, col = "red")
text(x = 250, y = -35, labels = "Time (Million years)", cex = 1.5, col = "red")
dev.off()


# plot the whole megatree
png("../plots/megatree.png", width = 10000, height = 10000, units = "px", res = 300)
plot <- phytools::plotTree(megatree, ftype = "i", fsize = 1.4, type = "fan", lwd = 1, part = 0.99)
dev.off()
