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
# runtime <- Sys.time() - runtime

# serialize the new phylogenetic tree
# ape::write.tree(phy = phylogeny$phylo, file = "../data/UPhyloMaker/chapter03.tre")
# ggtree::ggtree(phylogeny$phylo, layout = "fan", open.angle = 120)

fred <- utils::read.csv("../data/FRED/FRED3_Entire_Database_2021.csv", header = TRUE)

chap3 <- ape::read.tree("../data/UPhyloMaker/chapter03.tre")
# plot <- ggtree::ggtree(chap3, layout = "circular", size = 0.5) + ggtree::geom_tiplab(size = 3)
# ggtree::ggsave(filename = "../plots/phyolo.png", plot = plot, device = png, width = 10, height = 10, units = "in", bg = "transparent", dpi = 500, scale = 1.5)
# ggtree::gheatmap(plot)

htree <- max(phytools::nodeHeights(chap3)) # timescale of the tree
png("../plots/phyolo-phytools.png", width = 8000, height = 8000, units = "px", res = 300)
plot <- phytools::plotTree(chap3, ftype = "i", fsize = 1.4, type = "fan", lwd = 1, part = 0.98)
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1.75, labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-16, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1.5, col = "red")
text(x = 250, y = -35, labels = "Time (Million years)", cex = 1.5, col = "red")
dev.off()
