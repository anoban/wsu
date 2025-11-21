# https://jhudatascience.org/AnVIL_Phylogenetic-Techniques/ancestral-state-reconstruction.html
# http://www.phytools.org/eqg/Exercise_5.2/
# Revell, L.J. and Harmon, L.J. (2022) Phylogenetic comparative methods in R. Princeton Oxford: Princeton University Press.

library("ape")
library("phytools")
library("U.PhyloMaker")

# did not do root order based trait normalizations :(
rtd_srl <- read.csv("../data/chapter2/FREDv3subset/RTD_SRL_species_means.csv", row.names = "binominal") # average RTD and SRL trait values for the 203 species
# "F00727" - SRL, "F00709" - RTD
tree <- ape::read.tree("../data/chapter2/uphylomaker/fredv3subset.tre") # phylogenetic tree of the 203 species

#---------------------------------------------------------------------
# HYPOTHESIS 01 - EVOLUTIONARY HISTORY OF SRL AND RTD ARE INDEPENDENT
#---------------------------------------------------------------------

# these are mean RTD, SRL values in the same order as the tip labels of the phylogenetic tree
# tip.labels have underscores in-between genus name and specific epithet :/
tip_labels <- tree$tip.label
# phytools functions expect the passed trait values to be a named vector, so ....
named_rtd_vec <- setNames(rtd_srl[gsub(pattern = "_", replacement = " ", x = tip_labels), ]$F00709, tip_labels)
named_srl_vec <- setNames(rtd_srl[gsub(pattern = "_", replacement = " ", x = tip_labels), ]$F00727, tip_labels)

png("../plots/asr_RTD.png", width = 8000, height = 8000, units = "px", res = 300)
map <- phytools::contMap(tree = tree, x = named_rtd_vec, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
dev.off()

png("../plots/asr_SRL.png", width = 8000, height = 8000, units = "px", res = 300)
map <- phytools::contMap(tree = tree, x = named_srl_vec, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
dev.off()


# if we try to fit a linear regression (Ordinary Least Squares) model between the mean SRL and RTD values for the 203 species without accounting for phylogenetic relationships
crude_lm <- lm(log(named_srl_vec) ~ log(named_rtd_vec))
summary(crude_lm)
# plot the data on a log transformed axis!!!
par(mar = c(5, 5, 1, 1))
plot(x = named_srl_vec, y = named_rtd_vec,  xlab = "SRL", ylab = "RTD", log = "xy") # welp
lines(x = named_srl_vec, y = exp(predict(crude_lm)), lwd = 1, col = "red") # predict(model) gives the predictions without passing the inputs??? - strange that the model keeps it memorized???
# exp is exponentiation - to undo the log transformation - since out plot axis will also do a log transformation, we pass the raw predictions to avoid -> log(log(predictions))

# now to a model that accounts for phylogenetic relatedness, using Phylogenetically Independent Contrasts (PIC)
ape::pic(x = log(named_rtd_vec), phy = tree) # Error - 'phy' is not rooted and fully dichotomous\
# We do have polytomies in the tree :(

ape::is.binary(tree) # FALSE!!!
ape::is.binary(ape::multi2di(tree)) # TRUE - that's how you convert a tree with polytomies into one without

# https://www.mail-archive.com/r-sig-phylo@r-project.org/msg01363.html
dichot <- ape::multi2di(tree) # dichotomized tree
pic_rtd <- ape::pic(x = log(named_rtd_vec), phy = dichot) # PICs for RTD
pic_srl <- ape::pic(x = log(named_srl_vec), phy = dichot) # PICs for SRL

# when we fit this to lm again, we need to make sure that our regression line does not have an intercept (Revell and Harmon, 2022)
# this is because the position of right and left nodes is arbitrary for all nodes in our phylogeny, so is the direction of the subtraction of the PICs
# so the model shoud go through the origin (0, 0)
pic_lm <- lm(pic_srl~pic_rtd+0) # + 0 is used to specify that we do not want an intercept => WE ASK THE MODEL BE IN Y = MX FORMAT INSTEAD OF Y = MX + C
summary(pic_lm)

par(mar = c(5, 5, 1, 1))
plot(pic_rtd, pic_srl, xlab = "ape::pic(log(RTD))", ylab = "ape::pic(log(SRL))")
abline(h = 0, lty = "dotted")
abline(v = 0, lty = "dotted")
abline(pic_lm, lwd = 2, col = "red")


#-------------------------------------------------------------------------------------------------------------------
# HYPOTHESES 02 - CORRELATION BETWEEN THE EVOLUTIONARY HISTORY OF MYCORRHIZAL STATES AND COLLABORATION AXIS TRAITS
#-------------------------------------------------------------------------------------------------------------------

# F00679 - RD, F00727 - SRL, F00645 - mycorrhizal state

collab_states_n_species_svg_traits <- read.csv("../data/chapter2/FREDv3subset/FRED_subset_collab_states_n_species_avg_traits.csv", sep = ",")

# we'll need a new phylogeny as this is a superset of the previous phylogeny
megatree <- ape::read.tree("../data/chapter2/uphylomaker/GBOTB_extended_WP.tre")
genus_family_relations <- read.csv("../data/chapter2/uphylomaker/plant_genus_list.csv", sep = ",")
# we need a dataframe with columns - species,genus,family,species.relative,genus.relative for the species that we are interested in
species_of_interest <- data.frame(species=collab_states_n_species_svg_traits$binominal, genus=collab_states_n_species_svg_traits$F01286, family=NA, species.relative=NA, genus.relative=NA)
runtime <- Sys.time()
phylogeny <- U.PhyloMaker::phylo.maker(sp.list = species_of_interest, tree = megatree, gen.list = genus_family_relations)
runtime <- Sys.time() - runtime # 2.86052 mins
# serialize the phylogeny
ape::write.tree(phy = phylogeny$phylo, file = "../data/chapter2/uphylomaker/fredv3subset_collab_trait_n_states.tre") # cool

# save the phylogenetic tree
htree <- max(phytools::nodeHeights(phylogeny$phylo))
png("../plots/phylo_collab_states_n_traits.png", width = 10000, height = 10000, units = "px", res = 300)
plot <- phytools::plotTree(phylogeny$phylo, ftype = "i", fsize = 1.2, type = "fan", lwd = 1, part = 0.99)
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1.75, labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-16, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1.5, col = "red")
text(x = 250, y = -35, labels = "Time (Million years)", cex = 1.5, col = "red")
dev.off()

rooted_dichotomous_phylogeny <- ape::multi2di(phylogeny$phylo)

# create named trait vectors with species order identical to the phylogeny - PHYLOGENY HAS UNDERSCORES BETWEEN THE GENUS NAME AND SPECIFIC EPITHET TF???
named_mycorrhizal_state_vec <- setNames(collab_states_n_species_svg_traits$F00645, nm = collab_states_n_species_svg_traits$binominal |> gsub(pattern=' ', replacement='_'))
# do not confuse this with named_srl_vec
named_srl_vec_0 <- setNames(collab_states_n_species_svg_traits$F00727, nm = collab_states_n_species_svg_traits$binominal |> gsub(pattern=' ', replacement='_'))
named_rd_vec <- setNames(collab_states_n_species_svg_traits$F00679, nm = collab_states_n_species_svg_traits$binominal |> gsub(pattern=' ', replacement='_'))

png("../plots/asr_collab_RD.png", width = 8000, height = 8000, units = "px", res = 300)
map <- phytools::contMap(tree = rooted_dichotomous_phylogeny, x = named_rd_vec, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
dev.off()

png("../plots/asr_collab_SRL.png", width = 8000, height = 8000, units = "px", res = 300)
map <- phytools::contMap(tree = rooted_dichotomous_phylogeny, x = named_srl_vec_0, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
dev.off()

ape::is.binary(phylogeny$phylo) # FALSE
ape::is.binary(rooted_dichotomous_phylogeny) # TRUE

png("../plots/asr_collab_states_n_traits.png", width = 10000, height = 10000, units = "px", res = 300)
# ape::ace - Ancestral Character Estimation - use model = "ER" & type = "discrete" for discrete categorical traits
er_mystates <- phytools::rerootingMethod(x = named_mycorrhizal_state_vec, tree = rooted_dichotomous_phylogeny, model = "ER")
phytools::plotTree(rooted_dichotomous_phylogeny, ftype = "i", fsize = 1.2, type = "fan", lwd = 1, part = 0.99)
phytools::nodelabels(node = er_mystates$marginal.anc |> row.names() |> as.numeric(), pie = er_mystates$marginal.anc, piecol = c("red", "green", "yellow", "orange", "blue", "brown"))
