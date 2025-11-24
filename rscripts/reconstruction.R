# REFERENCES
#-------------

# https://jhudatascience.org/AnVIL_Phylogenetic-Techniques/ancestral-state-reconstruction.html
# http://www.phytools.org/eqg/Exercise_5.2/
# Revell, L.J. and Harmon, L.J. (2022) Phylogenetic comparative methods in R. Princeton Oxford: Princeton University Press.

library("ape")
library("phytools")
library("U.PhyloMaker")
library("nlme")

#--------------------------------------------------------------------------------------------------------------------------------------
# HYPOTHESIS 01 - EVOLUTIONARY HISTORY OF COLLABORATION AXIS TRAITS (E.G. SRL) AND CONSERVATION AXIS TRAITS (E.G. RTD) ARE INDEPENDENT
#--------------------------------------------------------------------------------------------------------------------------------------

rtd_srl <- read.csv("../data/chapter2/FREDv3subset/RTD_SRL_species_means.csv", row.names = "binominal")
# contains crude species averages of RTD and SRL for the 203 species - did not do root order based trait normalizations :(
# F00727 - SRL, F00709 - RTD
tree <- ape::read.tree("../data/chapter2/uphylomaker/fredv3subset.tre") # phylogenetic tree of the 203 species in the above subset

tip_labels <- tree$tip.label # tip.labels have underscores in-between genus names and specific epithets
# phytools:: functions expect the passed trait values to be a named vector, with names matching that of the phylogenetic tree's tip labels
named_rtd_vec <- setNames(rtd_srl[gsub(pattern = "_", replacement = " ", x = tip_labels), ]$F00709, tip_labels)
named_srl_vec_hypo_1 <- setNames(rtd_srl[gsub(pattern = "_", replacement = " ", x = tip_labels), ]$F00727, tip_labels)

# ancestral state reconstruction of root tissue density (conservation axis)
png("../plots/asr_rtd.png", width = 8000, height = 8000, units = "px", res = 300)
# phytools::contMap() does the reconstruction internally before plotting
map <- phytools::contMap(tree = tree, x = named_rtd_vec, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
htree <- max(phytools::nodeHeights(tree)) # timescale of the tree
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1., labels = FALSE, col = "black")
text(x = tscale_axis, y = rep(-10, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1, col = "black")
text(x = 250, y = -30, labels = "Time (Million years)", cex = 1.5, col = "black")
dev.off()

# ancestral state reconstruction of specific root length (collaboration axis)
png("../plots/asr_srl.png", width = 8000, height = 8000, units = "px", res = 300)
map <- phytools::contMap(tree = tree, x = named_srl_vec_hypo_1, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1., labels = FALSE, col = "black")
text(x = tscale_axis, y = rep(-10, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1, col = "black")
text(x = 250, y = -30, labels = "Time (Million years)", cex = 1.5, col = "black")
dev.off()


# fit a linear regression (Ordinary Least Squares) model between the mean SRL and RTD values for the 203 species without accounting for phylogenetic relationships
# models for lm() are specified symbolically. a typical model has the form response ~ terms
# where response is the (numeric) response vector and terms is a series of terms which specifies a linear predictor for response
crude_lm <- lm(log(named_srl_vec_hypo_1)~log(named_rtd_vec)) # response ~ predictor
crude_lm |> summary()

par(mar = c(5, 5, 1, 1))
png("../plots/rtd_srl.png", width = 5000, height = 5000, units = "px", res = 500)
plot(x = log(named_rtd_vec), y = log(named_srl_vec_hypo_1), xlab = "log(RTD)", ylab = "log(SRL)")#, log = "xy")
# log = "xy" TRANSFORMS THE DATA BUT PLOTS THEM ON A REGULAR AXIS
# plot(x, y, log = "xy") and plot(log(x), log(y)) WILL NOT GIVE INDENTICAL RESULTS!!!!!
lines(x = log(named_rtd_vec), y = predict(crude_lm), lwd = 1, col = "red") # predict(model) gives the predictions without passing the inputs??? - strange that the model keeps it memorized???
mtext(paste0("y = ", round(coef(crude_lm)[2], 2), " x + ", round(coef(crude_lm)[1], 2)), side = 3, col = "red", line = -2)
dev.off()
# the output of predict(model) will already be log transformed because we fit the model with log transformed data so exp() (exponentiation) is used to undo the log transformation - because out plotting axis will also do a log transformation
# we do not want log(log(predictions))


# METHOD 1 - Phylogenetically Independent Contrasts (PIC)
#---------------------------------------------------------

# now to a model that accounts for phylogenetic relatedness, using PICs
ape::pic(x = log(named_rtd_vec), phy = tree) # Error - 'phy' is not rooted and fully dichotomous
# We do have polytomies in the tree :(

ape::is.binary(tree) # FALSE!!!
ape::is.binary(ape::multi2di(tree)) # TRUE
# ape::multi2di() transforms polytomies into dichotomies with branch length 0

# https://www.mail-archive.com/r-sig-phylo@r-project.org/msg01363.html
dichot <- ape::multi2di(tree) # dichotomized phylogenetic tree
htree <- max(phytools::nodeHeights(dichot)) # timescale of the tree
pic_rtd <- ape::pic(x = log(named_rtd_vec), phy = dichot) # PICs for RTD
pic_srl <- ape::pic(x = log(named_srl_vec_hypo_1), phy = dichot) # PICs for SRL

# when we fit this to lm again, we need to make sure that our regression line does not have an intercept (Revell and Harmon, 2022)
# this is because the position of right and left nodes is arbitrary for all nodes in our phylogeny, so is the direction of the subtraction of the PICs
# so the model shoud go through the origin (0, 0)
pic_lm <- lm(pic_srl~pic_rtd+0) # + 0 is used to specify that we do not want an intercept => WE ASK THE MODEL BE IN Y = MX FORMAT INSTEAD OF Y = MX + C
pic_lm |> summary() # statistical summary for the model - accounting for phylogenetic relationships

par(mar = c(5, 5, 1, 1))
png("../plots/pic_rtd_srl.png", width = 5000, height = 5000, units = "px", res = 500)
plot(x = pic_rtd, y = pic_srl, xlab = "ape::pic(log(RTD))", ylab = "ape::pic(log(SRL))")
abline(h = 0, lty = "dotted")
abline(v = 0, lty = "dotted")
abline(pic_lm, lwd = 1, col = "red")
mtext(paste0("y = ", coef(pic_lm), " x"), side = 3, col = "red", line = -2)
dev.off()

# Phylomorphospace
#-------------------

par(mar = c(5, 5, 1, 1))
png("../plots/phylomorphospace_rtd_srl.png", width = 10000, height = 10000, units = "px", res = 800)
phytools::phylomorphospace(tree = dichot, X = cbind(log(named_rtd_vec), log(named_srl_vec_hypo_1)), xlab = "log(RTD)", ylab = "log(SRL)", label = "off", node.size = c(0, 0), log = "xy", xlim = c(0, max(log(named_rtd_vec))))
points(x = named_rtd_vec, y = named_srl_vec_hypo_1, pch = 21)
grid()
abline(lm(log(named_srl_vec_hypo_1)~log(named_rtd_vec)), lwd = 2, col = "red")
dev.off()


# METHOD 2 - Phylogenetic Generalized Least Squares (PGLS)
#-----------------------------------------------------------

# column order in rtd_srl is F00727 F00709
corr_matrix <- ape::corBrownian(phy = tree) # the form argument is used to specify the order of our data
# since we have vectors with data in the same order as the tree here, it's not necessary
pgls <- nlme::gls(log(named_srl_vec_hypo_1)~log(named_rtd_vec), correlation = corr_matrix)
pgls |> summary()

par(mar = c(5, 5, 1, 1))
png("../plots/pgls_rtd_srl.png", width = 5000, height = 5000, units = "px", res = 500)
plot(log(named_rtd_vec), log(named_srl_vec_hypo_1), xlab = "log(RTD)", ylab = "log(SRL)")
abline(pgls, lwd = 2, col = "red")
mtext(paste0("y = ", round(coef(pgls)[2], 2), " x + ", round(coef(pgls)[1], 2)), side = 3, col = "red", line = -2)
dev.off()

pgls |> coef() # coefficients of the PGLS model
pic_lm |> coef() # coefficients of the PIC OLS model
# the difference between the coefficients of these two models is negligible
abs(coef(pgls)[2] - coef(pic_lm)[1])




#-------------------------------------------------------------------------------------------------------------------
# HYPOTHESES 02 - CORRELATION BETWEEN THE EVOLUTIONARY HISTORY OF MYCORRHIZAL STATES AND COLLABORATION AXIS TRAITS
#-------------------------------------------------------------------------------------------------------------------

# F00679 - RD, F00727 - SRL, F00645 - mycorrhizal state
collab_states_n_species_avg_traits <- read.csv("../data/chapter2/FREDv3subset/FRED_subset_collab_states_n_species_avg_traits.csv", sep = ",", row.names = "binominal", stringsAsFactors = TRUE)
# this contains crude species avaraged records for RD and SRL (did not consider root order differences)

# FIRST TIME PHYLOGENETIC TREE CREATION AND SERIALIZATION
#---------------------------------------------------------

# we'll need a new phylogeny as this is a superset of the previous phylogeny
# megatree <- ape::read.tree("../data/chapter2/uphylomaker/GBOTB_extended_WP.tre")
# genus_family_relations <- read.csv("../data/chapter2/uphylomaker/plant_genus_list.csv", sep = ",")
# we need a dataframe with columns - species,genus,family,species.relative,genus.relative for the species that we are interested in
# species_of_interest <- data.frame(species=collab_states_n_species_avg_traits$binominal, genus=collab_states_n_species_avg_traits$F01286, family=NA, species.relative=NA, genus.relative=NA)
# runtime <- Sys.time()
# phylogeny <- U.PhyloMaker::phylo.maker(sp.list = species_of_interest, tree = megatree, gen.list = genus_family_relations)
# runtime <- Sys.time() - runtime # 2.86052 mins
# serialize the phylogeny
# ape::write.tree(phy = phylogeny$phylo, file = "../data/chapter2/uphylomaker/fredv3subset_collab_trait_n_states.tre") # cool

# DOWNSTREAM ANALYSES USING THE SERIALIZED PHYLOGENETIC TREE
#-----------------------------------------------------------

# read in the previously serialized phylogenetic tree
collab_phylo <- ape::read.tree(file = "../data/chapter2/uphylomaker/fredv3subset_collab_trait_n_states.tre")

# plot and save the phylogenetic tree
htree <- max(phytools::nodeHeights(collab_phylo))
png("../plots/phylo_collab_states_n_traits.png", width = 10000, height = 10000, units = "px", res = 300)
plot <- phytools::plotTree(collab_phylo, ftype = "i", fsize = 1.2, type = "fan", lwd = 1, part = 0.99)
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1.75, labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-16, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1.5, col = "red")
text(x = 250, y = -35, labels = "Time (Million years)", cex = 1.5, col = "red")
dev.off()

rooted_dichotomous_phylogeny <- ape::multi2di(collab_phylo)
htree <- max(phytools::nodeHeights(rooted_dichotomous_phylogeny)) # timescale of the tree

# create named trait vectors with species order identical to the phylogeny - PHYLOGENY HAS UNDERSCORES BETWEEN THE GENUS NAME AND SPECIFIC EPITHET TF???
named_mycorrhizal_state_vec <- setNames(collab_states_n_species_avg_traits[rooted_dichotomous_phylogeny$tip.label |> gsub(pattern='_', replacement=' '), ]$F00645, nm = rooted_dichotomous_phylogeny$tip.label)
# do not confuse this with named_srl_vec_hypo_1
named_srl_vec_hypo_2 <- setNames(collab_states_n_species_avg_traits[rooted_dichotomous_phylogeny$tip.label |> gsub(pattern='_', replacement=' '), ]$F00727, nm = rooted_dichotomous_phylogeny$tip.label)
named_rd_vec <- setNames(collab_states_n_species_avg_traits[rooted_dichotomous_phylogeny$tip.label |> gsub(pattern='_', replacement=' '), ]$F00679, nm = rooted_dichotomous_phylogeny$tip.label)

png("../plots/asr_collab_rd.png", width = 8000, height = 8000, units = "px", res = 300)
map <- phytools::contMap(tree = rooted_dichotomous_phylogeny, x = named_rd_vec, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1., labels = FALSE, col = "black")
text(x = tscale_axis, y = rep(-10, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1, col = "black")
text(x = 250, y = -30, labels = "Time (Million years)", cex = 1.5, col = "black")
dev.off()

png("../plots/asr_collab_srl.png", width = 8000, height = 8000, units = "px", res = 300)
map <- phytools::contMap(tree = rooted_dichotomous_phylogeny, x = named_srl_vec_hypo_2, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1., labels = FALSE, col = "black")
text(x = tscale_axis, y = rep(-10, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1, col = "black")
text(x = 250, y = -30, labels = "Time (Million years)", cex = 1.5, col = "black")
dev.off()

ape::is.binary(collab_phylo) # FALSE
ape::is.binary(rooted_dichotomous_phylogeny) # TRUE

# ape::ace - Ancestral Character Estimation - use model = "ER" & type = "discrete" for discrete categorical traits
# re rooting is a method used to reconstruct ancestral states of discrete categorical traits
runtime <- Sys.time()
er_mystates <- phytools::rerootingMethod(x = named_mycorrhizal_state_vec, tree = rooted_dichotomous_phylogeny, model = "ER")
runtime <- Sys.time() - runtime

# map the discrete character state evolution onto the phylogeny
myco_state_colours <- c("red", "green", "yellow", "orange", "lightblue", "purple")
png("../plots/asr_collab_myco_states.png", width = 12000, height = 12000, units = "px", res = 300)
plot <- phytools::plotTree(rooted_dichotomous_phylogeny, ftype = "i", fsize = 1.2, type = "fan", lwd = 1, part = 0.99, offset = 3)
# label the internal nodes
ape::nodelabels(node = er_mystates$marginal.anc |> row.names() |> as.numeric(), pie = er_mystates$marginal.anc, piecol = myco_state_colours, cex = 0.1)
ape::tiplabels(pie = to.matrix(named_mycorrhizal_state_vec, sort(unique(named_mycorrhizal_state_vec))), piecol = myco_state_colours, cex = 0.1) # label the leaf nodes
legend("topright", legend = sort(unique(named_mycorrhizal_state_vec)), pt.bg = myco_state_colours, cex = 3, pt.cex = 5, pch = 21)
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1., labels = FALSE, col = "black")
text(x = tscale_axis, y = rep(-10, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1, col = "black")
text(x = 250, y = -30, labels = "Time (Million years)", cex = 1.5, col = "black")
dev.off()


# TRY OVERLAYING THE MYCORRHIZAL STATE PHYLOGENY ON THE RD & SRL PHYLOGENIES
png("../plots/asr_collab_myco_states_n_rd.png", width = 12000, height = 12000, units = "px", res = 400)
map <- phytools::contMap(tree = rooted_dichotomous_phylogeny, x = named_rd_vec, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
ape::nodelabels(node = er_mystates$marginal.anc |> row.names() |> as.numeric(), pie = er_mystates$marginal.anc, piecol = myco_state_colours, cex = 0.1)
ape::tiplabels(pie = to.matrix(named_mycorrhizal_state_vec, sort(unique(named_mycorrhizal_state_vec))), piecol = myco_state_colours, cex = 0.1)
legend("topright", legend = sort(unique(named_mycorrhizal_state_vec)), pt.bg = myco_state_colours, cex = 3, pt.cex = 5, pch = 21)
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1., labels = FALSE, col = "black")
text(x = tscale_axis, y = rep(-10, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1, col = "black")
text(x = 250, y = -30, labels = "Time (Million years)", cex = 1.5, col = "black")
dev.off()

png("../plots/asr_collab_myco_states_n_srl.png", width = 12000, height = 12000, units = "px", res = 400)
map <- phytools::contMap(tree = rooted_dichotomous_phylogeny, x = named_srl_vec_hypo_2, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
ape::nodelabels(node = er_mystates$marginal.anc |> row.names() |> as.numeric(), pie = er_mystates$marginal.anc, piecol = myco_state_colours, cex = 0.1)
ape::tiplabels(pie = to.matrix(named_mycorrhizal_state_vec, sort(unique(named_mycorrhizal_state_vec))), piecol = myco_state_colours, cex = 0.1)
legend("topright", legend = sort(unique(named_mycorrhizal_state_vec)), pt.bg = myco_state_colours, cex = 3, pt.cex = 5, pch = 21)
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1., labels = FALSE, col = "black")
text(x = tscale_axis, y = rep(-10, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1, col = "black")
text(x = 250, y = -30, labels = "Time (Million years)", cex = 1.5, col = "black")
dev.off()

# PHYLOGENETIC GENERALIZED ANCOVA
#---------------------------------

# since the second hypothesis looks at correlations between a categorical trait and two continuous traits, PIC followed by OLS regression won't help
# we opt for phylogenetic generalized ANCOVA as recommended by Revell, L.J. and Harmon, L.J. (2022) Phylogenetic comparative methods in R. Princeton Oxford: Princeton University Press. (page 71)

collab_data <- data.frame(binominal = as.factor(gsub(rownames(collab_states_n_species_avg_traits), pattern = ' ', replacement = '_')), # because phylogenetic tree has underscores inplace of spaces
                          rd = collab_states_n_species_avg_traits$F00679, srl = collab_states_n_species_avg_traits$F00727, myco = as.factor(collab_states_n_species_avg_traits$F00645))
row_indices <- match(rooted_dichotomous_phylogeny$tip.label, collab_data$binominal) # row indices according to the order of species in the phylogenetic tree
# match(vector to be matched, vector to be matched against)
all(collab_data$binominal[row_indices] == rooted_dichotomous_phylogeny$tip.label) # good :)
collab_data <- collab_data[row_indices, ]
collab_data <- collab_data[rooted_dichotomous_phylogeny$tip.label, ] # reorder the dataframe based on the species order in the phylogenetic tree
all(collab_data$binominal == rooted_dichotomous_phylogeny$tip.label) # double checking

states_n_colours <- setNames(myco_state_colours, nm = sort(unique(collab_data$myco))) # mycorrhizal state - colour lookup table for plotting
dummy_rds <- seq(min(collab_data$rd), max(collab_data$rd), length.out = 100) # dummy root diameter values - independent variable

# to see how the form argument actually works look up https://www.mail-archive.com/search?l=r-sig-phylo@r-project.org&q=subject:%22Re%5C%3A+%5C%5BR%5C-sig%5C-phylo%5C%5D+How+to+sort+trait+data+according+to+tree%22&o=newest&f=1
corr_matrix <- ape::corBrownian(phy = rooted_dichotomous_phylogeny, form = ~binominal)
# form argument is used to pass the order (species names) in the data (trait values) (page 65)
ancova <- nlme::gls(log(srl)~log(rd)+myco, data = as.data.frame(collab_data), correlation = corr_matrix)
anova(ancova)

# plot the results
png("../plots/phylogenetic_ancova_rd_srl_mystates.png", width = 5000, height = 5000, units = "px", res = 400)
plot(x = collab_data$rd, y = collab_data$srl, pch = 21, cex = 1, xlab = "RD", ylab = "SRL", bg = states_n_colours[collab_data$myco])
legend("topright", legend = names(states_n_colours), pt.bg = myco_state_colours, cex = 1, pt.cex = 1, pch = 21)

# plot model predictions for each mycorrhizal state
for (state in sort(unique(collab_data$myco))) {
    # Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]) :
    # contrasts can be applied only to factors with 2 or more levels
    # ====>> to avoid this specify stringsAsFactors = TRUE when loading in the trait dataset
    lines(x = dummy_rds, y = exp(predict(ancova, newdata = data.frame(rd = dummy_rds, myco = rep(state, 100)))), lwd = 2, col = states_n_colours[state])
}
dev.off()


