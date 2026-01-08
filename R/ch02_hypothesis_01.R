# REFERENCES
#-------------

# https://jhudatascience.org/AnVIL_Phylogenetic-Techniques/ancestral-state-reconstruction.html
# http://www.phytools.org/eqg/Exercise_5.2/
# Revell, L.J. and Harmon, L.J. (2022) Phylogenetic comparative methods in R. Princeton Oxford: Princeton University Press.
# https://github.com/EEOB-Macroevolution/Practicals/blob/main/Phylo_Regression/scripts/PhyloRegressionContinuous.r

library("ape")
library("maps")
library("phytools")
library("nlme")
library("U.PhyloMaker")


#--------------------------------------------------------------------------------------------------------------------------------------
# HYPOTHESIS 01 - EVOLUTIONARY HISTORY OF COLLABORATION AXIS TRAITS (E.G. SRL) AND CONSERVATION AXIS TRAITS (E.G. RTD) ARE INDEPENDENT
#--------------------------------------------------------------------------------------------------------------------------------------

rtd_srl <- read.csv("../data/chapter2/FREDv3subset/FRED_subset_collab_states_n_species_avg_traits.csv", row.names = "binominal")
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
mtext(paste0("y = ", round(coef(pic_lm), 2), " x"), side = 3, col = "red", line = -2)
dev.off()

# Phylomorphospace
#-------------------

# TODO - redo this!!!!

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




