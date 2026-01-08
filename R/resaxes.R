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


# this is species averages of first order records for the below mentioned 4 root traits from the RES
res_traits <- read.csv("../data/chapter2/FREDv3subset/FRED_subset_ord1_cont_RES_traits.csv", row.names = "binominal")
stopifnot(length(rownames(res_traits))==length(unique(rownames(res_traits))))
# F00679 	Root diameter (mm)
# F00727 	Specific root length (SRL) (m/g)
# F00261 	Root N content 	(mg/g)
# F00709 	Root tissue density (RTD) (g/cm3)
colnames(res_traits)[5:8] <- c("RD", "SRL", "RN", "RTD")

# construct the phylogenetic tree for the species in the RES trait dataset
# genus_family_relations <- read.csv("../data/chapter2/uphylomaker/plant_genus_list.csv", sep = ",") # data from the UPhyloMaker library
# a dataset with columns species,genus,family,species.relative,genus.relative, for the species of interest WHERE everything beside the first two columns can be NA
# species_of_interest <- data.frame(list("species" = rownames(res_traits), "genus"=res_traits$F01286, "family"=NA, "species.relative"=NA, "genus.relative"=NA))
# megatree <- ape::read.tree("../data/chapter2/uphylomaker/GBOTB_extended_WP.tre")
# phylogeny <- U.PhyloMaker::phylo.maker(sp.list = species_of_interest, tree = megatree, gen.list = genus_family_relations) # this took forfuckingever ~ 3 minutes
# ape::write.tree(phy = phylogeny$phylo, file = "../data/chapter2/uphylomaker/FRED_subset_ord1_cont_RES_traits.tre")

tree <- ape::read.tree("../data/chapter2/uphylomaker/FRED_subset_ord1_cont_RES_traits.tre")
stopifnot(length(tree$tip.label)==length(rownames(res_traits)))

# update the row names to match the names in the tree (i.e with underscores between generic name and specific epithet)
rownames(res_traits) <- gsub(pattern = " ", replacement = "_", x = rownames(res_traits))

matched_row_indices <- match(tree$tip.label, table = rownames(res_traits)) # make sure the rows in the datset are ordered in the same way as the tip labels of the phylogeny
stopifnot(all(tree$tip.label==rownames(res_traits)[matched_row_indices]))

# now reorder our dataset
res_traits <- res_traits[matched_row_indices, ]
stopifnot(all(tree$tip.label==rownames(res_traits)))

# fit a linear regression (Ordinary Least Squares) model for the species averaged first order RES traits without accounting for phylogenetic relationships
par(mar = c(5, 5, 1, 1), mfrow=c(2, 2))
for (collab in c("RD", "SRL")) {
    for (conserv in c("RN", "RTD")) {
        mod <- lm(log(res_traits[, collab])~log(res_traits[, conserv])) # collaboration axis trait is fit as the response variable while the conservation axis trait was fit as the predictor
        # does this matter here????
        plot(x = log(res_traits[, conserv]), # response
             y = log(res_traits[, collab]), # predictor
             xlab = paste0("log(", conserv, ")"), ylab = paste0("log(", collab, ")"))#, log = "xy")
        # log = "xy" TRANSFORMS THE DATA BUT PLOTS THEM ON A REGULAR AXIS
        # plot(x, y, log = "xy") and plot(log(x), log(y)) WILL NOT GIVE INDENTICAL RESULTS!!!!!
        lines(x = log(res_traits[, conserv]), y = predict(mod), lwd = 1, col = "red") # predict(model) gives the predictions without passing the inputs??? - strange that the model keeps it memorized???
        mtext(paste0("y = ", round(coef(mod)[2], 2), " x + ", round(coef(mod)[1], 2)), side = 3, col = "red", line = -2)
        # the output of predict(model) will already be log transformed because we fit the model with log transformed data so exp() (exponentiation) is used to undo the log transformation - because out plotting axis will also do a log transformation
        # we do not want log(log(predictions))
    }
}

#---------------------------------------------------------
# Phylogenetically Independent Contrasts (PIC)
#---------------------------------------------------------

# now to a model that accounts for phylogenetic relatedness, using PICs
# https://www.mail-archive.com/r-sig-phylo@r-project.org/msg01363.html
# we may have polytomies in the tree :(
if (!ape::is.binary(tree)) tree <- ape::multi2di(tree) # ape::multi2di() transforms polytomies into dichotomies with branch length 0
stopifnot(ape::is.binary(tree))

pic_data <- data.frame(list("RD"=ape::pic(log(res_traits$RD), phy = tree), "SRL"=ape::pic(log(res_traits$RD), phy = tree),
                                                              "RN"=ape::pic(log(res_traits$RD), phy = tree), "RTD"=ape::pic(log(res_traits$RD), phy = tree)))

# when we fit PIC scores to lm, we need to make sure that our regression line does not have an intercept (Revell and Harmon, 2022)
# this is because the position of right and left nodes is arbitrary for all nodes in our phylogeny, so is the direction of the subtraction of the PICs
# so the model shoud go through the origin (0, 0)
par(mar = c(5, 5, 1, 1), mfrow=c(2, 2))
for (collab in c("RD", "SRL")) {
    for (conserv in c("RN", "RTD")) {
        picmod <- lm(pic_data[, collab]~pic_data[, conserv]+0)# + 0 is used to specify that we do not want an intercept => WE ASK THE MODEL BE IN Y = MX FORMAT INSTEAD OF Y = MX + C
        print(summary(picmod))
        plot(x = pic_data[, conserv], # response
             y = pic_data[, collab], # predictor
             xlab = paste0("PIC(log(", conserv, "))"), ylab = paste0("PIC(log(", collab, "))"))
        abline(h = 0, lty = "dotted")
        abline(v = 0, lty = "dotted")
        abline(picmod, lwd = 1, col = "red")
        mtext(paste0("y = ", round(coef(picmod)[2], 2), " x + ", round(coef(picmod)[1], 2)), side = 3, col = "red", line = -2)
    }
}


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

# column order in res_traits is F00727 F00709
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



#------------------
# PLOTS
#------------------

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
