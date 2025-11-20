# https://jhudatascience.org/AnVIL_Phylogenetic-Techniques/ancestral-state-reconstruction.html
# http://www.phytools.org/eqg/Exercise_5.2/

library("ape")
library("phytools")


# "F00727" - SRL, "F00709" - RTD


# did not do root order based trait normalizations :(
rtd_srl <- read.csv("../data/chapter2/FREDv3subset/RTD_SRL_species_means.csv", row.names = "binominal") # average RTD and SRL trait values for the 203 species
tree <- ape::read.tree("../data/chapter2/uphylomaker/fredv3subset.tre") # phylogenetic tree of the 203 species

# these are mean RTD, SRL values in the same order as the tip labels of the phylogenetic tree
# tip.labels have underscores in-between genus name and specific epithet :/
tip_labels <- tree$tip.label
named_rtd_vec <- setNames(rtd_srl[gsub(pattern = "_", replacement = " ", x = tip_labels), ]$F00709, tip_labels)
named_srl_vec <- setNames(rtd_srl[gsub(pattern = "_", replacement = " ", x = tip_labels), ]$F00727, tip_labels)

# ancestral state reconstruction
astate_rtd <- phytools::fastAnc(tree = tree, x = named_rtd_vec, CI = TRUE)
astate_srl <- phytools::fastAnc(tree = tree, x = named_srl_vec, CI = TRUE)

png("../plots/asrRTD.png", width = 8000, height = 8000, units = "px", res = 300)
rtd_map <- phytools::contMap(tree = tree, x = named_rtd_vec, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(rtd_map, type = "fan")
dev.off()

png("../plots/asrSRL.png", width = 8000, height = 8000, units = "px", res = 300)
rtd_map <- phytools::contMap(tree = tree, x = named_srl_vec, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(rtd_map, type = "fan")
dev.off()



# how to handle polytomies in ancestral state reconstruction - https://blog.phytools.org/2015/06/update-to-rerootingmethod-for-ancestral.html
# Phylogenetically Independent Contrasts (PIC)
ape::pic(x = log(named_rtd_vec), phy = tree) # Error - 'phy' is not rooted and fully dichotomous - we do have polytomies in the tree :/

ape::is.binary(tree) # FALSE!!!
ape::is.binary(ape::multi2di(tree)) # TRUE - that's how you convert a tree with polytomies into one without

# https://www.mail-archive.com/r-sig-phylo@r-project.org/msg01363.html
dichot <- ape::multi2di(tree) # dichotomized tree
ape::pic(x = log(named_rtd_vec), phy = dichot) # works :)

# if we try to fit a linear regression (Ordinary Least Squares) model between the mean SRL and RTD values for the 203 species without accounting for phylogenetic relationships
crude_lm <- lm(log(named_srl_vec) ~ log(named_rtd_vec))
summary(crude_lm)
# plot the data on a log transformed axis!!!
par(mar = c(1, 1, 1, 1))
plot(named_rtd_vec, named_srl_vec, xlab = "RTD", ylab = "SRL", log = "xy") # welp
lines(named_rtd_vec, exp(predict(crude_lm)), lwd = 1, col = "red") # predict(model) gives the predictions without passing the inputs??? - strange that the model keeps it memorized???
# exp is exponentiation - to undo the log transformation - since out plot axis will also do a log transformation, we pass the raw predictions to avoid -> log(log(predictions))
