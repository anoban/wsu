#---------------
# REFERENCES
#---------------

# https://jhudatascience.org/AnVIL_Phylogenetic-Techniques/ancestral-state-reconstruction.html
# http://www.phytools.org/eqg/Exercise_5.2/
# Revell, L.J. and Harmon, L.J. (2022) Phylogenetic comparative methods in R. Princeton Oxford: Princeton University Press.
# https://github.com/EEOB-Macroevolution/Practicals/blob/main/Phylo_Regression/scripts/PhyloRegressionContinuous.r

library("ape")
library("maps")
library("phytools")
library("nlme")
library("U.PhyloMaker")
library("corHMM")

#--------------------------------------------------------------------------------------------------------------------------------------
# HYPOTHESIS 01 - EVOLUTIONARY HISTORY OF COLLABORATION AXIS TRAITS (E.G. SRL) AND CONSERVATION AXIS TRAITS (E.G. RTD) ARE INDEPENDENT
#--------------------------------------------------------------------------------------------------------------------------------------

# this is species averages of first order records for the below mentioned 4 root traits from the RES
res_avgd_traits <- read.csv("../data/chapter2/FREDv3subset/FRED_subset_ord1_sp_avrgd_cont_RES_traits.csv", row.names = "binominal")
stopifnot(length(rownames(res_avgd_traits))==length(unique(rownames(res_avgd_traits))))
# F00679 	Root diameter (mm)
# F00727 	Specific root length (SRL) (m/g)
# F00261 	Root N content 	(mg/g)
# F00709 	Root tissue density (RTD) (g/cm3)
colnames(res_avgd_traits)[5:8] <- c("RD", "SRL", "RN", "RTD")

# construct the phylogenetic tree for the species in the RES trait dataset
# genus_family_relations <- read.csv("../data/chapter2/uphylomaker/plant_genus_list.csv", sep = ",") # data from the UPhyloMaker library
# a dataset with columns species,genus,family,species.relative,genus.relative, for the species of interest WHERE everything beside the first two columns can be NA
# species_of_interest <- data.frame(list("species" = rownames(res_avgd_traits), "genus"=res_avgd_traits$F01286, "family"=NA, "species.relative"=NA, "genus.relative"=NA))
# megatree <- ape::read.tree("../data/chapter2/uphylomaker/GBOTB_extended_WP.tre")
# phylogeny <- U.PhyloMaker::phylo.maker(sp.list = species_of_interest, tree = megatree, gen.list = genus_family_relations) # this took forfuckingever ~ 3 minutes
# ape::write.tree(phy = phylogeny$phylo, file = "../data/chapter2/uphylomaker/FRED_subset_ord1_cont_RES_traits.tre")

tree <- ape::read.tree("../data/chapter2/uphylomaker/FRED_subset_ord1_cont_RES_traits.tre")
stopifnot(length(tree$tip.label)==length(rownames(res_avgd_traits)))

# update the row names to match the names in the tree (i.e with underscores between generic name and specific epithet)
rownames(res_avgd_traits) <- gsub(pattern = " ", replacement = "_", x = rownames(res_avgd_traits))

matched_row_indices <- match(tree$tip.label, table = rownames(res_avgd_traits)) # make sure the rows in the datset are ordered in the same way as the tip labels of the phylogeny
stopifnot(all(tree$tip.label==rownames(res_avgd_traits)[matched_row_indices]))

# now reorder our dataset
res_avgd_traits <- res_avgd_traits[matched_row_indices, ]
stopifnot(all(tree$tip.label==rownames(res_avgd_traits)))

# fit a linear regression (Ordinary Least Squares) model for the species averaged first order RES traits without accounting for phylogenetic relationships
png(filename = "../plots/linearreg_log_species_avg_trait_values.png", width = 10, height = 10, units = "in", res = 1000)
par(mar = c(5, 5, 1, 1), mfrow=c(2, 2))
for (collab in c("RD", "SRL")) {
    for (conserv in c("RN", "RTD")) {
        mod <- lm(log(res_avgd_traits[, collab])~log(res_avgd_traits[, conserv])) # collaboration axis trait is fit as the response variable while the conservation axis trait was fit as the predictor
        # does this matter here????
        plot(x = log(res_avgd_traits[, conserv]), # response
             y = log(res_avgd_traits[, collab]), # predictor
             xlab = paste0("log(", conserv, ")"), ylab = paste0("log(", collab, ")"))#, log = "xy")
        # log = "xy" TRANSFORMS THE DATA BUT PLOTS THEM ON A REGULAR AXIS
        # plot(x, y, log = "xy") and plot(log(x), log(y)) WILL NOT GIVE INDENTICAL RESULTS!!!!!
        lines(x = log(res_avgd_traits[, conserv]), y = predict(mod), lwd = 1, col = "red") # predict(model) gives the predictions without passing the inputs??? - strange that the model keeps it memorized???
        mtext(paste0("y = ", round(coef(mod)[2], 2), " x + ", round(coef(mod)[1], 2)), side = 3, col = "red", line = -2)
        # the output of predict(model) will already be log transformed because we fit the model with log transformed data so exp() (exponentiation) is used to undo the log transformation - because out plotting axis will also do a log transformation
        # we do not want log(log(predictions))
    }
}
dev.off()

#---------------------------------------------------------
# Phylogenetically Independent Contrasts (PIC)
#---------------------------------------------------------

# now to a model that accounts for phylogenetic relatedness, using PICs
# https://www.mail-archive.com/r-sig-phylo@r-project.org/msg01363.html
# we may have polytomies in the tree :(
if (!ape::is.binary(tree)) tree <- ape::multi2di(tree) # ape::multi2di() transforms polytomies into dichotomies with branch length 0
stopifnot(ape::is.binary(tree))

# REMEMBER THAT ape::pic EXPECTS THE TRAIT VECTOR TO BE ORDERD AS THE TREE'S TIP LABELS OR THE TRAIT VECTOR TO BE A NAMED VECTOR WITH NAMES MATCHING THE TREE'S TIP LABELS
pic_data <- data.frame(list("RD"=ape::pic(log(res_avgd_traits$RD), phy = tree), "SRL"=ape::pic(log(res_avgd_traits$SRL), phy = tree),
                        "RN"=ape::pic(log(res_avgd_traits$RN), phy = tree), "RTD"=ape::pic(log(res_avgd_traits$RTD), phy = tree)))

# when we fit PIC scores to lm, we need to make sure that our regression line does not have an intercept (Revell and Harmon, 2022)
# this is because the position of right and left nodes is arbitrary for all nodes in our phylogeny, so is the direction of the subtraction of the PICs
# so the model shoud go through the origin (0, 0)
png(filename = "../plots/linearreg_pic_species_avg_log_traitvals.png", width = 10, height = 10, units = "in", res = 1000)
par(mar = c(5, 5, 1, 1), mfrow=c(2, 2))
for (collab in c("RD", "SRL")) {
    for (conserv in c("RN", "RTD")) {
        picmod <- lm(pic_data[, collab]~pic_data[, conserv]+0)# + 0 is used to specify that we do not want an intercept => WE ASK THE MODEL BE IN Y = MX FORMAT INSTEAD OF Y = MX + C
        plot(x = pic_data[, conserv], # response
             y = pic_data[, collab], # predictor
             xlab = paste0("PIC(log(", conserv, "))"), ylab = paste0("PIC(log(", collab, "))"))
        abline(h = 0, lty = "dotted")
        abline(v = 0, lty = "dotted")
        abline(picmod, lwd = 1, col = "red")
        mtext(paste0("y = ", round(coef(picmod), 2), " x"), side = 3, col = "red", line = -2)
    }
}
dev.off()

# repeat this with ape::pic.ortho() which allows the dataset to have
res_traits <- read.csv("../data/chapter2/FREDv3subset/FRED_subset_ord1_cont_RES_traits.csv")#, row.names = "binominal") => may contains multiple records for species
colnames(res_traits)[6:9] <- c("RD", "SRL", "RN", "RTD")
res_traits$binominal <- gsub(pattern = " ", replacement = "_", x = res_traits$binominal)

stopifnot(length(res_traits$binominal)!=length(unique(res_traits$binominal))) # must contain duplicates which is accepted by ape::pic.ortho()

# since we have duplicates, it's critical to pass arguments to ape::pic.ortho as named vectors!!!!
# from the documentation of ape::pic.ortho
# the data x can be in two forms: a vector if there is a single observation for each species,
# or a list whose elements are vectors containing the individual observations for each species. These vectors may be of different lengths.
pic_ortho_data <- list("RD"=ape::pic.ortho(split(x = log(res_traits$RD), f = res_traits$binominal), phy = tree),
                       "SRL"=ape::pic.ortho(split(x = log(res_traits$SRL), f = res_traits$binominal), phy = tree),
                       "RN"=ape::pic.ortho(split(x = log(res_traits$RN), f = res_traits$binominal), phy = tree),
                       "RTD"=ape::pic.ortho(split(x = log(res_traits$RTD), f = res_traits$binominal), phy = tree))

# when we fit PIC scores to lm, we need to make sure that our regression line does not have an intercept (Revell and Harmon, 2022)
# this is because the position of right and left nodes is arbitrary for all nodes in our phylogeny, so is the direction of the subtraction of the PICs
# so the model should go through the origin (0, 0)
png(filename = "../plots/linearreg_pic_ortho_log_traitvals.png", width = 10, height = 10, units = "in", res = 1000)
par(mar = c(5, 5, 1, 1), mfrow=c(2, 2))
for (collab in c("RD", "SRL")) {
    for (conserv in c("RN", "RTD")) {
        picorthomod <- lm(unlist(pic_ortho_data[collab])~unlist(pic_ortho_data[conserv])+0)
        plot(x = unlist(pic_ortho_data[conserv]), y = unlist(pic_ortho_data[collab]), xlab = paste0("PIC(log(", conserv, "))"), ylab = paste0("PIC(log(", collab, "))"))
        abline(h = 0, lty = "dotted")
        abline(v = 0, lty = "dotted")
        abline(picorthomod, lwd = 1, col = "red")
        mtext(paste0("y = ", round(coef(picorthomod), 2), " x"), side = 3, col = "red", line = -2)
    }
}
dev.off()

#------------------------
# Phylomorphospace
#------------------------

# TODO - redo this!!!!

par(mar = c(5, 5, 1, 1))
png("../plots/phylomorphospace_rtd_srl.png", width = 10000, height = 10000, units = "px", res = 800)
phytools::phylomorphospace(tree = dichot, X = cbind(log(named_rtd_vec), log(named_srl_vec_hypo_1)), xlab = "log(RTD)", ylab = "log(SRL)", label = "off", node.size = c(0, 0), log = "xy", xlim = c(0, max(log(named_rtd_vec))))
points(x = named_rtd_vec, y = named_srl_vec_hypo_1, pch = 21)
grid()
abline(lm(log(named_srl_vec_hypo_1)~log(named_rtd_vec)), lwd = 2, col = "red")
dev.off()


#-----------------------------------------------------------
# METHOD 2 - Phylogenetic Generalized Least Squares (PGLS)
#-----------------------------------------------------------


correlation_matrix <- ape::corBrownian(phy = tree) # the form argument is used to specify the order of our data
# since we have vectors with data in the same order as the tree here, it's not necessary

png(filename = "../plots/pgls_species_avg_log_traitvals.png", width = 10, height = 10, units = "in", res = 1000)
par(mar = c(5, 5, 1, 1), mfrow=c(2, 2))
for (collab in c("RD", "SRL")) {
    for (conserv in c("RN", "RTD")) {
        collab_trait <- log(res_avgd_traits[, collab])
        conserv_trait <- log(res_avgd_traits[, conserv])
        pgls <- nlme::gls(collab_trait~conserv_trait, correlation = correlation_matrix) # doing the above inline in the formula arg of nlme::gls raises an exception saying invalid type list for variable ......????
        plot(x = conserv_trait, y = collab_trait, xlab = paste0("log(", conserv, ")"), ylab = paste0("log(", collab, ")"))
        abline(pgls, lwd = 1, col = "red")
        mtext(paste0("y = ", round(coef(pgls)[2], 2), " x + ", round(coef(pgls)[1], 2)), side = 3, col = "red", line = -2)
    }
}
dev.off()

# dataframe[colname] returns a named list (with rownames)!!!

