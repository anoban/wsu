# map the extant and reconstructed trait values to the phylogeny and plot them

library("ape")
library("phytools")
library("corHMM")
library("RColorBrewer")

PHYLOGENY <- ape::read.tree("../data/chapter2/uphylomaker/FRED_subset_collab_395sp.tre") # 395 species
stopifnot(length(PHYLOGENY$tip.label) == 395)

# read in the continuous traits and categorical trait
FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD <- read.csv("../data/chapter2/FREDv3subset/collab_ord1_species_avgs_SRL_RD.csv")
names(FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD) <- c("binominal", "RD", "SRL")
stopifnot(length(FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD$binominal) == 395)
# make the names match with the phylogeny
FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD$binominal <- gsub(pattern = ' ', replacement = '_', x = FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD$binominal)
# reorder the rows to match the phylogeny
FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD <- FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD[match(x = PHYLOGENY$tip.label, FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD$binominal), ]
stopifnot(all(FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD$binominal == PHYLOGENY$tip.label))



FINALIZED_MYCORRHIZAL_STATES <- read.csv("../data/chapter2/FREDv3subset/finalized_states_395_species.csv")
FINALIZED_MYCORRHIZAL_STATES$state <- gsub(pattern = '/', replacement = '', x = FINALIZED_MYCORRHIZAL_STATES$state) # let's remove the forward slashes because it caused errors with OUwie
FINALIZED_MYCORRHIZAL_STATES$binominal <- gsub(pattern = ' ', replacement = '_', FINALIZED_MYCORRHIZAL_STATES$binominal) # replace the spaces with underscores to match the tip labels in the phylogeny
FINALIZED_MYCORRHIZAL_STATES <- FINALIZED_MYCORRHIZAL_STATES[match(x = PHYLOGENY$tip.label, FINALIZED_MYCORRHIZAL_STATES$binominal), ]
stopifnot(length(FINALIZED_MYCORRHIZAL_STATES$binominal) == 395)
stopifnot(all(FINALIZED_MYCORRHIZAL_STATES$binominal == PHYLOGENY$tip.label))

#---------------------------------------------------------
# Ancestral State Estimation (ACE) for continuous traits
#---------------------------------------------------------

# create named vectors trait values
RD <- setNames(FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD[, "RD"], nm = FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD$binominal)
SRL <- setNames(FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD[, "SRL"], nm = FIRST_ORDER_COLLAB_SP_AVERAGED_SRL_N_RD$binominal)

# aceRD <- phytools::fastAnc(tree = PHYLOGENY, x = RD, vars = TRUE, CI = TRUE)
# aceSRL <- phytools::fastAnc(tree = PHYLOGENY, x = SRL, vars = TRUE, CI = TRUE)

TSCALE_LENGTH <- max(phytools::nodeHeights(PHYLOGENY))

# for root diameter (first order roots)
mappedRD <- phytools::contMap(tree = PHYLOGENY, x = RD, plot = FALSE)
png("../plots/FRED_collab_395sp_RD_mapped_phylogeny.png", width = 12000, height = 12000, units = "px", res = 400)
plot(mappedRD, ftype = "i", fsize = 1.2, type = "fan", lwd = 3, part = 0.99, leg.txt = "RD in cm")
tscale_axis <- axis(1, pos = -2, at = TSCALE_LENGTH - seq(0, TSCALE_LENGTH, length.out = 10), cex.axis = 1.75, labels = FALSE, col = "black", lwd = 2)
text(x = tscale_axis, y = rep(-16, 10), labels = lapply(rev(seq(0, TSCALE_LENGTH, length.out = 10)), sprintf, fmt = "%.2f"), cex = 2, col = "black")
text(x = 250, y = -28, labels = "Time (Million years)", cex = 2, col = "black")
dev.off()

# for specific root length (first order roots)
mappedSRL <- phytools::contMap(tree = PHYLOGENY, x = SRL, plot = FALSE)
png("../plots/FRED_collab_395sp_SRL_mapped_phylogeny.png", width = 12000, height = 12000, units = "px", res = 400)
plot(mappedSRL, ftype = "i", fsize = 1.2, type = "fan", lwd = 3, part = 0.99, leg.txt = "SRL in m/g")
tscale_axis <- axis(1, pos = -2, at = TSCALE_LENGTH - seq(0, TSCALE_LENGTH, length.out = 10), cex.axis = 1.75, labels = FALSE, col = "black", lwd = 2)
text(x = tscale_axis, y = rep(-16, 10), labels = lapply(rev(seq(0, TSCALE_LENGTH, length.out = 10)), sprintf, fmt = "%.2f"), cex = 2, col = "black")
text(x = 250, y = -28, labels = "Time (Million years)", cex = 2, col = "black")
dev.off()


# for the ACE of discrete traits we first need to choose a evolutionary model, similar to what we did with OUwie
unique(FINALIZED_MYCORRHIZAL_STATES$state) # got 6 unique states - "AMNM"  "AM"    "ErM"   "NM"    "AMEcM" "EcM"
states <- setNames(FINALIZED_MYCORRHIZAL_STATES$state, nm = FINALIZED_MYCORRHIZAL_STATES$binominal)

# for a detailed walkthrough about the model arg, browse the documentation of ape::ace which is very similar (nearly identical) to the way OUwie handles regime rate matrices
# also check https://blog.phytools.org/2015/05/about-how-acemarginaltrue-does-not.html out

if (file.exists("./rdata/fitMk.RData")) load("./rdata/fitMk.RData")

discER <- phytools::fitMk(tree = PHYLOGENY, x = states, model = "ER")
discARD <- phytools::fitMk(tree = PHYLOGENY, x = states, model = "ARD")
discSYM <- phytools::fitMk(tree = PHYLOGENY, x = states, model = "SYM")

# we can also pass tailored regime rate matrices like OUwie::hOUwie

data.frame(model = c("ER", "ARD", "SYM"),
           AIC = c(stats::AIC(discER), stats::AIC(discARD), stats::AIC(discSYM)),
           lnLik = c(stats::logLik(discER), stats::logLik(discARD), stats::logLik(discSYM))
           )
#   model      AIC     lnLik
# 1    ER 286.6082 -142.3041
# 2   ARD 320.8487 -130.4244
# 3   SYM 300.4172 -135.2086

# save(discER, discSYM, discARD, file = "./rdata/fitMk.RData")


# https://michael-franke.github.io/intro-data-analysis/Chap-03-06-model-comparison-AIC.html
# our model evaluation suggests ER to be the best fit here???? (lowest AIC)

# for the ACE of discrete traits we have 2 options:
# 1) rerooting method
# 2) corHMM

marginalACE_corHMM <- corHMM::corHMM(phy = PHYLOGENY, data = FINALIZED_MYCORRHIZAL_STATES[, c("binominal", "state")], model = "ER", node.states = "marginal", rate.cat = 1)
rrStates <- phytools::rerootingMethod(tree = PHYLOGENY, x = states, model = "ER")
# A WARNING ABOUT THE phytools::rerootingMethod FUNCTION =>
# This function is redundant with 'phytools::ancr' in situations in which it should be used (symmetric Q matrices) & invalid for non-symmetric Q matrices (e.g., model='ARD').

#-----------------------------------------------------------
# PLOT THE ACE OF MYCORRHIZAL STATES ON THE PHYLOGENY
#-----------------------------------------------------------

STATE_COLOURS <- c("blue", "red", "green", "orange", "yellow", "magenta")

png("../plots/FRED_collab_395sp_corHMM_marginal_states_mapped_phylogeny.png", width = 12000, height = 12000, units = "px", res = 400)
plot <- phytools::plotTree(tree = PHYLOGENY, ftype = "i", fsize = 1.2, type = "fan", lwd = 1, part = 0.99, offset = 2) # space the species names a bit far from the tips with `offset`
tscale_axis <- axis(1, pos = -2, at = TSCALE_LENGTH - seq(0, TSCALE_LENGTH, length.out = 10), cex.axis = 1., labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-10, 10), labels = lapply(rev(seq(0, TSCALE_LENGTH, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1, col = "red")
text(x = 250, y = -20, labels = "Time (Million years)", cex = 1.5, col = "red")
ape::nodelabels(pie = marginalACE_corHMM$states, piecol = STATE_COLOURS, cex = 0.1) # internal nodes
ape::tiplabels(pie = to.matrix(FINALIZED_MYCORRHIZAL_STATES$state, sort(unique(FINALIZED_MYCORRHIZAL_STATES$state))), piecol = STATE_COLOURS, cex = 0.1) # nodes at the tips
legend("topright", legend = c("AM", "AM/EcM", "AM/NM", "EcM", "ErM", "NM"), # same order as => sort(unique(FINALIZED_MYCORRHIZAL_STATES$state)),
       pt.bg = STATE_COLOURS, cex = 3, pt.cex = 5, pch = 21, ncol = 2)
dev.off()


png("../plots/FRED_collab_395sp_rerooting_marginal_states_mapped_phylogeny.png", width = 12000, height = 12000, units = "px", res = 400)
plot <- phytools::plotTree(tree = PHYLOGENY, ftype = "i", fsize = 1.2, type = "fan", lwd = 1, part = 0.99, offset = 2) # space the species names a bit far from the tips with `offset`
tscale_axis <- axis(1, pos = -2, at = TSCALE_LENGTH - seq(0, TSCALE_LENGTH, length.out = 10), cex.axis = 1., labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-10, 10), labels = lapply(rev(seq(0, TSCALE_LENGTH, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1, col = "red")
text(x = 250, y = -20, labels = "Time (Million years)", cex = 1.5, col = "red")
ape::nodelabels(pie = rrStates$marginal.anc, piecol = STATE_COLOURS, cex = 0.1)
ape::tiplabels(pie = to.matrix(FINALIZED_MYCORRHIZAL_STATES$state, sort(unique(FINALIZED_MYCORRHIZAL_STATES$state))), piecol = STATE_COLOURS, cex = 0.1) # nodes at the tips
legend("topright", legend = c("AM", "AM/EcM", "AM/NM", "EcM", "ErM", "NM"), # same order as => sort(unique(FINALIZED_MYCORRHIZAL_STATES$state)),
       pt.bg = STATE_COLOURS, cex = 3, pt.cex = 5, pch = 21, ncol = 2)
dev.off()



#---------------------------------------------------------------------------------------------------
# PHYLOGENY OF THE 1005 SPECIES FRED V3 SUBSET
#---------------------------------------------------------------------------------------------------

state_colors <- c("red", "blue", "yellow", "orange", "green", "purple", "white")
phylogeny <- ape::read.tree("../data/chapter2/uphylomaker/FRED_subset_collab_1005sp.tre")
states <- read.csv("./parhOUwie/genus_state_rec_logged_species_avgd_RD_1005sp.csv")[, c("binominal", "state")]
png("../plots/1005_species_states_mapped_phylogeny.png", width = 22000, height = 22000, units = "px", res = 400)
plot <- phytools::plotTree(tree = phylogeny, ftype = "i", fsize = 1.2, type = "fan", lwd = 1, offset = 2)
ape::tiplabels(pie = to.matrix(states$state, sort(unique(states$state))), piecol = state_colors, cex = 0.1) # nodes at the tips
legend("topright", legend = sort(unique(states$state)), pt.bg = state_colors, cex = 3, pt.cex = 5, pch = 21, ncol = 2)
dev.off()

#------------------------------------------------------------------------
# PHYLOGENY OF THE 995 SPECIES 5 MYCORRHIZAL STATE FRED V3 SUBSET
# TO USE IN THE DRAFT FOR STEVEN SMITH & LUKE MCCORMACK
#------------------------------------------------------------------------

state_colors <- c("red", "blue", "yellow", "orange", "green")
phylogeny <- ape::read.tree("../data/chapter2/uphylomaker/collab_fineroots_log_995_species_means_5states.tre")
tscale_max <- max(phytools::nodeHeights(phylogeny))
states <- read.csv("../data/chapter2/FREDv3subset/collab_fineroots_log_995_species_means_5states.csv")[, c("binominal", "state")]

# par(bg = NA)
png("../plots/995_species_5states_mapped_phylogeny.png", width = 22000, height = 22000, units = "px", res = 400, bg = "transparent")
plot <- phytools::plotTree(tree = phylogeny, ftype = "i", fsize = 1.0, lwd = 1, offset = 2, part = 0.998, type = "fan")
tscale_ticks <- seq(0, tscale_max, length.out = 20)
tscale_axis <- axis(1, pos = -1, at = tscale_max - tscale_ticks, cex.axis = 1., labels = FALSE, col = "red")
text(x = tscale_axis, y = rep(-3, length(tscale_ticks)), labels = lapply(rev(tscale_ticks), sprintf, fmt = "%.2f"), cex = 1, col = "red")
text(x = tscale_max + 20, y = -2.5, labels = "Time (Million years)", cex = 1.00, col = "red")
ape::tiplabels(pie = to.matrix(states$state, sort(unique(states$state))), piecol = state_colors, cex = 0.05) # nodes at the tips
legend("topright", legend = sort(unique(states$state)), pt.bg = state_colors, cex = 3, pt.cex = 5, pch = 21, ncol = 1)
dev.off()

# the above tree with an outer circle added to show order delineation amongst tips
data <- read.csv("../data/chapter2/FREDv3subset/collab_fineroots_log_995_species_means_5states_name_matched_with_phylogeny.csv")
data
