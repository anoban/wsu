# map the extant and reconstructed trait values to the phylogeny and plot them

library("ape")
library("phytools")

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
stopifnot(length(FINALIZED_MYCORRHIZAL_STATES$binominal) == 395)


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
png("../plots/FRED_collab_395sp_RD_mapped_phylogeny.png", width = 8000, height = 8000, units = "px", res = 200)
plot(mappedRD, ftype = "i", fsize = 1.4, type = "fan", lwd = 3, part = 0.99, leg.txt = "RD in cm")
tscale_axis <- axis(1, pos = -2, at = TSCALE_LENGTH - seq(0, TSCALE_LENGTH, length.out = 10), cex.axis = 1.75, labels = FALSE, col = "black", lwd = 2)
text(x = tscale_axis, y = rep(-16, 10), labels = lapply(rev(seq(0, TSCALE_LENGTH, length.out = 10)), sprintf, fmt = "%.2f"), cex = 2, col = "black")
text(x = 250, y = -35, labels = "Time (Million years)", cex = 2, col = "black")
dev.off()

# for specific root length (first order roots)
mappedSRL <- phytools::contMap(tree = PHYLOGENY, x = SRL, plot = FALSE)
png("../plots/FRED_collab_395sp_SRL_mapped_phylogeny.png", width = 8000, height = 8000, units = "px", res = 200)
plot(mappedSRL, ftype = "i", fsize = 1.4, type = "fan", lwd = 3, part = 0.99, leg.txt = "SRL in m/g")
tscale_axis <- axis(1, pos = -2, at = TSCALE_LENGTH - seq(0, TSCALE_LENGTH, length.out = 10), cex.axis = 1.75, labels = FALSE, col = "black", lwd = 2)
text(x = tscale_axis, y = rep(-16, 10), labels = lapply(rev(seq(0, TSCALE_LENGTH, length.out = 10)), sprintf, fmt = "%.2f"), cex = 2, col = "black")
text(x = 250, y = -35, labels = "Time (Million years)", cex = 2, col = "black")
dev.off()

# RD and mycorrhizal states
png("../plots/FRED_collab_395sp_RD_n_states_mapped_phylogeny.png", width = 12000, height = 12000, units = "px", res = 400)
mappedRDSTATES <- phytools::contMap(tree = PHYLOGENY, x = RD, res = 400, ftype = "i", fsize = 1.4, type = "fan", lwd = 0.8, part = 0.99)
plot(map, type = "fan")
ape::nodelabels(node = er_mystates$marginal.anc |> row.names() |> as.numeric(), pie = er_mystates$marginal.anc, piecol = myco_state_colours, cex = 0.1)
ape::tiplabels(pie = to.matrix(named_mycorrhizal_state_vec, sort(unique(named_mycorrhizal_state_vec))), piecol = myco_state_colours, cex = 0.1)
legend("topright", legend = sort(unique(named_mycorrhizal_state_vec)), pt.bg = myco_state_colours, cex = 3, pt.cex = 5, pch = 21)
tscale_axis <- axis(1, pos = -2, at = htree - seq(0, htree, length.out = 10), cex.axis = 1., labels = FALSE, col = "black")
text(x = tscale_axis, y = rep(-10, 10), labels = lapply(rev(seq(0, htree, length.out = 10)), sprintf, fmt = "%.2f"), cex = 1, col = "black")
text(x = 250, y = -30, labels = "Time (Million years)", cex = 1.5, col = "black")
dev.off()
