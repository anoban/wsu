library("ape")
library("phytools")
library("corHMM")

#-------------------------------------------------------------------------------------------------------------------------------------------------
# STUDY THE RELATIONSHIPS BETWEEN THE DISCRETE STATE TRANSITIONS AND THE CONTINUOUS TRAIT CHANGES BASED ON INDEPENDENTLY ESTIMATED ACE
#-------------------------------------------------------------------------------------------------------------------------------------------------

STATES <- read.csv("../data/chapter2/FREDv3subset/finalized_states_395_species.csv", stringsAsFactors = TRUE)[, c("binominal", "state")] # finalized mycorrhizal states
COLLAB_AXIS <- read.csv("../data/chapter2/FREDv3subset/collab_ord1_species_avgs_SRL_RD.csv", stringsAsFactors = TRUE) # first order species averaged RD and SRL values
MERGED <- merge(x = STATES, y = COLLAB_AXIS, by = "binominal")
stopifnot(nrow(MERGED)==395)

COLLAB_395SP_TREE <- ape::multi2di(ape::read.tree("../data/chapter2/uphylomaker/FRED_subset_collab_395sp.tre")) # phylogenetic tree created for the 395 species using U.PhyloMaker
stopifnot(length(COLLAB_395SP_TREE$tip.label)==395)

# MERGED contains '/' in the binominal names, replace that with underscores
data <- data.frame(binominal = gsub(MERGED$binominal, pattern = ' ', replacement = '_'), RD = MERGED$F00679, SRL = MERGED$F00727, myco = gsub(x = MERGED$state, pattern = '/', replacement = ''))
matched_row_indices <- match(COLLAB_395SP_TREE$tip.label, data$binominal)
stopifnot(all(data$binominal[matched_row_indices] == COLLAB_395SP_TREE$tip.label))
data <- data[matched_row_indices, ] # reorder the dataset to match the species order in the phylogeny
stopifnot(all(data$binominal == COLLAB_395SP_TREE$tip.label))
stopifnot(length(unique(data$binominal)) == length(data$binominal))

# FOR CONVENIRNCE
RD <- setNames(object = data$RD, nm = data$binominal)
SRL <- setNames(object = data$SRL, nm = data$binominal)
STATES <- setNames(object = data$myco, nm = data$binominal)


# PLOT THE PHYLOGENY AND NODE & TIP NUMBERS, IT'LL HELP IN TRACING DOWN CHANGES AND SHIFTS IN TRAITS
par(mar=c(0, 0, 0, 0))
png(filename = "../plots/FRED_subset_collab_395sp_nodes_n_tips.png", width = 18000, height = 18000, units = "px", res = 300)
phytools::plotTree(tree = COLLAB_395SP_TREE, ftype = "i", fsize = 1.2, type = "fan", lwd = 1, offset = 4)
phytools::labelnodes(text = 1:(COLLAB_395SP_TREE$Nnode + length(COLLAB_395SP_TREE$tip.label)), node = 1:(COLLAB_395SP_TREE$Nnode + length(COLLAB_395SP_TREE$tip.label)), cex = 1, interactive = FALSE)
dev.off()


#---------------------------------------------------------------
# ACE OF DISCRETE CATEGORICAL TRAITS AND CONTINUOUS TRAITS
#---------------------------------------------------------------

# CATEGORICAL TRAITS
phytools::rerootingMethod(tree = COLLAB_395SP_TREE, x = STATES, model = "ER")
phytools::ancr()


# ACE of mycorrhizal states using corHMM (MARGINAL ACE)
corHMM::corHMM(phy = COLLAB_395SP_TREE, data = data[, c("binominal", "myco")], model = "ER", node.states = "marginal", rate.cat = 1)
