# ---
# Publication title: An integrated framework of plant form and function: The belowground perspective
# Authors: Alexandra Weigelt, Liesje Mommer, Karl Andraczek, Colleen M. Iversen, Joana Bergmann, Helge Bruelheide, Ying Fan, GrC)goire T. Freschet, Nathaly R. Guerrero-RamC-rez, Jens Kattge, Thom W. Kuyper, Daniel C. Laughlin, Ina C. Meier, Fons van der Plas, Hendrik Poorter, Catherine Roumet, Jasper van Ruijven, Francesco Maria Sabatini, Marina Semchenko, Christopher J. Sweeney, Oscar J. Valverde-Barrantes, Larry M. York, M. Luke McCormack
# Acceptance date: 13 June 2021
#
#
# R code title: "Weigelt et al. 2021 RCode.Analysis"
# R code author: "Karl Andraczek"
# co-authors: "Nathaly Guerrero Ramirez, Joana Bergmann, Alfons van der Plas, Larry York, Jens Kattge, Helge Bruelheide, Oscar Valverde-Barrantes, Daniel Laughlin"
# date: "23.03.2021"
#
# ---
#
# ###############################################################################################################################
#
# Last update 23.03.2021
#
# ###############################################################################################################################
#
# Datasets needed to run code:
#
# I)    Main PCA matrix: Weigelt_et_al_2021_Main.PCA.Matrix.csv
#
# II)   Individual PCA matrix:
#
# III)  nodDB Database on N-Fixation data from Tedersoo et al. 2018 (https://onlinelibrary.wiley.com/doi/abs/10.1111/jvs.12627)
#
# IV)   FungalRoot Database on Mycorrhizal association from Soudzilovskaia et al. 2020 (https://nph.onlinelibrary.wiley.com/doi/abs/10.1111/nph.16569)


### I) Set working directory ####################################################################################

rm(list = ls())

# set working directory to source file location if using RStudio, otherwise do manually

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# install packages

install.packages("brranching")
install.packages("phytools")
install.packages("shape")

install.packages("devtools") #
library(devtools) # required path for loading the pairwiseAdonis package
install_github("pmartinezarbizu/pairwiseAdonis/pairwiseAdonis") #
install.packages("pairwiseAdonis") #

install.packages("caper")
install.packages("Hmisc")

library(devtools)
install_version("brranching", version = "0.4.0", repos = "http://cran.us.r-project.org")

# load packages

require(plyr)
require(data.table)
library(tidyverse)
library(stringr)
require(reshape2)
library(lme4)
library(brranching)
library(shape)
library(pairwiseAdonis)
library(caper)
library(Hmisc)
library(ape) # contains package phangorn
library(taxize)
library(phytools)
library(clhs)
library(rentrez)

###############################################################################################################################
# PCA with 6 core traits
###############################################################################################################################

# load trait data

FungalRoot_db <- read.csv("FungalRoot_database_17_06_2020.csv", sep = ";", header = T, na.strings = c("", "NA")) # Fungal root database
nodDB_20_10_2020 <- read.csv("nodDB_20_10_2020.csv", header = T, sep = ";", na.strings = c("", "NA")) # N-fixation database

Core_combined_meta <- read.csv("Weigelt_et_al_2021_Main.PCA.Matrix.csv", header = T, sep = ";", na.strings = c("", "NA"))

# extract only the 6 core traits (LMA,LN,RN,RTD,SRL,MRD)

Core_combined_PCA <- Core_combined_meta[, c(1:3, 10:11, 13:14, 19:21)]
Core_combined_PCA <- Core_combined_PCA[complete.cases(Core_combined_PCA[, c(1:7)]), ]

## Fill missing information on Woodiness, Mycorrhizal associations and N-Fixation ability

# Add Woodiness information

Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Acer_coriaceifolium"] <- "woody"
Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Tanaecium_pyramidatum"] <- "woody"
Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Caldcluvia_rosifolia"] <- "woody"
Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Leiospermum_racemosum"] <- "woody"
Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Trachelospermum_axillare"] <- "woody"
Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Gutierrezia_sarothrae"] <- "woody"

Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Artemisia_mongolica"] <- "non-woody"
Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Bituminaria_bituminosa"] <- "non-woody"
Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Dalea_candida"] <- "non-woody"
Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Dalea_purpurea"] <- "non-woody"
Core_combined_PCA$woodiness[Core_combined_PCA$full_species == "Symphyotrichum_oolentangiense"] <- "non-woody"

# Add mycorrhizal Information and collaps subgroups to either unknown,EM,AM,EM+AM,ErM,NM

Core_combined_PCA$full_species <- gsub("_", " ", Core_combined_PCA$full_species)
Core_combined_PCA$Genus <- word(Core_combined_PCA$full_species, start = 1, end = 1)
Core_combined_PCA <- merge(Core_combined_PCA, FungalRoot_db, by = "Genus", all.x = TRUE)
Core_combined_PCA$full_species <- gsub(" ", "_", Core_combined_PCA$full_species)
Core_combined_PCA$Genus <- NULL

Core_combined_PCA$Mycorrhizal.type[which(Core_combined_PCA$Mycorrhizal.type == "uncertain")] <- "unknown"
Core_combined_PCA$Mycorrhizal.type[Core_combined_PCA$Mycorrhizal.type == "EcM"] <- "EM"
Core_combined_PCA$Mycorrhizal.type[Core_combined_PCA$Mycorrhizal.type == "EcM-AM"] <- "EM+AM"
Core_combined_PCA$Mycorrhizal.type[Core_combined_PCA$Mycorrhizal.type == "NM-AM"] <- "NM"
Core_combined_PCA$Mycorrhizal.type[is.na(Core_combined_PCA$Mycorrhizal.type)] <- "unknown"
Core_combined_PCA$Mycorrhizal.type[Core_combined_PCA$Mycorrhizal.type == "species-specific: AM or rarely EcM-AM or AM NM-AM, rarely EcM"] <- "AM"
Core_combined_PCA$Mycorrhizal.type[Core_combined_PCA$Mycorrhizal.type == "NM-AM, rarely EcM"] <- "NM"
Core_combined_PCA$Mycorrhizal.type[Core_combined_PCA$Mycorrhizal.type == "species-specific: AM or rarely EcM-AM or AM"] <- "AM"
Core_combined_PCA$Mycorrhizal.type[Core_combined_PCA$full_species == "Abelia_biflora"] <- "AM"
Core_combined_PCA$Mycorrhizal.type[Core_combined_PCA$full_species == "Embelia_vestita"] <- "AM"

Core_combined_PCA$Mycorrhizal_type <- NULL
colnames(Core_combined_PCA)[10] <- "Mycorrhizal_type"

# Add N-Fixation ability

Core_combined_PCA$full_species <- gsub("_", " ", Core_combined_PCA$full_species)
Core_combined_PCA$genus <- word(Core_combined_PCA$full_species, 1)
Core_combined_PCA <- merge(Core_combined_PCA, nodDB_20_10_2020[, c(4, 5)], by = "genus", all.x = T)
Core_combined_PCA$N_fixation[Core_combined_PCA$Consensus.estimate == "Rhizobia"] <- "N-fixing"
Core_combined_PCA$N_fixation[Core_combined_PCA$N_fixation == "unknown"] <- "Non-N-fixing"
Core_combined_PCA$genus <- NULL
Core_combined_PCA$Consensus.estimate <- NULL
Core_combined_PCA$full_species <- gsub(" ", "_", Core_combined_PCA$full_species)

Core_combined_PCA$N_fixation[is.na(Core_combined_PCA$N_fixation)] <- "None"
Core_combined_PCA$N_fixation[Core_combined_PCA$N_fixation == "likely_present"] <- "N-fixing"
Core_combined_PCA$N_fixation[Core_combined_PCA$N_fixation == "Frankia"] <- "N-fixing"
Core_combined_PCA$N_fixation[Core_combined_PCA$N_fixation == "unlikely_Rhizobia"] <- "None"

## Create final dataset for PCA analysis

PCA_NEW <- Core_combined_PCA

# clear workspace except for PCA_NEW dataset

rm(list = setdiff(ls(), "PCA_NEW"))

# Change species names (typo, synonym)

PCA_NEW$full_species <- gsub("Caryota_maxima", "Caryota_mitis", PCA_NEW$full_species) # will be assigned to mites only for the purpose of phylognetic position in the tree
PCA_NEW$full_species <- gsub("Quintinia_serrata", "Quintinia_verdonii", PCA_NEW$full_species) # will be assigned to verdonii only for the purpose of phylognetic position in the tree

## extract phylogenetic information on family names for all species in the PCA data set from ncbi (see https://www.ncbi.nlm.nih.gov/)
## this approach was recommended from the author of the package "brranching" after reporting errors related to the function "phylomatic" that is usally used to
## extract phylogeny (see GitHub https://github.com/ropensci/brranching/issues/42)

set_entrez_key("YOUR_KEY") # enter your API key
Sys.getenv("ENTREZ_KEY")
# API key from ncbi (see https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)

# extract family names from ncbi

my_names_phyl <- phylomatic_names(PCA_NEW$full_species, db = "ncbi")

# Merge extracted family names with PCA dataset

PCA_Species_PhylomaticNames <- data.frame(my_names_phyl)
colnames(PCA_Species_PhylomaticNames)[1] <- "full_species"
my_names_half <- PCA_Species_PhylomaticNames
my_names_half <- data.frame(my_names_half)
my_names_half$ID <- seq.int(nrow(my_names_half))
PCA_NEW$ID <- seq.int(nrow(PCA_NEW))
colnames(my_names_half)[1] <- "full_species_ncbi"
PCA_NEW_2 <- merge(PCA_NEW, my_names_half, by = "ID")
PCA_NEW_2$ID <- NULL
PCA_NEW <- PCA_NEW_2

# clear workspace except for PCA_NEW dataset

rm(list = setdiff(ls(), "PCA_NEW"))

# Short cut to load data from previous sections

# write.table(PCA_NEW,file="PCA_NEW_FullPhylo.csv",sep=";")
# PCA_NEW <- read.csv("PCA_NEW_FullPhylo.csv",header=T,sep=';',na.strings=c("","NA"))

# create subsets for woody, non-woody

PCA_NEW_woody <- PCA_NEW %>%
    dplyr::filter(woodiness %in% "woody")

PCA_NEW_Nonwoody <- PCA_NEW %>%
    dplyr::filter(woodiness %in% "non-woody")

# Extract phylogeny for main dataset, woody and non-woody subset

tree.species.set <- phylomatic(PCA_NEW$full_species_ncbi, taxnames = FALSE, get = "POST", storedtree = "zanne2014")

tree.species.set_woody <- phylomatic(PCA_NEW_woody$full_species_ncbi, taxnames = FALSE, get = "POST", storedtree = "zanne2014")

tree.species.set_nonwoody <- phylomatic(PCA_NEW_Nonwoody$full_species_ncbi, taxnames = FALSE, get = "POST", storedtree = "zanne2014")

# attache missing species to related sister species for main data set

tree.species.set <- phangorn::add.tips(tree.species.set,
    tips = c("Oxybasis_glauca"), edge.length = 1,
    where = which(tree.species.set$tip.label == "Atriplex_prostrata")
) # attach Oxybasis to Atriplex
tree.species.set <- phangorn::add.tips(tree.species.set,
    tips = c("Chondrosum_gracile"), edge.length = 1,
    where = which(tree.species.set$tip.label == "Bouteloua_curtipendula")
) # attach Chondrosum to Bouteloua
tree.species.set <- phangorn::add.tips(tree.species.set,
    tips = c("Leiospermum_racemosum"), edge.length = 1,
    where = which(tree.species.set$tip.label == "Caldcluvia_rosifolia")
) # attach Leiospermum to Caldcluvia
tree.species.set <- phangorn::add.tips(tree.species.set,
    tips = c("Manglietia_paruicula"), edge.length = 1,
    where = which(tree.species.set$tip.label == "Magnolia_grandiflora")
) # attach Manglietia to Magnolia

# attache missing species to related sister species for woody subset

tree.species.set_woody <- phangorn::add.tips(tree.species.set_woody,
    tips = c("Leiospermum_racemosum"),
    edge.length = 1, where = which(tree.species.set_woody$tip.label == "Caldcluvia_rosifolia")
) # attach Leiospermum to Caldcluvia
tree.species.set_woody <- phangorn::add.tips(tree.species.set_woody,
    tips = c("Manglietia_paruicula"),
    edge.length = 1, where = which(tree.species.set_woody$tip.label == "Magnolia_grandiflora")
) # attach Manglietia to Magnolia

# attache missing species to related sister species for non-woody subset

tree.species.set_nonwoody <- phangorn::add.tips(tree.species.set_nonwoody,
    tips = c("Oxybasis_glauca"), edge.length = 1,
    where = which(tree.species.set_nonwoody$tip.label == "Atriplex_prostrata")
) # attach Oxybasis to Atriplex
tree.species.set_nonwoody <- phangorn::add.tips(tree.species.set_nonwoody,
    tips = c("Chondrosum_gracile"), edge.length = 1,
    where = which(tree.species.set_nonwoody$tip.label == "Bouteloua_curtipendula")
) # attach Chondrosum to Bouteloua

# capitalize and remove underscore on all tip.labels for main dataset, woody and non-woody subset

tree.species.set$tip.label <- Hmisc::capitalize(gsub("_na", "_NA", tree.species.set$tip.label))
tree.species.set$tip.label <- Hmisc::capitalize(gsub("_NAnmu", "_nanmu", tree.species.set$tip.label))

tree.species.set_woody$tip.label <- Hmisc::capitalize(gsub("_na", "_NA", tree.species.set_woody$tip.label))
tree.species.set_woody$tip.label <- Hmisc::capitalize(gsub("_NAnmu", "_nanmu", tree.species.set_woody$tip.label))

tree.species.set_nonwoody$tip.label <- Hmisc::capitalize(gsub("_na", "_NA", tree.species.set_nonwoody$tip.label))
tree.species.set_nonwoody$tip.label <- Hmisc::capitalize(gsub("_NAnmu", "_nanmu", tree.species.set_nonwoody$tip.label))

# clean and sort dataframes and tree

p.dist.mat <- cophenetic(tree.species.set)
row.names(PCA_NEW) <- PCA_NEW$full_species
PCA_NEW <- PCA_NEW[row.names(p.dist.mat), ]

p.dist.mat2 <- cophenetic(tree.species.set_woody)
row.names(PCA_NEW_woody) <- PCA_NEW_woody$full_species
PCA_NEW_woody <- PCA_NEW_woody[row.names(p.dist.mat2), ]

p.dist.mat3 <- cophenetic(tree.species.set_nonwoody)
row.names(PCA_NEW_Nonwoody) <- PCA_NEW_Nonwoody$full_species
PCA_NEW_Nonwoody <- PCA_NEW_Nonwoody[row.names(p.dist.mat3), ]


# Rename columns

colnames(PCA_NEW) <- c("Species", "LMA", "LeafN", "RootN", "D", "RTD", "SRL", "Woodiness", "N_Fix", "Myco", "Species_ncbi")

colnames(PCA_NEW_woody) <- c("Species", "LMA", "LeafN", "RootN", "D", "RTD", "SRL", "Woodiness", "N_Fix", "Myco", "Species_ncbi")

colnames(PCA_NEW_Nonwoody) <- c("Species", "LMA", "LeafN", "RootN", "D", "RTD", "SRL", "Woodiness", "N_Fix", "Myco", "Species_ncbi")


## Non-phylogenetic informed PCA

RootLeafPCA <- prcomp(PCA_NEW[, c(2:7)], center = TRUE, scale. = TRUE)
summary(RootLeafPCA)
axes <- predict(RootLeafPCA)

eigenvalues <- RootLeafPCA$sdev^2

### Supporting information Figure S3 ########

# Non- phylogenetically informed principal component analysis of species mean traits (corresponding to phylogenetically informed Figure 3B).

postscript("Extended Figure 4.eps", width = 9.0, height = 9.0)
plot(RootLeafPCA$x[-which(PCA_NEW$Myco == "unknown"), 1],
    -RootLeafPCA$x[-which(PCA_NEW$Myco == "unknown"), 2],
    pch = 19,
    xlab = "PC1", ylab = "PC2", main = "Myco", xlim = c(-6, 6),
    ylim = c(-6, 6), cex = 0.5
)
points(RootLeafPCA$x[which(PCA_NEW$Myco == "AM"), 1],
    -RootLeafPCA$x[which(PCA_NEW$Myco == "AM"), 2],
    pch = 20, col = "#bdc9e1", cex = 1.5
)
points(RootLeafPCA$x[which(PCA_NEW$Myco == "EM"), 1],
    -RootLeafPCA$x[which(PCA_NEW$Myco == "EM"), 2],
    pch = 20, col = "#045a8d", cex = 1.5
)
points(RootLeafPCA$x[which(PCA_NEW$Myco == "NM"), 1],
    -RootLeafPCA$x[which(PCA_NEW$Myco == "NM"), 2],
    pch = 20, col = "#252525", cex = 1.5
)
points(RootLeafPCA$x[which(PCA_NEW$Myco == "EM+AM"), 1],
    -RootLeafPCA$x[which(PCA_NEW$Myco == "EM+AM"), 2],
    pch = 20, col = "#88419d", cex = 1.5
)
points(RootLeafPCA$x[which(PCA_NEW$Myco == "ErM"), 1],
    -RootLeafPCA$x[which(PCA_NEW$Myco == "ErM"), 2],
    pch = 20, col = "#238b45", cex = 1.5
)
x0 <- c(0, 0, 0, 0, 0, 0)
y0 <- c(0, 0, 0, 0, 0, 0)
x1 <- RootLeafPCA$rotation[, 1] * 8
y1 <- RootLeafPCA$rotation[, 2] * 8
Arrows(x0, y0, x1, -y1,
    col = 1, lwd = 1, code = 2, arr.type = "triangle",
    arr.width = 0.2, arr.length = 0.2
)
text(x1 * c(1.15, 1.2, 1.2, 1.2), -y1 * c(1.2, 1.2, 1.2, 1.2), c("LMA", "LN", "RN", "D", "RTD", "SRL"), col = 1, cex = 1.7)
dev.off()

## Fit phylogenetic informed PCA

phylLeafRootPCA <- phyl.pca(tree.species.set, PCA_NEW[, c(2:7)], mode = "corr", method = "lambda")
summary(phylLeafRootPCA)
print(phylLeafRootPCA)

### Figure 3 A ########

# Figure 3: The plant economics space. Phylogenetically informed principal component analyses of the core species set (n=804)
# based on species mean trait values for A) woody (n=480) and non-woody (n=324) plant species

# pairwise PERMANOVA

PCA_NEW$woodiness2 <- PCA_NEW$Woodiness
PCA_NEW$woodiness2[which(PCA_NEW$woodiness2 == "non-woody/woody")] <- NA
pairwise.adonis(
    phylLeafRootPCA$S[, c(1, 2)][
        complete.cases(PCA_NEW$woodiness2),
    ],
    factors = PCA_NEW$woodiness2[
        complete.cases(PCA_NEW$woodiness2)
    ],
    sim.function = "vegdist",
    sim.method = "euclidian", p.adjust.m = "fdr"
)

# plot PCA

postscript("Figure 3 A.eps", width = 9.0, height = 9.0)
plot(phylLeafRootPCA$S[-which(PCA_NEW$Woodiness == "unknown"), 1],
    -phylLeafRootPCA$S[-which(PCA_NEW$Woodiness == "unknown"), 2],
    pch = 19,
    xlab = "PC1", ylab = "PC2", main = "woodiness", xlim = c(-100, 100),
    ylim = c(-100, 100), cex = 0.5
)
points(phylLeafRootPCA$S[which(PCA_NEW$Woodiness == "woody"), 1],
    -phylLeafRootPCA$S[which(PCA_NEW$Woodiness == "woody"), 2],
    pch = 20, col = "#bae4b3", cex = 1.5
)
points(phylLeafRootPCA$S[which(PCA_NEW$Woodiness == "non-woody"), 1],
    -phylLeafRootPCA$S[which(PCA_NEW$Woodiness == "non-woody"), 2],
    pch = 20, col = "#238b45", cex = 1.5
)
x0 <- c(0, 0, 0, 0, 0, 0)
y0 <- c(0, 0, 0, 0, 0, 0)
x1 <- phylLeafRootPCA$L[, 1] * 80
y1 <- phylLeafRootPCA$L[, 2] * 80
Arrows(x0, y0, x1, -y1,
    col = 1, lwd = 1, code = 2, arr.type = "triangle",
    arr.width = 0.2, arr.length = 0.2
)
text(x1 * c(1.15, 1.2, 1.2, 1.2), -y1 * c(1.2, 1.2, 1.2, 1.2), c("LMA", "LN", "RN", "D", "RTD", "SRL"), col = 1, cex = 1.7)
dev.off()

### Figure 3 B ########

# Figure 3: The plant economics space. Phylogenetically informed principal component analyses of the core species set (n=804)
# based on species mean trait values for B) arbuscular mycorrhizal species (AM, n=630), ectomycorrhizal mycorrhizal species (EM, n=84),
# arbuscular and ectomycorrhizal species (EM-AM = 15), ericoid mycorrhizal species (ErM, n=12), or non-mycorrhizal species (NM, n= 63)

# pairwise PERMANOVA

PCA_NEW$Myco2 <- PCA_NEW$Myco
PCA_NEW$Myco2[which(PCA_NEW$Myco2 == "unknown")] <- NA
pairwise.adonis(
    phylLeafRootPCA$S[, c(1, 2)][
        complete.cases(PCA_NEW$Myco2),
    ],
    factors = PCA_NEW$Myco2[
        complete.cases(PCA_NEW$Myco2)
    ],
    sim.function = "vegdist",
    sim.method = "euclidian", p.adjust.m = "fdr"
)

# plot PCA

postscript("Figure 3 B.eps", width = 9.0, height = 9.0)
plot(phylLeafRootPCA$S[-which(PCA_NEW$Myco == "unknown"), 1],
    -phylLeafRootPCA$S[-which(PCA_NEW$Myco == "unknown"), 2],
    pch = 19,
    xlab = "PC1", ylab = "PC2", main = "Myco", xlim = c(-100, 100),
    ylim = c(-100, 100), cex = 0.5
)
points(phylLeafRootPCA$S[which(PCA_NEW$Myco == "AM"), 1],
    -phylLeafRootPCA$S[which(PCA_NEW$Myco == "AM"), 2],
    pch = 20, col = "#bdc9e1", cex = 1.5
)
points(phylLeafRootPCA$S[which(PCA_NEW$Myco == "EM"), 1],
    -phylLeafRootPCA$S[which(PCA_NEW$Myco == "EM"), 2],
    pch = 20, col = "#045a8d", cex = 1.5
)
points(phylLeafRootPCA$S[which(PCA_NEW$Myco == "NM"), 1],
    -phylLeafRootPCA$S[which(PCA_NEW$Myco == "NM"), 2],
    pch = 20, col = "#252525", cex = 1.5
)
points(phylLeafRootPCA$S[which(PCA_NEW$Myco == "EM+AM"), 1],
    -phylLeafRootPCA$S[which(PCA_NEW$Myco == "EM+AM"), 2],
    pch = 20, col = "#fb00de", cex = 1.5
)
points(phylLeafRootPCA$S[which(PCA_NEW$Myco == "ErM"), 1],
    -phylLeafRootPCA$S[which(PCA_NEW$Myco == "ErM"), 2],
    pch = 20, col = "#0ae1e1", cex = 1.5
)
x0 <- c(0, 0, 0, 0, 0, 0)
y0 <- c(0, 0, 0, 0, 0, 0)
x1 <- phylLeafRootPCA$L[, 1] * 80
y1 <- phylLeafRootPCA$L[, 2] * 80
Arrows(x0, y0, x1, -y1,
    col = 1, lwd = 1, code = 2, arr.type = "triangle",
    arr.width = 0.2, arr.length = 0.2
)
text(x1 * c(1.15, 1.2, 1.2, 1.2), -y1 * c(1.2, 1.2, 1.2, 1.2), c("LMA", "LN", "RN", "D", "RTD", "SRL"), col = 1, cex = 1.7)
dev.off()

### Figure 3 C ########

# Figure 3: The plant economics space. Phylogenetically informed principal component analyses of the core species set (n=804)
# based on species mean trait values for C) non-N-fixing (n=739) and N-fixing plant species (n=65).

# pairwise PERMANOVA

PCA_NEW$N_Fix2 <- PCA_NEW$N_Fix
PCA_NEW$N_Fix2[which(PCA_NEW$N_Fix2 == "unknown")] <- NA
pairwise.adonis(
    phylLeafRootPCA$S[, c(1, 2)][
        complete.cases(PCA_NEW$N_Fix2),
    ],
    factors = PCA_NEW$N_Fix2[
        complete.cases(PCA_NEW$N_Fix2)
    ],
    sim.function = "vegdist",
    sim.method = "euclidian", p.adjust.m = "fdr"
)

# plot PCA

postscript("Figure 3 C.eps", width = 9.0, height = 9.0)
plot(phylLeafRootPCA$S[-which(PCA_NEW$N_Fix == "unknown"), 1],
    -phylLeafRootPCA$S[-which(PCA_NEW$N_Fix == "unknown"), 2],
    pch = 19,
    xlab = "PC1", ylab = "PC2", main = "N_Fixation", xlim = c(-100, 100),
    ylim = c(-90, 90), cex = 0.5
)
points(phylLeafRootPCA$S[which(PCA_NEW$N_Fix == "None"), 1],
    -phylLeafRootPCA$S[which(PCA_NEW$N_Fix == "None"), 2],
    pch = 20, col = "#fdcc8a", cex = 1.5
)
points(phylLeafRootPCA$S[which(PCA_NEW$N_Fix == "N-fixing"), 1],
    -phylLeafRootPCA$S[which(PCA_NEW$N_Fix == "N-fixing"), 2],
    pch = 20, col = "#e34a33", cex = 1.5
)
x0 <- c(0, 0, 0, 0, 0, 0)
y0 <- c(0, 0, 0, 0, 0, 0)
x1 <- phylLeafRootPCA$L[, 1] * 80
y1 <- phylLeafRootPCA$L[, 2] * 80
Arrows(x0, y0, x1, -y1,
    col = 1, lwd = 1, code = 2, arr.type = "triangle",
    arr.width = 0.2, arr.length = 0.2
)
text(x1 * c(1.15, 1.2, 1.2, 1.2), -y1 * c(1.2, 1.2, 1.2, 1.2), c("LMA", "LN", "RN", "D", "RTD", "SRL"), col = 1, cex = 1.7)
dev.off()

### Supporting information Figure S4 A ########

# Supporting information Figure S4: Phylogenetically informed principal component analyses of the core species set of Figure 3 (total 804 species)
# for A) non-woody plant species (n=324).

## Fit phylogenetic informed PCA onlx for the subset of non-woody species

phylPCA_NEW_Nonwoody <- phyl.pca(tree.species.set_nonwoody, PCA_NEW_Nonwoody[, c(2:7)], mode = "corr", method = "lambda")
summary(phylPCA_NEW_Nonwoody)
print(phylPCA_NEW_Nonwoody)

# plot PCA

postscript("Extended Figure 3 A.eps", width = 9.0, height = 9.0)
plot(phylPCA_NEW_Nonwoody$S[-which(PCA_NEW_Nonwoody$Woodiness == "woody"), 1],
    -phylPCA_NEW_Nonwoody$S[-which(PCA_NEW_Nonwoody$Woodiness == "woody"), 2],
    pch = 19,
    xlab = "PC1", ylab = "PC2", main = "Myco", xlim = c(-100, 100),
    ylim = c(-100, 100), cex = 0.5
)
points(phylPCA_NEW_Nonwoody$S[which(PCA_NEW_Nonwoody$Woodiness == "non-woody"), 1],
    -phylPCA_NEW_Nonwoody$S[which(PCA_NEW_Nonwoody$Woodiness == "non-woody"), 2],
    pch = 20, col = "#238b45", cex = 1.5
)
x0 <- c(0, 0, 0, 0, 0, 0)
y0 <- c(0, 0, 0, 0, 0, 0)
x1 <- phylPCA_NEW_Nonwoody$L[, 1] * 80
y1 <- phylPCA_NEW_Nonwoody$L[, 2] * 80
Arrows(x0, y0, x1, -y1,
    col = 1, lwd = 1, code = 2, arr.type = "triangle",
    arr.width = 0.2, arr.length = 0.2
)
text(x1 * c(1.15, 1.2, 1.2, 1.2), -y1 * c(1.2, 1.2, 1.2, 1.2), c("LMA", "LN", "RN", "D", "RTD", "SRL"), col = 1, cex = 1.7)
dev.off()

### Supporting information Figure S4 B ########

# Supporting information Figure S4: Phylogenetically informed principal component analyses of the core species set of Figure 3 (total 804 species)
# for B) woody plant species (n=480); Abbreviations as Extended Figure 1.

## Fit phylogenetic informed PCA onlx for the subset of woody species

phylPCA_woody <- phyl.pca(tree.species.set_woody, PCA_NEW_woody[, c(2:7)], mode = "corr", method = "lambda")
summary(phylPCA_woody)
print(phylPCA_woody)

# plot PCA

postscript("Extended Figure 3 B.eps", width = 9.0, height = 9.0)
plot(phylPCA_woody$S[-which(PCA_NEW_woody$Woodiness == "non-woody"), 1],
    -phylPCA_woody$S[-which(PCA_NEW_woody$Woodiness == "non-woody"), 2],
    pch = 19,
    xlab = "PC1", ylab = "PC2", main = "Myco", xlim = c(-100, 100),
    ylim = c(-100, 100), cex = 0.5
)
points(phylPCA_woody$S[which(PCA_NEW_woody$Woodiness == "woody"), 1],
    -phylPCA_woody$S[which(PCA_NEW_woody$Woodiness == "woody"), 2],
    pch = 20, col = "#bae4b3", cex = 1.5
)
x0 <- c(0, 0, 0, 0, 0, 0)
y0 <- c(0, 0, 0, 0, 0, 0)
x1 <- phylPCA_woody$L[, 1] * 80
y1 <- phylPCA_woody$L[, 2] * 80
Arrows(x0, y0, x1, -y1,
    col = 1, lwd = 1, code = 2, arr.type = "triangle",
    arr.width = 0.2, arr.length = 0.2
)
text(x1 * c(1.15, 1.2, 1.2, 1.2), -y1 * c(1.2, 1.2, 1.2, 1.2), c("LMA", "LN", "RN", "D", "RTD", "SRL"), col = 1, cex = 1.7)
dev.off()


### V) Pairwise correlation ###################################################

rm(list = ls())

# load trait data

Core_combined_meta <- read.csv("Core_combined_meta.csv", header = T, sep = ";", na.strings = c("", "NA"))

PCA_NEW <- Core_combined_meta[, c(1:3, 10, 11, 13, 14)]

# remove Specis for which no information on core traits is available

PCA_NEW_NAs <- PCA_NEW[which(is.na(PCA_NEW$LMA) & is.na(PCA_NEW$LN) & is.na(PCA_NEW$RN) & is.na(PCA_NEW$MRD) & is.na(PCA_NEW$RTD) & is.na(PCA_NEW$SRL)), ]
PCA_NEW_NAs <- PCA_NEW_NAs$full_species
PCA_NEW <- PCA_NEW %>%
    dplyr::filter(!full_species %in% PCA_NEW_NAs)

# change colnames

colnames(PCA_NEW)[3] <- "LeafN"
colnames(PCA_NEW)[4] <- "RootN"
colnames(PCA_NEW)[5] <- "RDia"

# Assign species to related sister species for the purpose of phylogenetic position in tree

PCA_NEW$full_species <- gsub("Caryota_maxima", "Caryota_mitis", PCA_NEW$full_species) # will be assigned to mites only for the purpose of phylognetic position in the tree
PCA_NEW$full_species <- gsub("Quintinia_serrata", "Quintinia_verdonii", PCA_NEW$full_species) # will be assigned to verdonii only for the purpose of phylognetic position in the tree
PCA_NEW$full_species <- gsub("Rhipogonum_scandens", "Smilax_ripogonum", PCA_NEW$full_species) # will be assigned to Smilax ripogonum only for the purpose of phylognetic position in tree
PCA_NEW$full_species <- gsub("Abies_balsamea", "Abies_fraseri", PCA_NEW$full_species) # will be assigned to fraseri only for the purpose of phylognetic position in the tree

# extract phylogenetic information on family names for all species in the PCA data set from ncbi (see https://www.ncbi.nlm.nih.gov/)
# this approach was recommended from the author of the package "brranching" after reporting errors related to the function "phylomatic" that is usally used to
# extract phylogeny (see GitHub https://github.com/ropensci/brranching/issues/42)

library(rentrez)

set_entrez_key("YOUR_KEY") # enter your API key
Sys.getenv("ENTREZ_KEY")
# API key from ncbi (see https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)

names_half_1 <- PCA_NEW[c(1:1000), ]
names_half_2 <- PCA_NEW[c(1001:2406), ]

my_names_phyl1 <- phylomatic_names(names_half_1$full_species, db = "ncbi") # manually select plant family names
my_names_phyl2 <- phylomatic_names(names_half_2$full_species, db = "ncbi") # manually select plant family names

PCA_Species_PhylomaticNames1 <- data.frame(my_names_phyl1)
PCA_Species_PhylomaticNames2 <- data.frame(my_names_phyl2)
colnames(PCA_Species_PhylomaticNames1)[1] <- "full_species"
colnames(PCA_Species_PhylomaticNames2)[1] <- "full_species"

PCA_Species_Pairwise <- rbind(PCA_Species_PhylomaticNames1, PCA_Species_PhylomaticNames2)

PCA_Species_Pairwise$ID <- seq.int(nrow(PCA_Species_Pairwise))
PCA_NEW$ID <- seq.int(nrow(PCA_NEW))
colnames(PCA_Species_Pairwise)[1] <- "full_species_ncbi"
PCA_NEW_2 <- merge(PCA_NEW, PCA_Species_Pairwise, by = "ID")
PCA_NEW_2$ID <- NULL
PCA_NEW_pairwise <- PCA_NEW_2

# write.table(PCA_NEW_pairwise,file="PCA_pairwise_FullPhylo.csv",sep=";") # save PCA data with full phylogenetic information
PCA_NEW_pairwise <- read.csv("PCA_pairwise_FullPhylo.csv", header = T, sep = ";", na.strings = c("", "NA"))

# rename data

PCA_NEW <- PCA_NEW_pairwise

# clear workspace except for PCA_NEW dataset

rm(list = setdiff(ls(), "PCA_NEW"))

# remove one remaining clubmoss and correct ony typo

PCA_NEW <- PCA_NEW %>% dplyr::filter(!full_species == "Huperzia_selago")
PCA_NEW$full_species[PCA_NEW$full_species == "Monroa_squarrosa"] <- "Munroa_squarrosa"
PCA_NEW$full_species_ncbi[PCA_NEW$full_species_ncbi == "NA/monroa_squarrosa/monroa_squarrosa"] <- "poaceae/munroa_squarrosa/munroa_squarrosa"

# Extract phylogeny

# For large queries of data we need to use the function phylomatic_local() from the package "brranching"
# the function phylomatic() will not work for our data
# However, phylomatic_local() only works on Mac computers. Thus, we saved the output separetely in case you are not using a Mac.
# In that case you can skip this line of code and continue with loading the "treefull" phylo object

treefull <- phylomatic_local(PCA_NEW$full_species_ncbi, taxnames = FALSE, storedtree = "zanne2014")

# load phylo object

load("treefull.RData")

# capitalize and remove underscore on all tip.labels

treefull$tip.label <- Hmisc::capitalize(gsub("_", " ", treefull$tip.label))
treefull$tip.label <- gsub(" ", "_", treefull$tip.label)

# which species were not matched?

setdiff(PCA_NEW$full_species, treefull[["tip.label"]])

# Add missing species
# For the phylogenetic correction we assigned missing species to a closely related species from the same genus within the tree

treefull <- phangorn::add.tips(treefull,
    tips = c("Oxybasis_glauca"), edge.length = 1,
    where = which(treefull$tip.label == "Atriplex_prostrata")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Chondrosum_gracile"), edge.length = 1,
    where = which(treefull$tip.label == "Bouteloua_curtipendula")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Leiospermum_racemosum"), edge.length = 1,
    where = which(treefull$tip.label == "Caldcluvia_rosifolia")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Bassia_dasyphylla"), edge.length = 1,
    where = which(treefull$tip.label == "Bassia_prostrata")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Chenopodium_ficifolium"), edge.length = 1,
    where = which(treefull$tip.label == "Chenopodium_album")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Chenopodium_fremontii"), edge.length = 1,
    where = which(treefull$tip.label == "Chenopodium_album")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Cordia_dodecandra"), edge.length = 1,
    where = which(treefull$tip.label == "Cordia_alliodora")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Corispermum_heptapotamicum"), edge.length = 1,
    where = which(treefull$tip.label == "Corispermum_mongolicum")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Corispermum_puberulum"), edge.length = 1,
    where = which(treefull$tip.label == "Corispermum_mongolicum")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Haloxylon_scoparium"), edge.length = 1,
    where = which(treefull$tip.label == "Haloxylon_ammodendron")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Nama_dichotomum"), edge.length = 1,
    where = which(treefull$tip.label == "Hydrophyllum_canadense")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Nama_hispidum"), edge.length = 1,
    where = which(treefull$tip.label == "Hydrophyllum_canadense")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Petrosimonia_sibirica"), edge.length = 1,
    where = which(treefull$tip.label == "Haloxylon_ammodendron")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Pittosporopsis_kerrii"), edge.length = 1,
    where = which(treefull$tip.label == "Hydrophyllum_canadense")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Salsola_orientalis"), edge.length = 1,
    where = which(treefull$tip.label == "Salsola_laricifolia")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Abies_sachalinensis"), edge.length = 1,
    where = which(treefull$tip.label == "Abies_nephrolepis")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Manglietia_paruicula"), edge.length = 1,
    where = which(treefull$tip.label == "Magnolia_grandiflora")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Chenopodium_vulvaria"), edge.length = 1,
    where = which(treefull$tip.label == "Chenopodium_album")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Chondrosum_simplex"), edge.length = 1,
    where = which(treefull$tip.label == "Bouteloua_curtipendula")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Cyanus_cheiranthifolius"), edge.length = 1,
    where = which(treefull$tip.label == "Centaurea_aspera")
)
treefull <- phangorn::add.tips(treefull,
    tips = c("Kemulariella_caucasica"), edge.length = 1,
    where = which(treefull$tip.label == "Aster_diplostephioides")
)

# plot tree

postscript("Phylogenetic_Tree_FullDataset.eps", width = 9.0, height = 9.0)
plotTree(treefull, type = "fan", fsize = 0.1, lwd = 1, ftype = "i")
dev.off()

# See which species don't match original species list and phylogeny tips

setdiff(treefull$tip.label, PCA_NEW$full_species) # should be character(0)
setdiff(PCA_NEW$full_species, treefull$tip.label) # should be character(0)

# create caper comparative data objects

treefull$node.label <- NULL # fixes error in compData
compData <- comparative.data(phy = treefull, data = PCA_NEW, names.col = full_species, vcv = T, na.omit = F, warn.dropped = T)

# trait correlations

correlation.matrix <- matrix(nrow = 6, ncol = 6)
colnames(correlation.matrix) <- c("LMA", "LeafN", "RootN", "RDia", "RTD", "SRL")
rownames(correlation.matrix) <- c("LMA", "LeafN", "RootN", "RDia", "RTD", "SRL")
diag(correlation.matrix) <- 1
sample.sizes <- correlation.matrix
diag(sample.sizes) <- NA

### Supporting information Figure S2 ########

# Supporting information Figure S2: Pairwise correlation of all traits used in the analysis. Scatterplots represent species mean trait correlations after correction for
# study design and publication.
# (This Figure was manually arranged using the graphical software corel)

## Model for RTD ~ RootN

summary(m1 <- pgls(RTD ~ RootN,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[5, 3] <- correlation.matrix[3, 5] <-
    sqrt(as.numeric(as.character(summary(m1)[12]))) *
        if (summary(m1)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[5, 3] <- sample.sizes[3, 5] <- summary(m1)$df[2] + 2

# simple linear regression

summary(lm(PCA_NEW$RTD ~ PCA_NEW$RootN))

# plot correlation

postscript("Pair1.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$RTD, PCA_NEW$RootN,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "RTD",
    ylab = "RootN"
)
abline(lm(PCA_NEW$RTD ~ PCA_NEW$RootN), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m1)$coefficient[1, 1] + start.x * summary(m1)$coefficient[2, 1]
end.y <- summary(m1)$coefficient[1, 1] + end.x * summary(m1)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

## Model for RTD ~ LMA

summary(m2 <- pgls(RTD ~ LMA,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[5, 1] <- correlation.matrix[1, 5] <-
    sqrt(as.numeric(as.character(summary(m2)[12]))) *
        if (summary(m2)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[5, 1] <- sample.sizes[1, 5] <- summary(m2)$df[2] + 2

# plot correlation

postscript("Pair2.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$RTD, PCA_NEW$LMA,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "RTD",
    ylab = "LMA"
)
abline(lm(PCA_NEW$RTD ~ PCA_NEW$LMA), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m2)$coefficient[1, 1] + start.x * summary(m2)$coefficient[2, 1]
end.y <- summary(m2)$coefficient[1, 1] + end.x * summary(m2)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$RTD ~ PCA_NEW$LMA))

## Model for RN ~ LeafN

summary(m3 <- pgls(LeafN ~ RootN,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[3, 2] <- correlation.matrix[2, 3] <-
    sqrt(as.numeric(as.character(summary(m3)[12]))) *
        if (summary(m3)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[3, 2] <- sample.sizes[2, 3] <- summary(m3)$df[2] + 2

# plot correlation

postscript("Pair3.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$RootN, PCA_NEW$LeafN,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "RootN",
    ylab = "LeafN"
)
abline(lm(PCA_NEW$RootN ~ PCA_NEW$LeafN), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m3)$coefficient[1, 1] + start.x * summary(m3)$coefficient[2, 1]
end.y <- summary(m3)$coefficient[1, 1] + end.x * summary(m3)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$LeafN ~ PCA_NEW$RootN))

## Model for RN ~ LMA

summary(m4 <- pgls(RootN ~ LMA,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[3, 1] <- correlation.matrix[1, 3] <-
    sqrt(as.numeric(as.character(summary(m4)[12]))) *
        if (summary(m4)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[3, 1] <- sample.sizes[1, 3] <- summary(m4)$df[2] + 2

# plot correlation

postscript("Pair4.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$RootN, PCA_NEW$LMA,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "RootN",
    ylab = "LMA"
)
abline(lm(PCA_NEW$RootN ~ PCA_NEW$LMA), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m4)$coefficient[1, 1] + start.x * summary(m4)$coefficient[2, 1]
end.y <- summary(m4)$coefficient[1, 1] + end.x * summary(m4)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$RootN ~ PCA_NEW$LMA))

## Model for RN ~ RDia

summary(m5 <- pgls(RootN ~ RDia,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[3, 4] <- correlation.matrix[4, 3] <-
    sqrt(as.numeric(as.character(summary(m5)[12]))) *
        if (summary(m5)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[3, 4] <- sample.sizes[4, 3] <- summary(m5)$df[2] + 2

# plot correlation

postscript("Pair5.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$RDia, PCA_NEW$RootN,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "RDia",
    ylab = "RootN"
)
abline(lm(PCA_NEW$RDia ~ PCA_NEW$RootN), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m5)$coefficient[1, 1] + start.x * summary(m5)$coefficient[2, 1]
end.y <- summary(m5)$coefficient[1, 1] + end.x * summary(m5)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$RootN ~ PCA_NEW$RDia))

## Model for RN ~ SRL

summary(m6 <- pgls(RootN ~ SRL,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[6, 3] <- correlation.matrix[3, 6] <-
    sqrt(as.numeric(as.character(summary(m6)[12]))) *
        if (summary(m6)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[6, 3] <- sample.sizes[3, 6] <- summary(m6)$df[2] + 2

# plot correlation

postscript("Pair6.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$SRL, PCA_NEW$RootN,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "SRL",
    ylab = "RootN"
)
abline(lm(PCA_NEW$SRL ~ PCA_NEW$RootN), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m6)$coefficient[1, 1] + start.x * summary(m6)$coefficient[2, 1]
end.y <- summary(m6)$coefficient[1, 1] + end.x * summary(m6)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$RootN ~ PCA_NEW$SRL))

## Model for LMA ~ LN

summary(m7 <- pgls(LMA ~ LeafN,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[2, 1] <- correlation.matrix[1, 2] <-
    sqrt(as.numeric(as.character(summary(m7)[12]))) *
        if (summary(m7)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[2, 1] <- sample.sizes[1, 2] <- summary(m7)$df[2] + 2

# plot correlation

postscript("Pair7.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$LMA, PCA_NEW$LeafN,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "LMA",
    ylab = "LeafN"
)
abline(lm(PCA_NEW$LMA ~ PCA_NEW$LeafN), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m7)$coefficient[1, 1] + start.x * summary(m7)$coefficient[2, 1]
end.y <- summary(m7)$coefficient[1, 1] + end.x * summary(m7)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$LMA ~ PCA_NEW$LeafN))

## Model for LMA ~ RDia

summary(m8 <- pgls(LMA ~ RDia,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[4, 1] <- correlation.matrix[1, 4] <-
    sqrt(as.numeric(as.character(summary(m8)[12]))) *
        if (summary(m8)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[4, 1] <- sample.sizes[1, 4] <- summary(m8)$df[2] + 2

# plot correlation

postscript("Pair8.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$RDia, PCA_NEW$LMA,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "RDia",
    ylab = "LMA"
)
abline(lm(PCA_NEW$RDia ~ PCA_NEW$LMA), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m8)$coefficient[1, 1] + start.x * summary(m8)$coefficient[2, 1]
end.y <- summary(m8)$coefficient[1, 1] + end.x * summary(m8)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$LMA ~ PCA_NEW$RDia))

## Model for LMA ~ SRL

summary(m9 <- pgls(LMA ~ SRL,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[6, 1] <- correlation.matrix[1, 6] <-
    sqrt(as.numeric(as.character(summary(m9)[12]))) *
        if (summary(m9)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[6, 1] <- sample.sizes[1, 6] <- summary(m9)$df[2] + 2

# plot correlation

postscript("Pair9.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$SRL, PCA_NEW$LMA,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "SRL",
    ylab = "LMA"
)
abline(lm(PCA_NEW$SRL ~ PCA_NEW$LMA), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m9)$coefficient[1, 1] + start.x * summary(m9)$coefficient[2, 1]
end.y <- summary(m9)$coefficient[1, 1] + end.x * summary(m9)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$LMA ~ PCA_NEW$SRL))

## Model for LeafN ~ RDia

summary(m10 <- pgls(LeafN ~ RDia,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[4, 2] <- correlation.matrix[2, 4] <-
    sqrt(as.numeric(as.character(summary(m10)[12]))) *
        if (summary(m10)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[4, 2] <- sample.sizes[2, 4] <- summary(m10)$df[2] + 2

# plot correlation

postscript("Pair10.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$RDia, PCA_NEW$LeafN,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "RDia",
    ylab = "LeafN"
)
abline(lm(PCA_NEW$RDia ~ PCA_NEW$LeafN), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -5
end.x <- 4
start.y <- summary(m10)$coefficient[1, 1] + start.x * summary(m10)$coefficient[2, 1]
end.y <- summary(m10)$coefficient[1, 1] + end.x * summary(m10)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$LeafN ~ PCA_NEW$RDia))

## Model for LeafN ~ RTD

summary(m11 <- pgls(LeafN ~ RTD,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[5, 2] <- correlation.matrix[2, 5] <-
    sqrt(as.numeric(as.character(summary(m11)[12]))) *
        if (summary(m11)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[5, 2] <- sample.sizes[2, 5] <- summary(m11)$df[2] + 2

# plot correlation

postscript("Pair11.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$RTD, PCA_NEW$LeafN,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "RTD",
    ylab = "LeafN"
)
abline(lm(PCA_NEW$RTD ~ PCA_NEW$LeafN), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -5
end.x <- 4
start.y <- summary(m11)$coefficient[1, 1] + start.x * summary(m11)$coefficient[2, 1]
end.y <- summary(m11)$coefficient[1, 1] + end.x * summary(m11)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$LeafN ~ PCA_NEW$RTD))

## Model for LeafN ~ SRL

summary(m12 <- pgls(LeafN ~ SRL,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[6, 2] <- correlation.matrix[2, 6] <-
    sqrt(as.numeric(as.character(summary(m12)[12]))) *
        if (summary(m12)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[6, 2] <- sample.sizes[2, 6] <- summary(m12)$df[2] + 2

# plot correlation

postscript("Pair12.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$SRL, PCA_NEW$LeafN,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "SRL",
    ylab = "LeafN"
)
abline(lm(PCA_NEW$SRL ~ PCA_NEW$LeafN), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -5
end.x <- 4
start.y <- summary(m12)$coefficient[1, 1] + start.x * summary(m12)$coefficient[2, 1]
end.y <- summary(m12)$coefficient[1, 1] + end.x * summary(m12)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$LeafN ~ PCA_NEW$SRL))

## Model for RDia ~ RTD

summary(m13 <- pgls(RDia ~ RTD,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[4, 5] <- correlation.matrix[5, 4] <-
    sqrt(as.numeric(as.character(summary(m13)[12]))) *
        if (summary(m13)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[4, 5] <- sample.sizes[5, 4] <- summary(m13)$df[2] + 2

# plot correlation

postscript("Pair13.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$RDia, PCA_NEW$RTD,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "RDia",
    ylab = "RTD"
)
abline(lm(PCA_NEW$RDia ~ PCA_NEW$RTD), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m13)$coefficient[1, 1] + start.x * summary(m13)$coefficient[2, 1]
end.y <- summary(m13)$coefficient[1, 1] + end.x * summary(m13)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$RDia ~ PCA_NEW$RTD))

## Model for RDia ~ SRL

summary(m14 <- pgls(RDia ~ SRL,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[6, 4] <- correlation.matrix[4, 6] <-
    sqrt(as.numeric(as.character(summary(m14)[12]))) *
        if (summary(m14)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[6, 4] <- sample.sizes[4, 6] <- summary(m14)$df[2] + 2

# plot correlation

postscript("Pair14.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$RDia, PCA_NEW$SRL,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "RDia",
    ylab = "SRL"
)
abline(lm(PCA_NEW$RDia ~ PCA_NEW$SRL), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m14)$coefficient[1, 1] + start.x * summary(m14)$coefficient[2, 1]
end.y <- summary(m14)$coefficient[1, 1] + end.x * summary(m14)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$RDia ~ PCA_NEW$SRL))

## Model for RTD ~ SRL

summary(m15 <- pgls(RTD ~ SRL,
    data = compData,
    lambda = "ML", delta = "ML"
))
correlation.matrix[6, 5] <- correlation.matrix[5, 6] <-
    sqrt(as.numeric(as.character(summary(m15)[12]))) *
        if (summary(m15)$coefficients[2, 1] < 0) {
            -1
        } else {
            1
        }
sample.sizes[6, 5] <- sample.sizes[5, 6] <- summary(m15)$df[2] + 2

# plot correlation

postscript("Pair15.eps", width = 9.0, height = 9.0)
par(pty = "s")
plot(PCA_NEW$SRL, PCA_NEW$RTD,
    pch = 20, col = rgb(0, 114 / 255, 178 / 255), cex = 0.2, xlab = "SRL",
    ylab = "RTD"
)
abline(lm(PCA_NEW$SRL ~ PCA_NEW$RTD), col = rgb(0, 114 / 255, 178 / 255))
start.x <- -4
end.x <- 4
start.y <- summary(m15)$coefficient[1, 1] + start.x * summary(m15)$coefficient[2, 1]
end.y <- summary(m15)$coefficient[1, 1] + end.x * summary(m15)$coefficient[2, 1]
lines(x = c(start.x, end.x), y = c(start.y, end.y), lwd = 2)
dev.off()

# simple linear regression

summary(lm(PCA_NEW$RTD ~ PCA_NEW$SRL))


###############################################################################################################################
# Individual PES data
###############################################################################################################################


Indi_PCA <- read.csv("Weigelt_et_al_2021_Individal.PCA.Matrix.csv", header = T, sep = ";", na.strings = c("", "NA"))

# clear workspace except for Indi_PCA dataset

rm(list = setdiff(ls(), "Indi_PCA"))

# Extract phylogeny

tree.species.set <- phylomatic(Indi_PCA$Species_ncbi, taxnames = FALSE, get = "POST", storedtree = "zanne2014")
plot(tree.species.set, type = "f", cex = 0.1)

# add missing species to the phylogenetic tree

tree.species.set <- phangorn::add.tips(tree.species.set, tips = c("Chenopodium_glaucum"), edge.length = 1, where = which(tree.species.set$tip.label == "Bassia_prostrata"))
tree.species.set <- phangorn::add.tips(tree.species.set, tips = c("Manglietia_paruicula"), edge.length = 1, where = which(tree.species.set$tip.label == "Magnolia_grandiflora"))

# capitalize and remove underscore on all tip.labels

tree.species.set$tip.label <- Hmisc::capitalize(gsub("_na", "_NA", tree.species.set$tip.label))
tree.species.set$tip.label <- Hmisc::capitalize(gsub("_NAnmu", "_nanmu", tree.species.set$tip.label))

# clean and sort dataframes and tree

p.dist.mat <- cophenetic(tree.species.set) # trick to ensure correct order
row.names(Indi_PCA) <- Indi_PCA$Species
Indi_PCA <- Indi_PCA[row.names(p.dist.mat), ] # trick to ensure correct order

### Supporting information Figure S9 ########

# Supporting information Figure S9: Non-phylogenetically informed principal component analysis of traits measured on the same individuals

RootLeafPCA <- prcomp(Indi_PCA[, c(2:7)], center = TRUE, scale. = TRUE)
summary(RootLeafPCA)
axes <- predict(RootLeafPCA)

eigenvalues <- RootLeafPCA$sdev^2

# plot PCA

postscript("Extended Figure 5.eps", width = 9.0, height = 9.0)
plot(RootLeafPCA$x[-which(Indi_PCA$Myco == "unknown"), 1],
    -RootLeafPCA$x[-which(Indi_PCA$Myco == "unknown"), 2],
    pch = 19,
    xlab = "PC1", ylab = "PC2", main = "Myco", xlim = c(-7, 7),
    ylim = c(-7, 7), cex = 0.5
)
points(RootLeafPCA$x[which(Indi_PCA$Myco == "AM"), 1],
    -RootLeafPCA$x[which(Indi_PCA$Myco == "AM"), 2],
    pch = 20, col = "#bdc9e1", cex = 1.5
)
points(RootLeafPCA$x[which(Indi_PCA$Myco == "EM"), 1],
    -RootLeafPCA$x[which(Indi_PCA$Myco == "EM"), 2],
    pch = 20, col = "#045a8d", cex = 1.5
)
points(RootLeafPCA$x[which(Indi_PCA$Myco == "NM"), 1],
    -RootLeafPCA$x[which(Indi_PCA$Myco == "NM"), 2],
    pch = 20, col = "#252525", cex = 1.5
)
points(RootLeafPCA$x[which(Indi_PCA$Myco == "EM+AM"), 1],
    -RootLeafPCA$x[which(Indi_PCA$Myco == "EM+AM"), 2],
    pch = 20, col = "#88419d", cex = 1.5
)
points(RootLeafPCA$x[which(Indi_PCA$Myco == "ErM"), 1],
    -RootLeafPCA$x[which(Indi_PCA$Myco == "ErM"), 2],
    pch = 20, col = "#238b45", cex = 1.5
)
x0 <- c(0, 0, 0, 0, 0, 0)
y0 <- c(0, 0, 0, 0, 0, 0)
x1 <- RootLeafPCA$rotation[, 1] * 8
y1 <- RootLeafPCA$rotation[, 2] * 8
Arrows(x0, y0, x1, -y1,
    col = 1, lwd = 1, code = 2, arr.type = "triangle",
    arr.width = 0.2, arr.length = 0.2
)
text(x1 * c(1.15, 1.2, 1.2, 1.2), -y1 * c(1.2, 1.2, 1.2, 1.2), c("LMA", "LN", "RN", "SRL", "RTD", "D"), col = 1, cex = 1.7)
dev.off()

### Figure 5 ########

# Figure 5: Phylogenetically informed principal component analyses of traits measured on the same individual showing arbuscular mycorrhizal species (AM, n=372),
# ericoid mycorrhizal species (ErM, n=3), ectomycorrhizal mycorrhizal species(EM, n=42), ectomycorrhizal/ arbuscular mycorrhizal species (EM-AM, n=5)
# or non-mycorrhizal species (NM, n=33) associated plant species (total n=455).

## Fit phylogenetic PCA

phylLeafRootPCA <- phyl.pca(tree.species.set, Indi_PCA[, c(2:7)], mode = "corr", method = "lambda")
summary(phylLeafRootPCA)
print(phylLeafRootPCA)

# plot PCA

postscript("Figure 6.eps", width = 9.0, height = 9.0)
plot(phylLeafRootPCA$S[-which(Indi_PCA$Myco == "unknown"), 1],
    -phylLeafRootPCA$S[-which(Indi_PCA$Myco == "unknown"), 2],
    pch = 19,
    xlab = "PC1", ylab = "PC2", main = "Myco", xlim = c(-100, 100),
    ylim = c(-100, 100), cex = 0.5
)
points(phylLeafRootPCA$S[which(Indi_PCA$Myco == "AM"), 1],
    -phylLeafRootPCA$S[which(Indi_PCA$Myco == "AM"), 2],
    pch = 20, col = "#bdc9e1", cex = 1.5
)
points(phylLeafRootPCA$S[which(Indi_PCA$Myco == "EM"), 1],
    -phylLeafRootPCA$S[which(Indi_PCA$Myco == "EM"), 2],
    pch = 20, col = "#045a8d", cex = 1.5
)
points(phylLeafRootPCA$S[which(Indi_PCA$Myco == "NM"), 1],
    -phylLeafRootPCA$S[which(Indi_PCA$Myco == "NM"), 2],
    pch = 20, col = "#252525", cex = 1.5
)
points(phylLeafRootPCA$S[which(Indi_PCA$Myco == "EM+AM"), 1],
    -phylLeafRootPCA$S[which(Indi_PCA$Myco == "EM+AM"), 2],
    pch = 20, col = "#88419d", cex = 1.5
)
points(phylLeafRootPCA$S[which(Indi_PCA$Myco == "ErM"), 1],
    -phylLeafRootPCA$S[which(Indi_PCA$Myco == "ErM"), 2],
    pch = 20, col = "#238b45", cex = 1.5
)
x0 <- c(0, 0, 0, 0, 0, 0)
y0 <- c(0, 0, 0, 0, 0, 0)
x1 <- phylLeafRootPCA$L[, 1] * 80
y1 <- phylLeafRootPCA$L[, 2] * 80
Arrows(x0, y0, x1, -y1,
    col = 1, lwd = 1, code = 2, arr.type = "triangle",
    arr.width = 0.2, arr.length = 0.2
)
text(x1 * c(1.15, 1.2, 1.2, 1.2), -y1 * c(1.2, 1.2, 1.2, 1.2), c("LMA", "LN", "RN", "SRL", "RTD", "D"), col = 1, cex = 1.7)
dev.off()

# pairwise PERMANOVA for woody/non-woody, mycorrhizal types and N-fixation ability

Indi_PCA$woodiness2 <- Indi_PCA$Woodiness
Indi_PCA$woodiness2[which(Indi_PCA$woodiness2 == "non-woody/woody")] <- NA
pairwise.adonis(
    phylLeafRootPCA$S[, c(1, 2)][
        complete.cases(Indi_PCA$woodiness2),
    ],
    factors = Indi_PCA$woodiness2[
        complete.cases(Indi_PCA$woodiness2)
    ],
    sim.function = "vegdist",
    sim.method = "euclidian", p.adjust.m = "fdr"
)

Indi_PCA$Myco2 <- Indi_PCA$Myco
pairwise.adonis(
    phylLeafRootPCA$S[, c(1, 2)][
        complete.cases(Indi_PCA$Myco2),
    ],
    factors = Indi_PCA$Myco2[
        complete.cases(Indi_PCA$Myco2)
    ],
    sim.function = "vegdist",
    sim.method = "euclidian", p.adjust.m = "fdr"
)

Indi_PCA$N_fixation2 <- Indi_PCA$N_fixation
Indi_PCA$N_fixation2[which(Indi_PCA$N_fixation2 == "unknown")] <- NA
pairwise.adonis(
    phylLeafRootPCA$S[, c(1, 2)][
        complete.cases(Indi_PCA$N_fixation2),
    ],
    factors = Indi_PCA$N_fixation2[
        complete.cases(Indi_PCA$N_fixation2)
    ],
    sim.function = "vegdist",
    sim.method = "euclidian", p.adjust.m = "fdr"
)


###############################################################################################################################
# PCA based on pairwise complete correlations
###############################################################################################################################

rm(list = ls())

# load data

Core_combined_meta <- read.csv("Weigelt_et_al_2021_Main.PCA.Matrix.csv", header = T, sep = ";", na.strings = c("", "NA"))

# rename data

data1 <- Core_combined_meta

### Supporting Information Figure S6 #############################

# Figure S6: Principal component analysis based on a correlation matrix of species mean values of root and leaf traits (species n=2510)
# expanding the six core traits (see Figure 3) to a set of 14 leaf and root traits.

data2 <- data1[, c(2, 3, 5:8, 10:15, 17:18)]

corr1 <- cor(data2, use = "pairwise.complete.obs")
eigen_res <- eigen(corr1)
eigen_res$values[eigen_res$values < 0] <- 1e-10 # produce regularized covariance matrix
corr1_reg <- eigen_res$vectors %*% diag(eigen_res$values) %*% t(eigen_res$vectors)
str(corr1_reg)
dimnames(corr1_reg) <- dimnames(corr1)

pca1 <- princomp(covmat = corr1_reg, cor = T)

# Proportion of variance explained

eigen_res$values / sum(eigen_res$values)

# plot the PCA

postscript("Figure 4.eps", width = 9.0, height = 9.0)
plot(pca1$loadings[, 2] ~ pca1$loadings[, 1])
text(pca1$loadings[, 1], pca1$loadings[, 2], dimnames(pca1$loadings)[[1]])
arrows(rep(0, 8), rep(0, 8), pca1$loadings[, 1], pca1$loadings[, 2], length = 0.1)
dev.off()

# Eigenvalues

zdat <- scale(data2) # this is just to standardize the original data, M = 0, SD =1
e1 <- eigen(corr1) # solving for the eigenvalues and eigenvectors from the correlation matrix


### Figure 4 #############################

# Figure 4: Principal component analysis based on a correlation matrix of species mean values of root and leaf traits (species n=2510)
# representing the six core traits (see Figure 3) together with overall plant size for (A) the first and second axes and (B) the third and fourth axis.

data3 <- data1[, c(2:4, 10, 11, 13, 14, 16)]
colnames(data3)[5] <- "RD"

corr1 <- cor(data3, use = "pairwise.complete.obs")
eigen_res <- eigen(corr1)
eigen_res$values[eigen_res$values < 0] <- 1e-10 # produce regularized covariance matrix
corr1_reg <- eigen_res$vectors %*% diag(eigen_res$values) %*% t(eigen_res$vectors)
str(corr1_reg)
dimnames(corr1_reg) <- dimnames(corr1)

pca2 <- princomp(covmat = corr1_reg, cor = T)

# Proportion of variance explained

eigen_res$values / sum(eigen_res$values)

## Figure A

postscript("Figure 5 A.eps", width = 9.0, height = 9.0)
plot(pca2$loadings[, 2] ~ pca2$loadings[, 1], ylim = c(-0.7, 0.7), xlim = c(-0.7, 0.7))
text(pca2$loadings[, 1], pca2$loadings[, 2], dimnames(pca2$loadings)[[1]])
arrows(rep(0, 8), rep(0, 8), pca2$loadings[, 1], pca2$loadings[, 2], length = 0.1)
dev.off()

## Figure B

postscript("Figure 5 B.eps", width = 9.0, height = 9.0)
plot(pca2$loadings[, 4] ~ pca2$loadings[, 3], ylim = c(-1, 1), xlim = c(-1, 1))
text(pca2$loadings[, 3], pca2$loadings[, 4], dimnames(pca2$loadings)[[1]])
arrows(rep(0, 8), rep(0, 8), pca2$loadings[, 3], pca2$loadings[, 4], length = 0.1)
dev.off()

# Eigenvalues

zdat <- scale(data3) # this is just to standardize the original data, M = 0, SD =1
e1 <- eigen(corr1) # solving for the eigenvalues and eigenvectors from the correlation matrix

### Supporting Information Figure S5 #############################

# Extended 3D version of Figure 4 (Interactive plot)

library(rgl)

plot3d(pca2$loadings[, 1], pca2$loadings[, 2], pca2$loadings[, 3], xlab = "PC1", ylab = "PC2", zlab = "PC3", expand = 1.2, box = F, type = "p")
text3d(pca2$loadings[, 1:3], texts = rownames(corr1_reg), font = 4, cex = 2, adj = c(-0.1, -0.01))
coords <- NULL
for (i in 1:nrow(pca2$loadings)) {
    coords <- rbind(coords, rbind(c(0, 0, 0), pca2$loadings[i, 1:3]))
}
lines3d(coords, col = "black", lwd = 4)


# Save 3D PCA as a gif

PCA3d <- function(size = 700, outputFile = "Supplement_data_file_3.gif") { #
    suppressPackageStartupMessages(library(rgl)) # opens rgl package
    ## set the size of the window
    open3d(windowRect = c(0, 0, size, size))
    plot3d(pca2$loadings[, 1], pca2$loadings[, 2], pca2$loadings[, 3], xlab = "PC1", ylab = "PC2", zlab = "PC3", expand = 1.2, box = F, type = "p") #
    text3d(pca2$loadings[, 1:3], texts = rownames(corr1_reg), font = 4, cex = 2, adj = c(-0.1, -0.01)) #
    coords <- NULL # PCA code
    for (i in 1:nrow(pca2$loadings)) { # - execute this code all at once
        coords <- rbind(coords, rbind(c(0, 0, 0), pca2$loadings[i, 1:3]))
    } #
    lines3d(coords, col = "black", lwd = 4) #
    outputFile <- sub(".gif$", "", outputFile)
    movie3d(spin3d(axis = c(0, 0, 1), rpm = 3), duration = 12, dir = getwd(), movie = outputFile) # save PCA as gif
}


### Supporting information Figure S7 #############################

# Supporting information Figure S7: Principal component analysis based on a correlation matrix of species mean values of root, leaf and stem traits (species n=2510).

data4 <- data1[, c(2:18)]

corr1 <- cor(data4, use = "pairwise.complete.obs")
eigen_res <- eigen(corr1)
eigen_res$values[eigen_res$values < 0] <- 1e-10 # produce regularized covariance matrix
corr1_reg <- eigen_res$vectors %*% diag(eigen_res$values) %*% t(eigen_res$vectors)
str(corr1_reg)
dimnames(corr1_reg) <- dimnames(corr1)

pca3 <- princomp(covmat = corr1_reg, cor = T)

# Proportion of variance explained

eigen_res$values / sum(eigen_res$values)

# plot the PCA

postscript("Extended Figure 6.eps", width = 9.0, height = 9.0)
plot(pca3$loadings[, 2] ~ pca3$loadings[, 1])
text(pca3$loadings[, 1], pca3$loadings[, 2], dimnames(pca3$loadings)[[1]])
arrows(rep(0, 8), rep(0, 8), pca3$loadings[, 1], pca3$loadings[, 2], length = 0.1)
dev.off()

# Eigenvalues

zdat <- scale(data4) # this is just to standardize the original data, M = 0, SD =1
e1 <- eigen(corr1) # solving for the eigenvalues and eigenvectors from the correlation matrix


### Supporting information Figure S8 #############################

# Supporting information Figure S8: Sensitivity analysis for data shown in figure 3 main text to test if using more species but the same number of traits
# as in the main analysis (right) or using more traits but the exact same species as in figure 3 (left) would change the outcome or
# our main analysis. Principal component analysis based on a correlation matrix of species mean values of root and leaf traits.
# Left: Trait selection as in Extended figure 5 but only using the 804 species for we also have full data coverage as in figure 3.

## Figure A

data5 <- data1[, c(1, 2:18)] # 2510 species for all traits
data6 <- data1[, c(1, 2, 3, 10, 11, 13, 14)] %>% drop_na() # 804 species for the 6 core traits (= main PCA of 6 core traits)
data6 <- data6$full_species # list of 804 species
data5 <- data5 %>% dplyr::filter(full_species %in% data6) # 804 species (same as for main PCA of 6 core traits) for all traits

data5 <- data5[, c(2:18)]

corr1 <- cor(data5, use = "pairwise.complete.obs")
eigen_res <- eigen(corr1)
eigen_res$values[eigen_res$values < 0] <- 1e-10 # produce regularized covariance matrix
corr1_reg <- eigen_res$vectors %*% diag(eigen_res$values) %*% t(eigen_res$vectors)
str(corr1_reg)
dimnames(corr1_reg) <- dimnames(corr1)

pca4 <- princomp(covmat = corr1_reg, cor = T)

# Proportion of variance explained

eigen_res$values / sum(eigen_res$values)

# plot the PCA

postscript("Extended Figure 7 A.eps", width = 9.0, height = 9.0)
plot(pca4$loadings[, 2] ~ pca4$loadings[, 1]) +
    text(pca4$loadings[, 1], pca4$loadings[, 2], dimnames(pca4$loadings)[[1]])
arrows(rep(0, 8), rep(0, 8), pca4$loadings[, 1], pca4$loadings[, 2], length = 0.1)
dev.off()

# Eigenvalues

zdat <- scale(data5) # this is just to standardize the original data, M = 0, SD =1
e1 <- eigen(corr1) # solving for the eigenvalues and eigenvectors from the correlation matrix


## Figure B

data7 <- data1[, c(2, 3, 10, 11, 13, 14)] %>% drop_na()

corr1 <- cor(data7, use = "pairwise.complete.obs")
eigen_res <- eigen(corr1)
eigen_res$values[eigen_res$values < 0] <- 1e-10 # produce regularized covariance matrix
corr1_reg <- eigen_res$vectors %*% diag(eigen_res$values) %*% t(eigen_res$vectors)
str(corr1_reg)
dimnames(corr1_reg) <- dimnames(corr1)

pca5 <- princomp(covmat = corr1_reg, cor = T)

# Proportion of variance explained

eigen_res$values / sum(eigen_res$values)

# plot the PCA

postscript("Extended Figure 7 B.eps", width = 9.0, height = 9.0)
plot(pca5$loadings[, 2] ~ pca5$loadings[, 1])
text(pca5$loadings[, 1], pca5$loadings[, 2], dimnames(pca5$loadings)[[1]])
arrows(rep(0, 8), rep(0, 8), pca5$loadings[, 1], pca5$loadings[, 2], length = 0.1)
dev.off()

# Eigenvalues

zdat <- scale(data7) # this is just to standardize the original data, M = 0, SD =1
e1 <- eigen(corr1) # solving for the eigenvalues and eigenvectors from the correlation matrix
str(e1)
