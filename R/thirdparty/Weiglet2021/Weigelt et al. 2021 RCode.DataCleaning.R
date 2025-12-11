# ---
# Publication title: An integrated framework of plant form and function: The belowground perspective
# Authors: Alexandra Weigelt, Liesje Mommer, Karl Andraczek, Colleen M. Iversen, Joana Bergmann, Helge Bruelheide, Ying Fan, GrC)goire T. Freschet, Nathaly R. Guerrero-RamC-rez, Jens Kattge, Thom W. Kuyper, Daniel C. Laughlin, Ina C. Meier, Fons van der Plas, Hendrik Poorter, Catherine Roumet, Jasper van Ruijven, Francesco Maria Sabatini, Marina Semchenko, Christopher J. Sweeney, Oscar J. Valverde-Barrantes, Larry M. York, M. Luke McCormack
# Acceptance date: 13 June 2021
#
#
# R code title: "Weigelt et al. 2021 RCode.DataCleaning"
# R code author: "Karl Andraczek"
# R code co-authors: "Nathaly Guerrero Ramirez, Joana Bergmann, Alfons van der Plas, Larry York, Jens Kattge, Helge Bruelheide, Oscar Valverde-Barrantes, Daniel Laughlin"
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
#   I) TRY data
#
# - not filtered by species
# - selected traits: (multiple traits were selected, but only a subset was used for final analyses)
#
# [1] "Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)"
# [2] "Leaf photosynthesis rate per leaf area"
# [3] "Leaf thickness"
# [4] "Leaf nitrogen (N) content per leaf dry mass"
# [5] "Leaf phosphorus (P) content per leaf dry mass"
# [6] "Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): petiole excluded"
# [7] "Leaf dry mass per leaf fresh mass (leaf dry matter content, LDMC)"
# [8] "Leaf density (leaf tissue density, leaf dry mass per leaf volume)"
# [9] "Leaf carbon (C) content per leaf dry mass"
# [10] "Leaf nitrogen/phosphorus (N/P) ratio"
# [11] "Leaf photosynthesis rate per leaf dry mass"
# [12] "Plant biomass and allometry: Leaf dry mass per plant dry mass (leaf weight ratio, LWR)"
# [13] "Plant biomass and allometry: Stem dry mass per plant"
# [14] "Plant biomass and allometry: Leaf dry mass per plant"
# [15] "Plant height vegetative"
# [16] "Plant biomass and allometry: Stem dry mass per plant dry mass per plant"
# [17] "Leaf carbon/nitrogen (C/N) ratio"
# [18] "Stem carbon/nitrogen (C/N) ratio"
# [19] "Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): petiole included"
# [20] "Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): undefined if petiole is in- or excluded"
# [21] "Leaf photosynthesis carboxylation capacity (Vcmax) per leaf dry mass (Farquhar model)"
# [22] "Stem carbon (C) content per stem dry mass"
# [23] "Stem nitrogen (N) content per stem dry mass"
# [24] "Stem phosphorus (P) content per stem dry mass"
# [25] "Stem nitrogen/phosphorus (N/P) ratio"
# [26] "Plant biomass and allometry: Plant dry mass"
# [27] "Leaf photosynthesis electron transport capacity (Jmax) per leaf dry mass (Farquhar model)"
# [28] "Leaf vein density"
# [29] "Wood (sapwood) specific conductivity (stem specific conductivity)"
# [30] "Stem dry mass per stem fresh mass (stem dry matter content, StDMC)"
# [31] "Wood vessel diameter"
# [32] "Wood vessel density"
# [33] "Leaf photosynthesis rate per stomatal conductance"
# [34] "Leaf photosynthetic A/Ci curve: photosynthetic rate per leaf area"
# [35] "Leaf photosynthetic A/Ci curve: stomata conductance per leaf area"
# [36] "Leaf photosynthetic A/Ci curve: intercellular CO2 concentration"
# [37] "Leaf photosynthetic A/Ci curve: transpiration rate per leaf area"
# [38] "Branch vessel density"
# [39] "Branch vessel diameter"
# [40] "Stomata conductance per leaf dry mass"
# [41] "Leaf respiration rate in the dark per leaf dry mass"
#
#
# II) GRoot database from Guerrero-RamCB-rez et al. 2020 (https://groot-database.github.io/GRooT/) (Downloaded in Sep 2020)
#
# III) Additional data collected from single publications (Provided dataset)
#
# IV) Addtional data on rooting depth from Fan et al. 2017 (https://www.pnas.org/content/114/40/10572)
#
# V) nodDB Database on N-Fixation data from Tedersoo et al. 2018 (https://onlinelibrary.wiley.com/doi/abs/10.1111/jvs.12627)
#
# VI) FungalRoot Database on Mycorrhizal association from Soudzilovskaia et al. 2020 (https://nph.onlinelibrary.wiley.com/doi/abs/10.1111/nph.16569)


###############################################################################################################################
# Inspect and filter Data
###############################################################################################################################

#### Update R version (RVersion under which this code was written: R version 4.0.3)

# install.packages("installr")
# library(installr)
# updateR()

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

### II) Load data #################################################################################################

# load data

# Public_restricted = all requested traits from TRY (selected for sRoot) -> downloaded 20.11.2019
# Add_TRY_Data      = additional traits (Stomata conductance per leaf dry mass, Leaf respiration rate in the dark per leaf dry mass) -> downloaded 16.12.2019
# TRY_Trait_53      = additional trait (Leaf photosynthesis rate per leaf area) -> downloaded 13.01.2020

Public_restricted_1 <- fread("Public_restricted_raw.txt", header = T, sep = "\t", dec = ".", quote = "", data.table = T)
Add_TRY_data <- fread("Add_TRY_Data.txt", header = T, sep = "\t", dec = ".", quote = "", data.table = T)
TRY_Trait_53 <- fread("TRY_Trait_53.txt", header = T, sep = "\t", dec = ".", quote = "", data.table = T)
TRY_Trait_53 <- subset(TRY_Trait_53, UnitName == "micro mol m-2 s-1")

# additional to the data from TRY we collected data from other databases and single publications (See Extended Tables: Additional references)

All_data <- read.csv("All_data_new_06_10_2020.csv", header = T, sep = ";")

# We bind both TRY datasets together

Public_restricted_2 <- rbind(Public_restricted_1, Add_TRY_data, TRY_Trait_53)

# dataset containing available metadata (Dataset IDs indicating rows with meta data information: 327,308,210,413,1961)

Public_restricted_metadata <- Public_restricted_2 %>%
    dplyr::filter(DataID == 327 | DataID == 308 | DataID == 210 | DataID == 413 | DataID == 1961)

write.table(Public_restricted_metadata, file = "Public_restricted_metadata.csv", sep = ";")

### III) Remove all NAs from TraitID and artefact column ############################################

# !some trait values come with additional line of information from TRY containing the primary reference (if data were contributed to TRY via an aggregated Database) -> additional lines have NO measured values and NO TraitID but IDENTICAL ObservationID -> by removing all lines with NA as TraitID we also remove additional lines that contain only additional information but no measured values

Public_restricted_2 <- Public_restricted_2[!is.na(Public_restricted_2$TraitID), ]

# Data from TRY contain a column named "v28" = artefact column -> can be removed

Public_restricted_2$V28 <- NULL

# we exclude the "BROT" dataset (that is already included in TRY) because there is a more recent version of the database in All_data = BROT 2.0)

Public_restricted_2 <- Public_restricted_2[!Public_restricted_2$DatasetID == 27, ]

# Column Data_source: database from which data was downloaded

Public_restricted_2$Data_source <- "TRY"

# Bind TRY data and additional data from single publications/ databases together

Public_restricted <- rbind(Public_restricted_2, All_data)
Public_restricted$Unique_ID <- seq.int(nrow(Public_restricted)) # Creates unique ID for each trait value

# creates the column "Only_species_name" that contains ONLY species AND genus name NOT infraspecies

Public_restricted$Only_species_name <- word(Public_restricted$AccSpeciesName, start = 1, end = 2)

# save the file public restricted

write.table(Public_restricted, file = "Public_restricted.csv", sep = ";")

# clear objects from the workspace

rm(list = setdiff(ls(), c("Public_restricted_metadata", "Public_restricted")))


###############################################################################################################################
# Extract and standardize metadata on Growth conditions and Health status from TRY
###############################################################################################################################


# shortcut to load data from previous section

# Public_restricted_metadata <- read.csv('Public_restricted_metadata.csv',header=T,sep=';')


### Filter: non-natural exposition: Exclude the whole Observation

Metadata_exposition_all <- Public_restricted_metadata %>%
    dplyr::filter(DataID == 327 | DataID == 308 | DataID == 210)

# check different states of OriglValueStr work_data %>%

Metadata_exposition_filtered <- Public_restricted_metadata %>%
    dplyr::filter(DataID == 327 | DataID == 308 | DataID == 210) %>%
    dplyr::select(ObservationID, OriglName, OrigValueStr, Comment, Reference) %>%
    arrange(OrigValueStr)

# replaces " " with x (= no additional information is available)

Metadata_exposition_filtered$Comment <- sub("^$", "x", Metadata_exposition_filtered$Comment)

# each reference is assigned to a unique ID (for collapsing on reference level it is easier to have a ID)

References <- data.frame(unique(Metadata_exposition_filtered$Reference))
References$ID <- seq.int(nrow(References))
colnames(References)[1] <- "Reference"
Metadata_exposition_filtered <- merge(Metadata_exposition_filtered, References[, c(1:2)], by = "Reference")

## Collapsing exposition names into the levels: pot, hydroponic and field
## All exposition names were collapsed into pot, hydroponic and field. If exposition names were missing we checked publications manually.
## This process was done seperately for the metadata without comments and the metadata with comments

# subset with no additional information (No comments)

Exposition_NOcomment <- Metadata_exposition_filtered %>%
    dplyr::filter(Comment == "x")
Exposition_NOcomment$Growth_conditions <- ""
Exposition_NOcomment <- Exposition_NOcomment[, c(2, 3, 4, 7, 5, 6, 1)]

# Not clear

Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "branch bag")] <- "Unknown"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Natural/C")] <- "Unknown"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "lab")] <- "Unknown"

# Field

Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural vegetation")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Natural Vegetation")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "field")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Field")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Field Experiment")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural vegetation, but not top canopy")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "university campus")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Trees in field")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Planted vines")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Planted trees")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Planted mature trees/ shrubs")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 135)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 31)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural wetlands (field conditions)")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural vegetation, but not top canopy")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural environment, sun exposed")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "undisturbed soil treatment; fallow wet meadow")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 7)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 8)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 38)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 83)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 84)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 45)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 34)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Natural")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "nat env")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural env.")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural env")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural environment")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural enviroment")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "natural_environment")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Natural environment")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Natural Environment")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Outside \natural\ vegetation")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 131)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 25)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 35)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 82)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 58)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 14)] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Herbarium")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "polyculture in natural field conditions")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "meadows (M) and pastures (P) on south east to south west exposed slopes")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Natural environment (sun leaves)")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "monoculture in natural field conditions")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "cultivated")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Field plants")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "forest fertilization")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "forest stand")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Forest trees")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "disturbed soil treatment; deciduous alluvial forest")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "disturbed soil treatment; fallow wet meadow")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "undisturbed soil treatment; coniferous forest")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "disturbed soil treatment; coniferous forest")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "undisturbed soil treatment; deciduous alluvial forest")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Mosses in forest")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Outside \natural\ vegetation")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 22)] <- "Field"

Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Common Garden")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Common garden trees")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Common garden")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Botanical Garden")] <- "Field"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Botanical garden")] <- "Field"

# Pot

Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse, grrowth container")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse plants")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Controlled climate chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "open-top chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Climate Chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "pots, outside in natural environment")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Pot-grown")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "whole-tree chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "climate chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Climate chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "climate chambers")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 23)] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 43)] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 30)] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 36)] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 37)] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 70)] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$ID == 44)] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "G")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "C")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Glasshouse")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse, Indiana University")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "greenhouse")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse: highlight_lowpH_competition")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse: highlight_lowpH_nocompetition")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse: lowlight_lowpH_nocompetition")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse: highlight_highpH_nocompetition")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse: highlight_highpH_competition")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse: lowleight_lowpH_competition")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse: lowlight_highpH_competition")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Greenhouse: lowlight_highpH_nocompetition")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "Growth chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "growth-chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "groth chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "mesocosm")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "common garden in growth containers with soil corresponding to seed origin")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "heterospecific")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "conspecific")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "controlled environment room")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "petri dish in lab")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "growth chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "open-sided growth chamber")] <- "Pot"
Exposition_NOcomment$Growth_conditions[which(Exposition_NOcomment$OrigValueStr == "mini-ecosystem")] <- "Pot"

# subset with additional information (With comments)

Exposition_comment <- Metadata_exposition_filtered %>%
    dplyr::filter(Comment != "x")

Exposition_comment$Growth_conditions <- ""
Exposition_comment <- Exposition_comment[, c(2, 3, 4, 7, 5, 6, 1)]

# Not clear

Exposition_comment$Growth_conditions[which(Exposition_comment$ID == 2)] <- "Unknown"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "20")] <- "Unknown"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "opt")] <- "Unknown"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "ambient")] <- "Unknown"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "drought")] <- "Unknown"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "shade")] <- "Unknown"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Growth exp")] <- "Unknown"

# Field

Exposition_comment$Growth_conditions[which(Exposition_comment$ID == 1)] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$ID == 45)] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$ID == 87)] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$ID == 74)] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$ID == 33)] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "natural environment")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Outdoor")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Field")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "field")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Outdoor?")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "in situ")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "N")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "position within canopy of measured leaf")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "7")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "6")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "5")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "FW")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "PM")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "FE")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "PU")] <- "Field"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "upper part of juvenile crown")] <- "Field"

# Pot

Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "pot")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Pot exp")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "greenhouse")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Greenhouse")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Green house")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Glass house")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Glasshouse")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "GH")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "CG")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Growth Chamber")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "growth chambers")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "growth_chamber")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Growth chamber")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Growth chamber, -N")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "Growth chamber, +N")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "water stress experiment")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "G")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "C")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$OrigValueStr == "E")] <- "Pot"
Exposition_comment$Growth_conditions[which(Exposition_comment$ID == 77)] <- "Pot"

# rbind both metadata sets together

Metadata_growthConditions <- rbind(Exposition_comment, Exposition_NOcomment)
Metadata_growthConditions$ID <- NULL

write.table(Metadata_growthConditions, file = "Growth_Conditions.csv", sep = ";")

### Filter: Health Status of plants (include only healthy plants)

Metadata_health <- Public_restricted_metadata %>%
    dplyr::filter(DataID == 1961) %>%
    dplyr::select(ObservationID, OriglName, OrigValueStr, Comment, Reference) %>%
    arrange(OrigValueStr)

write.table(Metadata_health, file = "Health_status.csv", sep = ";")

# clear objects from the workspace

rm(list = setdiff(ls(), "Public_restricted"))

###############################################################################################################################
# Increase number of matches (Synonyms) and clean data
###############################################################################################################################

# To increase the number of matches between species sets from TRY and GRooT we used the Leipzig Catalogue of Vascular Plants (LCVP, Freiberg et al. 2020).
# This enabled us to identify more possible synonyms
# from both source lists as the LCVP provides a more
# updated synonym list compared to tools of taxonomic name resolution (Freiberg et al. 2020).
# Scientific names of data from TRY and individual studies were collapsed on species level and standardized using The Plant List (The Plant List 2013).

# set of species present in the GRooT database used to filter requested trait data from TRY

gRoot_Species <- read.csv("Full_species_GRooT_18_11_2019.csv", header = T, sep = ";")

# Shortcut to load data from previous sections

# Public_restricted <- read.csv('Public_restricted.csv',header=T,sep=';')

### I) Synonyms ############################################

## 1) Search for SYNONYMS ##

# By using the "Leipziger Plant List" we searched for synonyms for all sRoot plant list species -> in order to get more species data out of the TRY/Add data
# For this approach the sRoot species list was adjusted = only genus names (e.g. abies) were removed and species containing a third name got the infraspec. rank
# (e.g. Abies lasiocarpa arizonica -> Abies lasiocarpa var. arizonica)
# The file "LCP_results_11_12_2019_Final" contains the results from the synonym search based on the Leipzig Catalogue of Vascular Plants (LCVP, Freiberg et al. 2020)
# You can download the database at "https://idata.idiv.de/ddm/Data/ShowData/1806"


Synonyms_PlantSpec <- read.csv("LCP_results_11_12_2019_Final.csv", header = T, sep = ";")

Synonyms_PlantSpec$LCP_AccSpeciesName <- word(Synonyms_PlantSpec$LCP_Accepted_Taxon, start = 1, end = 2) # only species and genus name (infraspecies ranks are collapsed in one genus)


## 2) Merge Synonyms with TRY data

# Select all unique species from the LCP and the GRoot List

# contains the gRoot species list + the adjusted gRoot species list (adjusted by Alessandro -> removed spec. with only genus name [e.g. abies] and added infraspec. to spec having third name
# [e.g. Abies lasiocarpa arizonica -> Abies lasiocarpa var. arizonica]) + LCP_AccSpeciesName (Synonyms found by using the "Leipziger Species List")

Unique_Spec_List <- Synonyms_PlantSpec %>% dplyr::select(LCP_AccSpeciesName, gRoot_Accepted_Name, Status)

# Only species and genus names in LCP and gRoot columns

Unique_Spec_List$gRoot_Only_species_name <- word(Unique_Spec_List$gRoot_Accepted_Name, start = 1, end = 2)
Unique_Spec_List$gRoot_Accepted_Name <- NULL
colnames(Unique_Spec_List)[3] <- "gRoot_Accepted_Name"
Unique_Spec_List <- Unique_Spec_List[!duplicated(Unique_Spec_List[, c("LCP_AccSpeciesName", "gRoot_Accepted_Name", "Status")]), ]

# Standardize LCP list by removing all duplicated synonyms (standardized species for GRoot)

Unique_Spec_List_dup_a <- Unique_Spec_List %>%
    group_by(gRoot_Accepted_Name) %>%
    dplyr::filter(n() > 1)
Unique_Spec_List_dup_a <- as.data.frame(Unique_Spec_List_dup_a)
Unique_Spec_List_dup <- Unique_Spec_List_dup_a[which(Unique_Spec_List_dup_a$Status == "valid"), ]

Unique_Spec_List_dup_sim <- Unique_Spec_List_dup_a %>% dplyr::filter(Status %in% "synonym")

Unique_Spec_List_nodup <- Unique_Spec_List %>%
    group_by(gRoot_Accepted_Name) %>%
    dplyr::filter(n() == 1)
Unique_Spec_List_nodup <- as.data.frame(Unique_Spec_List_nodup)

Stand_LCP <- rbind(Unique_Spec_List_nodup, Unique_Spec_List_dup) # standardized LCP List

# contains both gRoot and LCP species in two columns (standardized species for TRY)

Unique_pre <- Unique_Spec_List %>% dplyr::select(LCP_AccSpeciesName)
gRoot_pre <- Unique_Spec_List %>% dplyr::select(gRoot_Accepted_Name)
colnames(gRoot_pre)[1] <- "LCP_AccSpeciesName"
Unique_final <- rbind(Unique_pre, gRoot_pre)
Unique_final <- aggregate(Unique_final, by = list(Unique_final$LCP_AccSpeciesName), function(x) x[sample(1:length(x), 1)])
Unique_final$Group.1 <- NULL

# Species selection from TRY/Add data using LCP + gRoot list

colnames(Unique_final)[1] <- "Only_species_name"
Public_restricted_species <- merge(Public_restricted, Unique_final, by = "Only_species_name")

colnames(Public_restricted_species)[1] <- "gRoot_Accepted_Name"
LCP_linked_gRoot_1 <- merge(Public_restricted_species, Stand_LCP, by = "gRoot_Accepted_Name", all = TRUE)

str(LCP_linked_gRoot_1)

LCP_linked_gRoot_1$LCP_AccSpeciesName <- as.factor(LCP_linked_gRoot_1$LCP_AccSpeciesName)
SppSinonimos <- LCP_linked_gRoot_1[is.na(LCP_linked_gRoot_1$LCP_AccSpeciesName), ]
SppSinonimos$LCP_AccSpeciesName <- NULL
colnames(SppSinonimos)[1] <- "LCP_AccSpeciesName"
fixed_version <- merge(SppSinonimos, Unique_Spec_List_dup_sim, by = "LCP_AccSpeciesName")
fixed_version$LCP_AccSpeciesName <- NULL
final_fixed <- merge(fixed_version, Stand_LCP, by = "gRoot_Accepted_Name")
final_fixed$Status.x <- NULL
final_fixed$Status.y <- NULL

Happy_final_list <- rbind(final_fixed, LCP_linked_gRoot_1)

# Final version of TRY/Add data with gRoot species names and LCP names

Public_restricted_species_FINAL <- Happy_final_list[!duplicated(Happy_final_list$Unique_ID), ]

### II) Clean data ############################################

# clear workspace except for the "Public_restricted_species_FINAL" data set

rm(list = setdiff(ls(), "Public_restricted_species_FINAL"))

# TRY: The data may contain duplicates, e.g. if the same data have been contributed to TRY by different contributors.
# If we have identified an entry as duplicate you will find the ID of the original entry in the column OrigObsDataID.
# Thus, by removing any data entries that contain a value within the column OrigObsDataID all duplicates can be removed

Public_restricted_species_FINAL$OrigObsDataID[is.na(Public_restricted_species_FINAL$OrigObsDataID)] <- 0

# all duplicates removed

TRY_PubR_single_occ <- subset(Public_restricted_species_FINAL, OrigObsDataID == 0)

# Value kind is not always the same (single measurement, mean, median, etc.) -> but we need only single mesured values, means, site specific means, Best estimate and species means. Thus, we exclude all others

table(TRY_PubR_single_occ$ValueKindName) # to check the ValueKindNames

TRY_PubR_single_occ$ValueKindName <- sub("^$", "unknown", TRY_PubR_single_occ$ValueKindName) # replaces ValueKind " " with "unknown"

TRY_PubR_cleaned_1 <- subset(TRY_PubR_single_occ, ValueKindName == "Best estimate" | ValueKindName == "Single" | ValueKindName == "single" | ValueKindName == "mean" | ValueKindName == "Mean" | ValueKindName == "Species mean" | ValueKindName == "Site specific mean" | ValueKindName == "unknown")


# For vegetative height the ValueKindeName "Maximum" resonable, thus we include "Maximum" but only for vegetative height

height_max <- subset(TRY_PubR_single_occ, ValueKindName == "Maximum" & TraitID == 3106)
TRY_PubR_cleaned_2 <- rbind(TRY_PubR_cleaned_1, height_max)


# We could now easily remove all NA's occurring in the column StdValue
# !BUT! We will loose data entries as they are not yet standardized <- UnitName and StdValue = NA)!
# First of all we create a subset that contains only the not yet standardized values

TRY_PubR_single_occ_NotStandar <- TRY_PubR_cleaned_2[is.na(TRY_PubR_cleaned_2$StdValue), ] # NOT standardized values
TRY_PubR_single_occ_Standar <- TRY_PubR_cleaned_2[!is.na(TRY_PubR_cleaned_2$StdValue), ] # Standardized values

# Parts of Daniel Laughlins data on LDMC was not standardized (Kramer-Walter et al.)

TRY_PubR_single_occ_Standar_daniel <- TRY_PubR_single_occ_Standar %>%
    dplyr::filter(LastName == "Kramer-Walter") %>%
    dplyr::filter(DataName == "LDMC")
TRY_PubR_single_occ_Standar_daniel$OrigUnitStr <- "mg g-1"
TRY_PubR_single_occ_Standar_daniel$StdValue <- TRY_PubR_single_occ_Standar_daniel$StdValue / 1000
TRY_PubR_single_occ_Standar_nodaniel <- TRY_PubR_single_occ_Standar %>% dplyr::filter(!(LastName == "Kramer-Walter" & DataName == "LDMC" & OrigUnitStr == "g g-1"))

# Already standardized data set

TRY_PubR_single_occ_Standar_total <- rbind(TRY_PubR_single_occ_Standar_daniel, TRY_PubR_single_occ_Standar_nodaniel)

# remove NA's in originalValueStr in NOT standardized data

TRY_PubR_single_occ_NotStandar$OrigValueStr <- as.numeric(TRY_PubR_single_occ_NotStandar$OrigValueStr)
TRY_PubR_single_occ_NotStandar <- TRY_PubR_single_occ_NotStandar[!is.na(TRY_PubR_single_occ_NotStandar$OrigValueStr), ]


# We need to identify the different Units for the Traits that are NOT standardized
# (not standardized TraitValues are within the column "OrigValueStr" and not standardized TraitUnits within the column "OrigUnitStr")

with(TRY_PubR_single_occ_NotStandar, tapply(OrigUnitStr, TraitID, FUN = function(x) unique(x)))

## We should now standardize the OrigValueStr that are not yet standardized by TRY:

TRY_PubR_single_occ_NotStandar$OrigUnitStr <- as.character(TRY_PubR_single_occ_NotStandar$OrigUnitStr)
TRY_PubR_single_occ_NotStandar$UnitName <- as.character(TRY_PubR_single_occ_NotStandar$UnitName)

# FOR kg/cm3 TraitID 4
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "kg/m3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "kg/m3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] / 1000

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "kg/m3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] <- "g/cm3"

# FOR cm TraitID 3106
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm" & TRY_PubR_single_occ_NotStandar$TraitID == 3106)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm" & TRY_PubR_single_occ_NotStandar$TraitID == 3106)] / 100

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm" & TRY_PubR_single_occ_NotStandar$TraitID == 3106)] <- "m"

# FOR cm TraitID 46
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm" & TRY_PubR_single_occ_NotStandar$TraitID == 46)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm" & TRY_PubR_single_occ_NotStandar$TraitID == 46)] * 10

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm" & TRY_PubR_single_occ_NotStandar$TraitID == 46)] <- "mm"

# FOR m TraitID 3106
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "m" & TRY_PubR_single_occ_NotStandar$TraitID == 3106)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "m" & TRY_PubR_single_occ_NotStandar$TraitID == 3106)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "m" & TRY_PubR_single_occ_NotStandar$TraitID == 3106)] <- "m"

# FOR g / cm3 TraitID 4
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g / cm3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g / cm3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g / cm3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] <- "g/cm3"

# FOR g/cm3 TraitID 4
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/cm3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/cm3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/cm3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] <- "g/cm3"

# FOR g*cm-3 TraitID 4
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g*cm-3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g*cm-3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g*cm-3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] <- "g/cm3"

# FOR g cm-3 TraitID 4
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g cm-3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g cm-3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g cm-3" & TRY_PubR_single_occ_NotStandar$TraitID == 4)] <- "g/cm3"

# FOR mg/mm3 TraitID 48
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/mm3" & TRY_PubR_single_occ_NotStandar$TraitID == 48)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/mm3" & TRY_PubR_single_occ_NotStandar$TraitID == 48)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/mm3" & TRY_PubR_single_occ_NotStandar$TraitID == 48)] <- "g/cm3"

# FOR mg/cm3 TraitID 48
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/cm3" & TRY_PubR_single_occ_NotStandar$TraitID == 48)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/cm3" & TRY_PubR_single_occ_NotStandar$TraitID == 48)] / 1000

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/cm3" & TRY_PubR_single_occ_NotStandar$TraitID == 48)] <- "g/cm3"

# FOR mg/g TraitID 14
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/g" & TRY_PubR_single_occ_NotStandar$TraitID == 14)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/g" & TRY_PubR_single_occ_NotStandar$TraitID == 14)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/g" & TRY_PubR_single_occ_NotStandar$TraitID == 14)] <- "mg/g"

# FOR mg/g TraitID 13
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/g" & TRY_PubR_single_occ_NotStandar$TraitID == 13)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/g" & TRY_PubR_single_occ_NotStandar$TraitID == 13)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg/g" & TRY_PubR_single_occ_NotStandar$TraitID == 13)] <- "mg/g"

# FOR mm2/mg TraitID 3116
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mm2/mg" & TRY_PubR_single_occ_NotStandar$TraitID == 3116)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mm2/mg" & TRY_PubR_single_occ_NotStandar$TraitID == 3116)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mm2/mg" & TRY_PubR_single_occ_NotStandar$TraitID == 3116)] <- "mm2 mg-1"

# FOR g/plant TraitID 129
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/plant" & TRY_PubR_single_occ_NotStandar$TraitID == 129)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/plant" & TRY_PubR_single_occ_NotStandar$TraitID == 129)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/plant" & TRY_PubR_single_occ_NotStandar$TraitID == 129)] <- "g"

# FOR g TraitID 129
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g" & TRY_PubR_single_occ_NotStandar$TraitID == 129)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g" & TRY_PubR_single_occ_NotStandar$TraitID == 129)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g" & TRY_PubR_single_occ_NotStandar$TraitID == 129)] <- "g"

# FOR mg TraitID 129
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg" & TRY_PubR_single_occ_NotStandar$TraitID == 129)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg" & TRY_PubR_single_occ_NotStandar$TraitID == 129)] / 1000

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "mg" & TRY_PubR_single_occ_NotStandar$TraitID == 129)] <- "g"

# gstem/gplant TraitID 136
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "gstem/gplant" & TRY_PubR_single_occ_NotStandar$TraitID == 136)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "gstem/gplant" & TRY_PubR_single_occ_NotStandar$TraitID == 136)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "gstem/gplant" & TRY_PubR_single_occ_NotStandar$TraitID == 136)] <- "g/g"

# FOR g/g TraitID 136
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/g" & TRY_PubR_single_occ_NotStandar$TraitID == 136)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/g" & TRY_PubR_single_occ_NotStandar$TraitID == 136)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/g" & TRY_PubR_single_occ_NotStandar$TraitID == 136)] <- "g/g"

# FOR g/g TraitID 686
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/g" & TRY_PubR_single_occ_NotStandar$TraitID == 686)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/g" & TRY_PubR_single_occ_NotStandar$TraitID == 686)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/g" & TRY_PubR_single_occ_NotStandar$TraitID == 686)] <- "g/g"

# FOR g/g TraitID 110
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/g" & TRY_PubR_single_occ_NotStandar$TraitID == 110)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/g" & TRY_PubR_single_occ_NotStandar$TraitID == 110)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g/g" & TRY_PubR_single_occ_NotStandar$TraitID == 110)] <- "g/g"

# FOR g (stem)/g (all) TraitID 136
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g (stem)/g (all)" & TRY_PubR_single_occ_NotStandar$TraitID == 136)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g (stem)/g (all)" & TRY_PubR_single_occ_NotStandar$TraitID == 136)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g (stem)/g (all)" & TRY_PubR_single_occ_NotStandar$TraitID == 136)] <- "g/g"

# FOR g g-1 TraitID 136
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g g-1" & TRY_PubR_single_occ_NotStandar$TraitID == 136)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g g-1" & TRY_PubR_single_occ_NotStandar$TraitID == 136)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "g g-1" & TRY_PubR_single_occ_NotStandar$TraitID == 136)] <- "g/g"

# FOR kg/m/s/Mpa TraitID 1096
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "kg/m/s/Mpa" & TRY_PubR_single_occ_NotStandar$TraitID == 1096)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "kg/m/s/Mpa" & TRY_PubR_single_occ_NotStandar$TraitID == 1096)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "kg/m/s/Mpa" & TRY_PubR_single_occ_NotStandar$TraitID == 1096)] <- "kg/m/s/Mpa"

# FOR cm2/g (n.r.) TraitID 3116
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm2/g (n.r.)" & TRY_PubR_single_occ_NotStandar$TraitID == 3116)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm2/g (n.r.)" & TRY_PubR_single_occ_NotStandar$TraitID == 3116)] / 10

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm2/g (n.r.)" & TRY_PubR_single_occ_NotStandar$TraitID == 3116)] <- "mm2 mg-1"

# FOR cm2/g TraitID 3117
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm2/g" & TRY_PubR_single_occ_NotStandar$TraitID == 3117)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm2/g" & TRY_PubR_single_occ_NotStandar$TraitID == 3117)] / 10

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "cm2/g" & TRY_PubR_single_occ_NotStandar$TraitID == 3117)] <- "mm2 mg-1"

# FOR micromolCO2 m-2 s-1 / mmolH2O m-2 s-1 TraitID 3128
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "micromolCO2 m-2 s-1 / mmolH2O m-2 s-1" & TRY_PubR_single_occ_NotStandar$TraitID == 3128)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "micromolCO2 m-2 s-1 / mmolH2O m-2 s-1" & TRY_PubR_single_occ_NotStandar$TraitID == 3128)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "micromolCO2 m-2 s-1 / mmolH2O m-2 s-1" & TRY_PubR_single_occ_NotStandar$TraitID == 3128)] <- "micromolCO2 m-2 s-1 / mmolH2O m-2 s-1"

# FOR n/mm2 TraitID 3390
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "n/mm2" & TRY_PubR_single_occ_NotStandar$TraitID == 3390)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "n/mm2" & TRY_PubR_single_occ_NotStandar$TraitID == 3390)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "n/mm2" & TRY_PubR_single_occ_NotStandar$TraitID == 3390)] <- "n/mm2"

# FOR microm TraitID 3391
TRY_PubR_single_occ_NotStandar$StdValue[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "microm" & TRY_PubR_single_occ_NotStandar$TraitID == 3391)] <- TRY_PubR_single_occ_NotStandar$OrigValueStr[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "microm" & TRY_PubR_single_occ_NotStandar$TraitID == 3391)]

TRY_PubR_single_occ_NotStandar$UnitName[which(TRY_PubR_single_occ_NotStandar$OrigUnitStr == "microm" & TRY_PubR_single_occ_NotStandar$TraitID == 3391)] <- "microm"

# correct false Unit names (such as: g/cm3 for StDMC)

TRY_PubR_single_occ_Standar_total$UnitName[which(TRY_PubR_single_occ_Standar_total$TraitID == 1181 & TRY_PubR_single_occ_Standar_total$UnitName == "g/cm3")] <- "g g-1"

TRY_PubR_single_occ_Standar_total$UnitName[which(TRY_PubR_single_occ_Standar_total$TraitID == 56 & TRY_PubR_single_occ_Standar_total$UnitName == "g/cm3")] <- "g g-1"

TRY_PubR_single_occ_Standar_total$UnitName[which(TRY_PubR_single_occ_Standar_total$TraitID == 165 & TRY_PubR_single_occ_Standar_total$UnitName == "g/cm3")] <- "g g-1"

TRY_PubR_single_occ_Standar_total$UnitName[which(TRY_PubR_single_occ_Standar_total$TraitID == 136 & TRY_PubR_single_occ_Standar_total$UnitName == "g/g")] <- "g g-1"

TRY_PubR_single_occ_Standar_total$UnitName[which(TRY_PubR_single_occ_Standar_total$TraitID == 3106 & TRY_PubR_single_occ_Standar_total$UnitName == "m ")] <- "m"

TRY_PubR_single_occ_Standar_total$UnitName[which(TRY_PubR_single_occ_Standar_total$TraitID == 46 & TRY_PubR_single_occ_Standar_total$UnitName == "mm ")] <- "mm"

TRY_PubR_cleaned <- TRY_PubR_single_occ_Standar_total

# safety check (should be = 0)

sum(is.na(TRY_PubR_cleaned$StdValue))

# are there still multiple units/ TraitID?

with(TRY_PubR_cleaned, tapply(UnitName, TraitID, FUN = function(x) unique(x))) # NO multiple units

# calculate LMA (leaf mass per area) based on SLA (specific leaf area) and add it to TRY_PubR_cleaned

TRY_LMA <- subset(TRY_PubR_cleaned, TRY_PubR_cleaned$TraitID == 3117 | TRY_PubR_cleaned$TraitID == 3115 | TRY_PubR_cleaned$TraitID == 3116)

TRY_LMA$StdValue <- (1 / TRY_LMA$StdValue)
TRY_LMA$UnitName <- "mg/mm2"
TRY_LMA$TraitName <- "Leaf mass per area (LMA)"
TRY_LMA$TraitID <- NA

TRY_PubR_cleaned <- rbind(TRY_PubR_cleaned, TRY_LMA)

# Add metadata on growth conditions

Growth_Conditions <- read.csv("Growth_Conditions.csv", header = T, sep = ";")
colnames(Growth_Conditions)[3] <- "Growth_Conditions_OrigInfo"
Growth_Conditions <- Growth_Conditions[!duplicated(Growth_Conditions$ObservationID), ]

Health_status <- read.csv("Health_status.csv", header = T, sep = ";")
colnames(Health_status)[3] <- "Health_status"
Health_status$Health_status[which(Health_status$Health_status == "healthy")] <- "Healthy"
Health_status <- Health_status[!duplicated(Health_status$ObservationID), ]

TRY_PubR_cleaned_meta <- merge(TRY_PubR_cleaned, Health_status[, c(1, 3)], by = "ObservationID", all.x = T)
TRY_PubR_cleaned_meta <- merge(TRY_PubR_cleaned_meta, Growth_Conditions[, c(1, 3, 4)], by = "ObservationID", all.x = T)

# adjust remainign wrong assigned grwoth condition

TRY_PubR_cleaned_meta$Growth_conditions[which(TRY_PubR_cleaned_meta$Growth_Conditions_OrigInfo == "Climate chamber")] <- "Pot"

# save data

write.table(TRY_PubR_cleaned_meta, file = "TRY_PubR_cleaned_meta.csv", sep = ";")

# clear workspace

rm(list = setdiff(ls(), "TRY_PubR_cleaned_meta"))


###############################################################################################################################
# Add Individual data (species specific individual trait data where root and
# shoot traits were measured on the same plant individual or plot.) to GRoot and TRY
###############################################################################################################################


# Shortcut to load data from previous sections

# TRY_PubR_cleaned_meta <- read.csv('TRY_PubR_cleaned_meta.csv',header=T,sep=';',na.strings=c("","NA"))

# load data

Indi_PES <- read.csv("Indi_PES.csv", header = T, sep = ";")
GRooTFullVersion <- read.csv("GRooTFullVersionNew.csv", header = T, sep = ",", na.strings = c("", "NA"))
TNRS_Indi_PES <- fread("TNRS_results_Indi_PES_new.txt", header = T, sep = "\t", dec = ".", quote = "", data.table = T)


# Overlapping data
# Exclude duplicated datasets that are both in the GRoot/TRY data and in the Individual PES data

TRY_delete_list <- c(
    "De Long J. R., Jackson B. G., Wilkinson A., Pritchard W. J., Oakley S., Mason K. E., Stephan J. G.,Ostle N. J.m Johnson D., Baggs E. M., Bardgett R. D 2019, Relationships between monocultures and mixed communities in temperate grassland, Journal of Ecology,
                     doi: 10.1111/1365-2745.13237",
    "De Vries F., Bardgett R.D. (2016) Plant community controls on short-term ecosystem nitrogen retention. New Phytologist.
                     doi: 10.1111/nph.13832",
    "Fort, F., Jouany, C., & Cruz, P. (2012). Root and leaf functional trait relations in Poaceae species: implications of differing resource-acquisition strategies. Journal of Plant Ecology, 6(3), 211-219.",
    "Freschet, G. T., J. H. C. Cornelissen, R. S. P. van Logtestijn, and R. Aerts. 2010. Evidence of the 'plant economics spectrum' in a subarctic flora. Journal of Ecology 98:362-373.",
    "Freschet GT, Kichenin E, Wardle DA. 2015. Explaining within-community variation in plant biomass allocation: a balance between organ biomass and morphology above vs below ground? Journal of Vegetation Science 26: 431-440.",
    "Geng Y., Ma W., Wang L., Baumann F., KCB<hn P., Scholten T., He J-S. 2017, Linking above- and belowground traits to soil and climate variables: an integrated database on China's grassland species, Ecology 98, doi: 10.1002/ecy.1780/suppinfo",
    "Hu Y.-K., Pan X., Yang X.-J., Liu G.-F., Liu X.-Y., Song Y.-B., Zhang M.-Y., Cui L.-J., Dong M. 2019, Is there coordination of leaf and fine root traits at local scales? A test in temperate forest swamps, Ecology and Evolution, doi: 10.1002/ece3.5421",
    "Jo I., Fridley J. D., Frank D. A. 2015, More of the same? In situ leaf and root decomposition rates do not vary between 80 native and nonnative decidious forest species, New Phytologist, doi: 10.1111/nph.13619",
    "Laughlin, D. C., J. J. Leppert, M. M. Moore, and C. H. Sieg. 2010. A multi-trait test of the leaf-height-seed plant strategy scheme with 133 species from a pine forest flora. Functional Ecology 24:493-501.",
    "Laughlin D. C., Lusk C. H., Bellingham P. J., Burslem D. F. R. P., Simpson A. H., Kramer-Walter K. R., Intraspecific trait variation can weaken interspecific trait correlations when assessing the whole-plant economic spectrum, Ecology and Evolution, doi: https://doi.org/10.1002/ece3.3447",
    "Smith, S. W., Woodin, S. J., Pakeman, R. J., Johnson, D. and van der Wal, R. (2014), Root traits predict decomposition across a landscape-scale grazing experiment. New Phytologist. Doi: 10.1111/nph.12845",
    "Tjoelker M. G., Craine J. M., Wedin D., Reich P. B., Tilman D. 2005, Linking leaf and root trait syndromes among 39 grassland and savannah species, New Phytologist, doi: 10.1111/j.1469-8137.2005.01428.x",
    "Schroeder-Georgi, T., Wirth, C., Nadrowski, K., Meyer, S. T., Mommer, L. and Weigelt, A. (2016), From pots to plots: hierarchical trait-based prediction of plant performance in a mesic grassland. J Ecol, 104: 206-218. doi:10.1111/1365-2745.12489"
)


gRoot_delete_list <- c(
    "Abiven S, Recous S, Reyes V, Oliver R. 2005. Mineralisation of C and N from root, stem and leaf residues in soil and role of their biochemical quality. Biology and Fertility of Soils 42: 119-128.",
    "Chanteloup et Bonis. 2013. Basic and Applied Ecology",
    "Comas LH, Eissenstat DM. 2009. Patterns in root trait variation among 25 co-existing North American forest species. New Phytologist 182: 919-928.",
    "De Vries F., Bardgett R.D. (2016) Plant community controls on short-term ecosystem nitrogen retention. New Phytologist. doi: 10.1111/nph.13832",
    "Fort F, Jouany C, Cruz P. 2013.Root and leaf functional trait relations in Poaceae species: implications of differing resource-acquisition strategies. Journal of Plant Ecology 6:211-219",
    "Freschet GT, Cornelissen JHC, van Logtestijn RSP, Aerts R. 2010. Evidence of the 'plant economics spectrum' in a subarctic flora. Journal of Ecology 98: 362-373.",
    "Freschet GT, Aerts R, Cornelissen JHC. 2012. A plant economics spectrum of litter decomposability. Functional Ecology 26: 56-65.",
    "Freschet GT, Bellingham PJ, Lyver PO'B, Bonner KI, Wardle DA. 2013. Plasticity in above- and belowground resource acquisition traits in response to single and multiple environmental factors in three tree species. Ecology and Evolution 3: 1065-1078.",
    "Freschet GT, Kichenin E, Wardle DA. 2015. Explaining within-community variation in plant biomass allocation: a balance between organ biomass and morphology above vs below ground? Journal of Vegetation Science 26: 431-440.",
    "Guyonnet JP, Cantarel AAM, Simon L, Haichar FZ. 2018. Root exudation rate as functional trait involved in plant nutrient-use strategy classification. Ecology and Evolution, in press.",
    "Jo I, Fridley JD, Frank DA. 2016. More of the same? In situ leaf and root decomposition rates do not vary between 80 native and nonnative deciduous forest species. New Phytologist 209: 115-122.",
    "Kembel SW, Cahill JF, Jr. 2011. Independent Evolution of Leaf and Root Traits within and among Temperate Grassland Plant Communities. PLoS ONE 6(6): e19992.",
    "Kong D, Ma C, Zhang Q, Li L, Chen X, Zeng H, Guo D. 2014. Leading dimensions in absorptive root trait variation across 96 subtropical forest species. New Phytologist 203: 863-872.",
    "Laughlin DC, Leppert JJ, Moore MM, Sieg CH. 2010. A multi-trait test of the leaf-height-seed plant strategy scheme with 133 species froma pine florest flora. Functional Ecology 24: 493-501",
    "Li FL, Bao WK. 2015. New insights into leaf and fine-root trait relationships: implications of resource acquisition among 23 xerophytic woody species. Ecology and Evolution 5: 5344-5351.",
    "Liese R, Alings K, Meier IC. 2017. Root branching is a leading root trait of the plant economics spectrum in temperate trees. Frontiers in Plant Science 8:315.",
    "Liu G, Freschet GT, Pan X, Cornelissen JHC, Li Y, Dong M. 2010. Coordinated variation in leaf and root traits across multiple spatial scales in Chinese semi-arid and arid ecosystems. New Phytologist 188: 543-553.",
    "Mokany K, Ash J. 2008. Are traits measured on pot grown plants representative of those in natural communities? Journal of Vegetation Science 19: 119-126.",
    "Perez-Ramos IM, Volaire F, Fattet M, Blanchard A, Roumet C. Tradeoffs between functional strategies for resource-use and drought-survival in Mediterranean rangeland species. Environmental and Experimental Botany 87: 126-136.",
    "Reich et al. 2003 . New Phytol. 157: 617-631",
    "Roumet C, Lafont F, Sari M, Warembourg F, Garnier E. 2008. Root traits and taxonomic affiliation of nine herbaceous species grown in glasshouse conditions. Plant Soil 312:69-83. ",
    "Roumet C, Birouste M, Picon-Cochard C, Ghestem M, Osman N, Vrignon-Brenas S, Cao K-F, Stokes A. 2016. Root structure-function relationships in 74 species: evidence of root economics spectrum related to carbon economy. New Phytologist 210: 815-826.",
    "Smith SW, Woodin SJ, Pakeman RJ, Johnson D, van der Wal R. 2014. Root traits predict decomposition across a landscape-scale grazing experiment. New Phytologist 203: 851-862.",
    "Tjoelker MG, Craine JM, Wedin D, Reich PB, Tilman D. 2005. Linking leaf and root trait syndromes among 39 grassland and savannah species. New Phytologist 167: 493-508.",
    "Valverde-Barrantes OJ, Smemo KA,  Feinstein LM, Kershner MW, Blackwood CB. 2015. Aggregated and complementary: symmetric proliferation, overyielding, and mass effects explain fine-root biomass in soil patches in a diverse temperate deciduous forest landscape. New Phytologist 205(2): 731-742.",
    "Wardle et al. 1998 - Journal of Ecology, 86, 405-436",
    "Withington JM, Reich PB, Oleksyn J, Eissenstat DM. 2006. Comparisons of structure and life span in roots and leaves among temperate trees. Ecological Monographs 76: 381-397.", "Schroeder-Georgi T, Wirth C, Nadrowski K, Meyer S T, Mommer L, Weigelt A. 2016. From pots to plots: hierarchical trait-based prediction of plant performance in a mesic grassland. Journal of Ecology 104(1): 206-218"
)


# Reduce Indi_PES only to desired traits

Indi_PES <- Indi_PES[, c(1, 2, 17, 18, 27, 30, 35, 36, 38:40, 42, 43, 47, 48, 52, 56, 61, 62, 65, 71, 72, 78)]

# Standardize traits / rename columns

colnames(Indi_PES)[7] <- "Leaf mass per area (LMA)"
colnames(Indi_PES)[8] <- "Leaf density (leaf tissue density, leaf dry mass per leaf volume)"
colnames(Indi_PES)[9] <- "Leaf nitrogen (N) content per leaf dry mass"
colnames(Indi_PES)[10] <- "Specific_root_length"
colnames(Indi_PES)[11] <- "Root_tissue_density"
colnames(Indi_PES)[12] <- "Mean_Root_diameter"
colnames(Indi_PES)[13] <- "Root_N_concentration"
colnames(Indi_PES)[14] <- "Leaf phosphorus (P) content per leaf dry mass"
colnames(Indi_PES)[17] <- "Root_mycorrhizal colonization"
colnames(Indi_PES)[18] <- "Root_P_concentration"
colnames(Indi_PES)[19] <- "Root_lignin_concentration"
colnames(Indi_PES)[23] <- "Plant height vegetative"

# adds unique ID

Indi_PES$Individual_data <- seq.int(nrow(Indi_PES))
Indi_PES$Row.ID <- NULL

# reshape data

Indi_final_reshape <- reshape2::melt(Indi_PES,
    id.vars = c(
        "Individual_data", "full_species", "growth_conditions",
        "Woodiness", "Reference", "Root.Entity"
    )
)

Indi_final_reshape <- Indi_final_reshape %>% drop_na()
colnames(Indi_final_reshape)[7] <- "TraitName"

# Synonym check of Individual PES species

TNRS_Indi_PES$Accepted_name[TNRS_Indi_PES$Accepted_name == ""] <- "Unknown"
TNRS_Indi_PES$Accepted_name[TNRS_Indi_PES$Accepted_name == "Unknown"] <- TNRS_Indi_PES$Name_submitted[TNRS_Indi_PES$Accepted_name == "Unknown"]
TNRS_Indi_PES <- TNRS_Indi_PES[!is.na(TNRS_Indi_PES$Accepted_name)]

TNRS_results <- TNRS_Indi_PES[, c(1, 6)]
TNRS_results$Accepted_name_Only_species_name <- word(TNRS_results$Accepted_name, start = 1, end = 2)
TNRS_results <- TNRS_results[!duplicated(TNRS_results$Name_submitted)]


colnames(TNRS_results)[1] <- "full_species"
Indi_final_reshape_SYN <- merge(Indi_final_reshape, TNRS_results, by = "full_species")

Indi_final_reshape_SYN$Accepted_name <- NULL
colnames(Indi_final_reshape_SYN)[9] <- "gRoot_Accepted_Name"
Indi_final_reshape_SYN$full_species <- NULL

# subset Individual PES data in traits supposed to be merged with TRY or GRooT

Subset_Indi_gRoot <- Indi_final_reshape_SYN %>%
    dplyr::filter(TraitName %in%
        c(
            "Specific_root_length",
            "Root_tissue_density",
            "Mean_Root_diameter",
            "Root_N_concentration",
            "Root_mycorrhizal colonization",
            "Root_P_concentration",
            "Root_lignin_concentration",
            "Fine.root.longevity..yr.",
            "Cortex...m.",
            "Stele...m."
        ))

Subset_Indi_TRY <- Indi_final_reshape_SYN %>%
    dplyr::filter(TraitName %in%
        c(
            "Leaf mass per area (LMA)",
            "Leaf density (leaf tissue density, leaf dry mass per leaf volume)",
            "Leaf nitrogen (N) content per leaf dry mass",
            "Leaf phosphorus (P) content per leaf dry mass",
            "Leaf_Lignin",
            "Leaf.Longevity..Yr.",
            "Plant height vegetative"
        ))

# check colnames for consistency to gRoot and rename if needed

colnames(Subset_Indi_gRoot)[2] <- "measurementProvenance"
colnames(Subset_Indi_gRoot)[3] <- "woodiness"
colnames(Subset_Indi_gRoot)[4] <- "references"
colnames(Subset_Indi_gRoot)[5] <- "belowgroundEntities"
colnames(Subset_Indi_gRoot)[6] <- "traitName"
colnames(Subset_Indi_gRoot)[7] <- "traitValue"

Subset_Indi_gRoot$genusTNRS <- word(Subset_Indi_gRoot$gRoot_Accepted_Name, start = 1, end = 1)
Subset_Indi_gRoot$speciesTNRS <- word(Subset_Indi_gRoot$gRoot_Accepted_Name, start = 2, end = 2)

Subset_Indi_gRoot$gRoot_Accepted_Name <- NULL

# check colnames for consistency to TRY and rename if needed

colnames(Subset_Indi_TRY)[2] <- "Growth_conditions"
colnames(Subset_Indi_TRY)[7] <- "StdValue"
Subset_Indi_TRY$Woodiness <- NULL
Subset_Indi_TRY$Root.Entity <- NULL

# add Individual PES data to GRoot

Indi_and_GRoot <- GRooTFullVersion

Indi_and_GRoot <- Indi_and_GRoot %>%
    dplyr::filter(!references %in% gRoot_delete_list)

Indi_and_GRoot$Individual_data <- "NA"

Indi_and_GRoot <- rbind.fill(Indi_and_GRoot, Subset_Indi_gRoot)

Indi_and_GRoot$errorRisk <- NULL
Indi_and_GRoot$errorRiskEntries <- NULL

write.csv(Indi_and_GRoot, "Indi_and_GRoot.csv", row.names = FALSE)

# add Individual PES data to TRY

Indi_and_TRY <- TRY_PubR_cleaned_meta

Indi_and_TRY <- Indi_and_TRY %>%
    dplyr::filter(!Reference %in% TRY_delete_list)

Indi_and_TRY$Individual_data <- "NA"

Indi_and_TRY <- rbind.fill(Indi_and_TRY, Subset_Indi_TRY)

Indi_and_TRY$ErrorRisk <- NULL

write.table(Indi_and_TRY, file = "Indi_and_TRY.csv", sep = ";")

# clear workspace

rm(list = setdiff(ls(), c("Indi_and_GRoot", "Indi_and_TRY")))


###############################################################################################################################
# Addtional data on rooting depth from Fan et al. 2017 (https://www.pnas.org/content/114/40/10572)
###############################################################################################################################


# Shortcut to load data from previous sections

# Indi_and_GRoot <- read.csv("Indi_and_GRoot.csv",header=T)

# load data

Rooting_Depth_Fan <- read.csv("Rooting_Depth_Dataset_Fan_adjusted.csv", header = T, sep = ";")
RDep_Fan_TNRS <- fread("TNRS_results_RDep_Fan.txt", header = T, sep = "\t", dec = ".", quote = "", data.table = T, na.strings = c("", "NA"))
# This dataset was manually adjusted, meaning an additional column was added in which values of rooting depth were expressed as numerical values
# Thus, rooting depth values given as ranges ect. were changed to mean values (e.g. 1-2 was changed to 1.5)

Rooting_Depth_Fan$Species <- word(Rooting_Depth_Fan$Species.Name, start = 1, end = 2)
Rooting_Depth_Fan$Species <- gsub(",", "", Rooting_Depth_Fan$Species)

# Load TNRS results and merge results with Rooting Depth Dataset

RDep_Fan_TNRS$Name_submitted <- gsub(",", "", RDep_Fan_TNRS$Name_submitted)
colnames(RDep_Fan_TNRS)[1] <- "Species"
RDep_Fan_TNRS <- RDep_Fan_TNRS[, c(1, 6)]

RDep_Fan_TNRS <- RDep_Fan_TNRS %>%
    drop_na()

Rooting_Depth_Fan <- merge(Rooting_Depth_Fan, RDep_Fan_TNRS, by = "Species")

# Create separate columns containing Genus and Species Names

Rooting_Depth_Fan$genusTNRS <- word(Rooting_Depth_Fan$Accepted_name, start = 1, end = 1)
Rooting_Depth_Fan$speciesTNRS <- word(Rooting_Depth_Fan$Accepted_name, start = 2, end = 2)

# Rename Columns according to column names in GRooT

Rooting_Depth_Fan$traitName <- "Max_Rooting_Depth"
colnames(Rooting_Depth_Fan)[11] <- "traitValue"
colnames(Rooting_Depth_Fan)[2] <- "references"

# reduce dataset to important information

Rooting_Depth_Fan <- Rooting_Depth_Fan[, c(2, 32, 33, 34, 11)]

# Connect GRooT dataset with Rooting Depth Data

GRoot_and_RDep_Fan <- rbind.fill(Indi_and_GRoot, Rooting_Depth_Fan)
Indi_and_GRoot <- GRoot_and_RDep_Fan

# Save data

write.csv(Indi_and_GRoot, "Indi_and_GRoot.csv", row.names = FALSE)

# clear workspace

rm(list = setdiff(ls(), c("Indi_and_GRoot", "Indi_and_TRY")))


###############################################################################################################################
# Additional calculations
###############################################################################################################################

# Shortcut to load data from previous sections

# Indi_and_TRY <- read.csv('Indi_and_TRY.csv',header=T,sep=';')
# Indi_and_GRoot <- read.csv('Indi_and_GRoot.csv',header=T)

# add unique ID that also includes the individual data

Indi_and_TRY$ID_TRY_INDI <- seq.int(nrow(Indi_and_TRY))
Indi_and_GRoot$ID_GRoot_INDI <- seq.int(nrow(Indi_and_GRoot))


### I) Calculate Error risk values FOR TRY ############################################

# remove values = 0 or inf

length(Indi_and_TRY$StdValue[which(Indi_and_TRY$StdValue == 0)]) # 234
Indi_and_TRY <- Indi_and_TRY[!Indi_and_TRY$StdValue == 0, ] # remove all 0's

Indi_and_TRY <- Indi_and_TRY[!is.infinite(Indi_and_TRY$StdValue), ]

### data entries with information at species level ###
### since no entry at only genus level is present we don't need a separate dataset for traits measured at genus level ###

speciesTRY <- Indi_and_TRY[which(!grepl("^\\w+$", Indi_and_TRY$gRoot_Accepted_Name)), ]

### Error risk calculation ###

# not normally distirbuted data

speciesTRYlog <- speciesTRY %>%
    dplyr::select(ID_TRY_INDI, TraitName, StdValue, gRoot_Accepted_Name) %>%
    group_by(gRoot_Accepted_Name, TraitName) %>%
    dplyr::filter(TraitName %in% c(
        "Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)",
        "Leaf nitrogen (N) content per leaf dry mass", "Leaf phosphorus (P) content per leaf dry mass",
        "Leaf thickness", "Leaf dry mass per leaf fresh mass (leaf dry matter content, LDMC)",
        "Leaf density (leaf tissue density, leaf dry mass per leaf volume)", "Leaf nitrogen/phosphorus (N/P) ratio",
        "Plant biomass and allometry: Stem dry mass per plant",
        "Plant biomass and allometry: Leaf dry mass per plant",
        "Plant biomass and allometry: Stem dry mass per plant dry mass per plant",
        "Leaf carbon/nitrogen (C/N) ratio", "Stem carbon/nitrogen (C/N) ratio",
        "Leaf photosynthesis electron transport capacity (Jmax) per leaf dry mass (Farquhar model)",
        "Stem nitrogen (N) content per stem dry mass",
        "Stem phosphorus (P) content per stem dry mass", "Stem nitrogen/phosphorus (N/P) ratio",
        "Plant biomass and allometry: Plant dry mass", "Leaf vein density",
        "Wood (sapwood) specific conductivity (stem specific conductivity)",
        "Stem dry mass per stem fresh mass (stem dry matter content, StDMC)",
        "Wood vessel diameter", "Plant height vegetative",
        "Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): petiole excluded",
        "Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): petiole included",
        "Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): undefined if petiole is in- or excluded",
        "Leaf mass per area (LMA)", "Wood vessel density",
        "Leaf.Longevity..Yr."
    )) %>%
    mutate(errorRiskEntries = n()) %>%
    mutate(StdValuelog2 = log2(StdValue)) %>%
    mutate(meanSpp = mean(StdValuelog2), sdSpp = sd(StdValuelog2)) %>%
    group_by(TraitName) %>%
    mutate(SDSppAvg = mean(sdSpp, na.rm = T)) %>%
    mutate(errorRisk = ((meanSpp - StdValuelog2) / SDSppAvg)) %>%
    dplyr::select(ID_TRY_INDI, gRoot_Accepted_Name, TraitName, StdValue, errorRiskEntries, errorRisk)

speciesTRYlog$StdValuelog2 <- NULL

# normally distributed data

speciesTRYnorm <- speciesTRY %>%
    dplyr::select(ID_TRY_INDI, TraitName, StdValue, gRoot_Accepted_Name) %>%
    group_by(gRoot_Accepted_Name, TraitName) %>%
    dplyr::filter(TraitName %in% c(
        "Leaf carbon (C) content per leaf dry mass",
        "Plant biomass and allometry: Leaf dry mass per plant dry mass (leaf weight ratio, LWR)",
        "Stem carbon (C) content per stem dry mass",
        "Branch vessel density", "Branch vessel diameter",
        "Leaf photosynthesis rate per stomatal conductance",
        "Leaf_Lignin"
    )) %>%
    mutate(errorRiskEntries = n()) %>%
    mutate(meanSpp = mean(StdValue), sdSpp = sd(StdValue)) %>%
    group_by(TraitName) %>%
    mutate(SDSppAvg = mean(sdSpp, na.rm = T)) %>%
    mutate(errorRisk = ((meanSpp - StdValue) / SDSppAvg)) %>%
    dplyr::select(ID_TRY_INDI, gRoot_Accepted_Name, TraitName, StdValue, errorRiskEntries, errorRisk)

# negative not normally distributed data

speciesTRYneg <- speciesTRY %>%
    dplyr::select(ID_TRY_INDI, TraitName, StdValue, gRoot_Accepted_Name) %>%
    group_by(gRoot_Accepted_Name, TraitName) %>%
    dplyr::filter(TraitName %in% c(
        "Leaf photosynthetic A/Ci curve: photosynthetic rate per leaf area",
        "Leaf photosynthetic A/Ci curve: intercellular CO2 concentration",
        "Leaf photosynthesis rate per leaf dry mass",
        "Leaf photosynthetic A/Ci curve: stomata conductance per leaf area",
        "Leaf photosynthetic A/Ci curve: transpiration rate per leaf area",
        "Leaf photosynthesis carboxylation capacity (Vcmax) per leaf dry mass (Farquhar model)",
        "Stomata conductance per leaf dry mass",
        "Leaf respiration rate in the dark per leaf dry mass", "Leaf photosynthesis rate per leaf area"
    )) %>%
    mutate(errorRiskEntries = n()) %>%
    mutate(StdValuetrans = 1 / (StdValue + 1)) %>%
    mutate(StdValuetrans = 1 / (StdValue + 1)) %>%
    mutate(meanSpp = mean(StdValuetrans), sdSpp = sd(StdValuetrans)) %>%
    group_by(TraitName) %>%
    mutate(SDSppAvg = mean(sdSpp, na.rm = T)) %>%
    mutate(errorRisk = ((meanSpp - StdValuetrans) / SDSppAvg)) %>%
    dplyr::select(ID_TRY_INDI, gRoot_Accepted_Name, TraitName, StdValue, errorRiskEntries, errorRisk)

speciesTRYneg$StdValuetrans <- NULL

speciesRisk_TRY <- rbind(speciesTRYlog, speciesTRYnorm, speciesTRYneg)

### Note! ###
### NA values for error risk are produced when only 1 data entry is available or ###
### when all data entries have the same value for the species ###

### merge error risk with other information in the database ###

TRYFull_Indi_ErrorRisk <- merge(speciesTRY, speciesRisk_TRY,
    by = c("ID_TRY_INDI", "TraitName", "StdValue", "gRoot_Accepted_Name")
)

### save data ###

write.table(TRYFull_Indi_ErrorRisk, file = "TRYFull_Indi_ErrorRisk.csv", sep = ";")


### II) Calculate Error risk values FOR GROOT ######################################################################


### data entries with information only at genus level ###
### these information is not included to calculate error risks at species level ###

genusGRooT <- dplyr::filter(Indi_and_GRoot, is.na(speciesTNRS))
genusGRooT$errorRiskEntries <- ""
genusGRooT$errorRisk <- ""

### data entries with information at species level ###
### these data was used to calculate error risk at species level ###

speciesGRooT <- dplyr::filter(Indi_and_GRoot, !is.na(speciesTNRS))

# scale_this <- function(x) as.vector(scale(x))

### error risks calculated for trait in which logarithmic transformation is required ###

speciesGRooTlog <- speciesGRooT %>%
    dplyr::select(ID_GRoot_INDI, genusTNRS, speciesTNRS, traitName, traitValue) %>%
    group_by(genusTNRS, speciesTNRS, traitName) %>%
    dplyr::filter(traitName %in% c(
        "Root_cortex_thickness", "Root_stele_diameter", "Root_stele_fraction", "Root_vessel_diameter",
        "Root_branching_density", "Root_branching_ratio", "Root_C_N_ratio",
        "Root_Ca_concentration", "Root_K_concentration", "Root_Mg_concentration",
        "Root_Mn_concentration", "Root_N_concentration", "Root_N_P_ratio", "Root_P_concentration",
        "Root_lifespan_mean", "Root_lifespan_median", "Root_litter_mass_loss_rate", "Root_production",
        "Root_turnover_rate", "Mean_Root_diameter", "Root_dry_matter_content", "Root_tissue_density",
        "Specific_root_area", "Specific_root_length", "Specific_root_respiration",
        "Coarse_root_fine_root_mass_ratio", "Fine_root_mass_leaf_mass_ratio", "Root_length_density_volume",
        "Root_mass_density", "Max_Rooting_Depth", "Cortex...m.", "Stele...m.", "Fine.root.longevity..yr."
    )) %>%
    mutate(errorRiskEntries = n()) %>%
    mutate(traitValuelog2 = log2(traitValue + 0.0001)) %>% ### 0.0001 was added to include values = 0
    mutate(meanSpp = mean(traitValuelog2), sdSpp = sd(traitValuelog2)) %>%
    group_by(traitName) %>%
    mutate(SDSppAvg = mean(sdSpp, na.rm = T)) %>%
    mutate(errorRisk = ((meanSpp - traitValuelog2) / SDSppAvg)) %>%
    dplyr::select(ID_GRoot_INDI, genusTNRS, speciesTNRS, traitName, traitValue, errorRiskEntries, errorRisk)

#### Error risk of zero means that only one observations is present for the that specific trait and species combination##


#### error risk for trait which follow a normal distribution###

speciesGRooTother <- speciesGRooT %>%
    dplyr::select(ID_GRoot_INDI, genusTNRS, speciesTNRS, traitName, traitValue) %>%
    group_by(genusTNRS, speciesTNRS, traitName) %>%
    dplyr::filter(traitName %in% c(
        "Root_xylem_vessel_number", "Root_mass_fraction", "Root_C_concentration",
        "Root_lignin_concentration", "Root_total_structural_carbohydrate_concentration",
        "Lateral_spread", "Root_mycorrhizal colonization", "Net_nitrogen_uptake_rate"
    )) %>%
    mutate(errorRiskEntries = n()) %>%
    mutate(meanSpp = mean(traitValue), sdSpp = sd(traitValue)) %>%
    group_by(traitName) %>%
    mutate(SDSppAvg = mean(sdSpp, na.rm = T)) %>%
    mutate(errorRisk = ((meanSpp - traitValue) / SDSppAvg)) %>%
    dplyr::select(ID_GRoot_INDI, genusTNRS, speciesTNRS, traitName, traitValue, errorRiskEntries, errorRisk)

speciesRisk_GRoot <- rbind(speciesGRooTlog, speciesGRooTother)

### Zero values for error risk are produced when only 1 data entry is available or ###
### when all data entries have the same value for the species ###

### merge error risk with other information in the database ###

speciesTotal_GRoot <- merge(speciesGRooT, speciesRisk_GRoot, by = c("ID_GRoot_INDI", "genusTNRS", "speciesTNRS", "traitName", "traitValue"))

### join the data at species and genus level ###

GRootFull_Indi_ErrorRisk <- rbind(speciesTotal_GRoot, genusGRooT)

### save data ###

write.csv(GRootFull_Indi_ErrorRisk, "GRootFull_Indi_ErrorRisk.csv", row.names = FALSE)

### III) Calculate Cortex fraction ###########################################

# clear workspace

rm(list = setdiff(ls(), c("GRootFull_Indi_ErrorRisk", "TRYFull_Indi_ErrorRisk")))

# Shortcut to load data from previous sections

# GRootFull_Indi_ErrorRisk <- read.csv('GRootFull_Indi_ErrorRisk.csv',header=T)

### calculate and add cortex fraction (CF)

GRootFull_Indi_ErrorRisk$Individual_data <- as.numeric(as.character(GRootFull_Indi_ErrorRisk$Individual_data))

Only_CF_cal <- GRootFull_Indi_ErrorRisk %>%
    dplyr::filter(between(errorRisk, -4, 4)) %>%
    dplyr::filter(traitName %in% c("Root_stele_diameter", "Mean_Root_diameter")) %>%
    dplyr::filter(is.na(Individual_data))

Only_CF_cal$ID_GRoot_INDI <- NULL
Only_CF_cal$errorRiskEntries <- NULL
Only_CF_cal$errorRisk <- NULL

Add_CF_wide <- spread(Only_CF_cal, traitName, traitValue)

Add_CF_wide$stele_area <- 3.14 * (Add_CF_wide$Root_stele_diameter / 1000)^2
Add_CF_wide$total_area <- 3.14 * Add_CF_wide$Mean_Root_diameter^2
Add_CF_wide$cortex_fraction <- 1 - (Add_CF_wide$stele_area / Add_CF_wide$total_area)

Add_CF_wide <- Add_CF_wide %>%
    drop_na(cortex_fraction)

Add_CV_long <- gather(Add_CF_wide, traitName, traitValue, Mean_Root_diameter:Root_stele_diameter:stele_area:total_area:cortex_fraction, factor_key = TRUE)

Only_CF <- Add_CV_long %>%
    dplyr::filter(traitName %in% c("cortex_fraction"))

Only_CF$errorRisk[Only_CF$traitName == "cortex_fraction"] <- 0
Only_CF$errorRiskEntries[Only_CF$traitName == "cortex_fraction"] <- 1

Only_CF$ID_GRoot_INDI <- seq(107269, 108112)

GRootFull_Indi_ErrorRisk <- rbind(GRootFull_Indi_ErrorRisk, Only_CF)

### save data ###

write.csv(GRootFull_Indi_ErrorRisk, "GRootFull_Indi_ErrorRisk.csv", row.names = FALSE)


### IV) Calculate mean, median, first and third percentiles per trait ###########################################

# clear workspace

rm(list = setdiff(ls(), c("GRootFull_Indi_ErrorRisk", "TRYFull_Indi_ErrorRisk")))

# Shortcut to load data from previous sections

# TRYFull_Indi_ErrorRisk <- read.csv('TRYFull_Indi_ErrorRisk.csv',header=T,sep=';')
# GRootFull_Indi_ErrorRisk <- read.csv('GRootFull_Indi_ErrorRisk.csv',header=T)

# change remaining synonym Fuscospora in Nothofagus

TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Fuscospora solandri")] <- "Nothofagus solandri"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Lophozonia menziesii")] <- "Nothofagus menziesii"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Fuscospora fusca")] <- "Nothofagus fusca"

## Mean, median and quantiles for Aboveground data

speciesTRYFull <- TRYFull_Indi_ErrorRisk[which(!grepl("^\\w+$", TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name)), ]

speciesTRYFull <- speciesTRYFull[!speciesTRYFull$StdValue == 0, ]

speciesTRYFull$errorRisk[is.na(speciesTRYFull$errorRisk)] <- 0


# Normal distributed traits

TRY_Indi_Summary_nor <- speciesTRYFull %>%
    dplyr::filter(between(errorRisk, -4, 4)) %>%
    dplyr::select(gRoot_Accepted_Name, TraitName, StdValue) %>%
    group_by(TraitName) %>%
    dplyr::filter(TraitName %in% c(
        "Leaf carbon (C) content per leaf dry mass",
        "Plant biomass and allometry: Leaf dry mass per plant dry mass (leaf weight ratio, LWR)",
        "Stem carbon (C) content per stem dry mass",
        "Branch vessel density", "Branch vessel diameter",
        "Leaf photosynthesis rate per stomatal conductance",
        "Leaf_Lignin",
        "Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)"
    )) %>%
    group_by(TraitName) %>%
    summarise(
        nspec = length(unique(gRoot_Accepted_Name)), nobs = n(), meanTrait = mean(StdValue), medianTrait = median(StdValue), firstQuantileTrait = quantile(StdValue, probs = c(0.25)),
        thirdQuantileTrait = quantile(StdValue, probs = c(0.75))
    )


# Not normal distributed traits

speciesTRYFull <- TRYFull_Indi_ErrorRisk[which(!grepl("^\\w+$", TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name)), ]

speciesTRYFull <- speciesTRYFull[!speciesTRYFull$StdValue == 0, ]

TRY_Indi_Summary_notnor <- speciesTRYFull %>%
    dplyr::filter(between(errorRisk, -4, 4)) %>%
    dplyr::select(gRoot_Accepted_Name, TraitName, StdValue) %>%
    group_by(TraitName) %>%
    dplyr::filter(TraitName %in% c(
        "Leaf nitrogen (N) content per leaf dry mass", "Leaf phosphorus (P) content per leaf dry mass",
        "Leaf thickness", "Leaf dry mass per leaf fresh mass (leaf dry matter content, LDMC)",
        "Leaf density (leaf tissue density, leaf dry mass per leaf volume)", "Leaf nitrogen/phosphorus (N/P) ratio",
        "Plant biomass and allometry: Stem dry mass per plant",
        "Plant biomass and allometry: Leaf dry mass per plant",
        "Plant biomass and allometry: Stem dry mass per plant dry mass per plant",
        "Leaf carbon/nitrogen (C/N) ratio", "Stem carbon/nitrogen (C/N) ratio",
        "Leaf photosynthesis electron transport capacity (Jmax) per leaf dry mass (Farquhar model)",
        "Stem nitrogen (N) content per stem dry mass",
        "Stem phosphorus (P) content per stem dry mass", "Stem nitrogen/phosphorus (N/P) ratio",
        "Plant biomass and allometry: Plant dry mass", "Leaf vein density",
        "Wood (sapwood) specific conductivity (stem specific conductivity)",
        "Stem dry mass per stem fresh mass (stem dry matter content, StDMC)",
        "Wood vessel diameter", "Plant height vegetative",
        "Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): petiole excluded",
        "Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): petiole included",
        "Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA): undefined if petiole is in- or excluded",
        "Leaf mass per area (LMA)", "Wood vessel density",
        "Leaf.Longevity..Yr."
    )) %>%
    mutate(logStdValue = log(StdValue)) %>%
    group_by(TraitName) %>%
    summarise(
        nspec = length(unique(gRoot_Accepted_Name)), nobs = n(), logmeanTrait = mean(logStdValue), expmean = exp(logmeanTrait),
        medianTrait = median(StdValue), firstQuantileTrait = quantile(StdValue, probs = c(0.25)),
        thirdQuantileTrait = quantile(StdValue, probs = c(0.75))
    )

# rbind summarys

TRY_Indi_Summary_notnor$logmeanTrait <- NULL
colnames(TRY_Indi_Summary_notnor)[4] <- "meanTrait"

# final summary containing number of observations per trait, species per trait, mean, median and quantiles

TRY_Summary <- rbind(TRY_Indi_Summary_nor, TRY_Indi_Summary_notnor)

## Mean, median and quantiles for belowground data

speciesGRootFull <- dplyr::filter(GRootFull_Indi_ErrorRisk, !is.na(speciesTNRS))

speciesGRootFull <- speciesGRootFull[!speciesGRootFull$traitValue == 0, ]

speciesGRootFull$gRoot_Accepted_Name <- paste(speciesGRootFull$genusTNRS, speciesGRootFull$speciesTNRS)

speciesGRootFull$errorRisk[is.na(speciesGRootFull$errorRisk)] <- 0

# change remaining synonyms

speciesGRootFull$gRoot_Accepted_Name[which(speciesGRootFull$gRoot_Accepted_Name == "Fuscospora fusca")] <- "Nothofagus fusca"
speciesGRootFull$gRoot_Accepted_Name[which(speciesGRootFull$gRoot_Accepted_Name == "Lophozonia menziesii")] <- "Nothofagus menziesii"
speciesGRootFull$gRoot_Accepted_Name[which(speciesGRootFull$gRoot_Accepted_Name == "Fuscospora fusca")] <- "Nothofagus fusca"
speciesGRootFull$gRoot_Accepted_Name[which(speciesGRootFull$gRoot_Accepted_Name == "Fuscospora truncata")] <- "Nothofagus fusca"

# Normal distributed traits

GRoot_Indi_Summary_nor <- speciesGRootFull %>%
    dplyr::filter(between(errorRisk, -4, 4)) %>%
    dplyr::select(gRoot_Accepted_Name, traitName, traitValue) %>%
    group_by(traitName) %>%
    dplyr::filter(traitName %in% c(
        "Root_xylem_vessel_number", "Root_mass_fraction", "Root_C_concentration",
        "Root_lignin_concentration", "Root_total_structural_carbohydrate_concentration",
        "Lateral_spread", "Root_mycorrhizal colonization", "Net_nitrogen_uptake_rate", "cortex_fraction"
    )) %>%
    group_by(traitName) %>%
    summarise(
        nspec = length(unique(gRoot_Accepted_Name)), nobs = n(), meanTrait = mean(traitValue), medianTrait = median(traitValue),
        firstQuantileTrait = quantile(traitValue, probs = c(0.25)),
        thirdQuantileTrait = quantile(traitValue, probs = c(0.75))
    )

# Not normal distributed traits

GRoot_Indi_Summary_notnor <- speciesGRootFull %>%
    dplyr::filter(between(errorRisk, -4, 4)) %>%
    dplyr::select(gRoot_Accepted_Name, traitName, traitValue) %>%
    group_by(traitName) %>%
    dplyr::filter(traitName %in% c(
        "Root_cortex_thickness", "Root_stele_diameter", "Root_stele_fraction", "Root_vessel_diameter",
        "Root_branching_density", "Root_branching_ratio", "Root_C_N_ratio",
        "Root_Ca_concentration", "Root_K_concentration", "Root_Mg_concentration",
        "Root_Mn_concentration", "Root_N_concentration", "Root_N_P_ratio", "Root_P_concentration",
        "Root_lifespan_mean", "Root_lifespan_median", "Root_litter_mass_loss_rate", "Root_production",
        "Root_turnover_rate", "Mean_Root_diameter", "Root_dry_matter_content", "Root_tissue_density",
        "Specific_root_area", "Specific_root_length", "Specific_root_respiration",
        "Coarse_root_fine_root_mass_ratio", "Fine_root_mass_leaf_mass_ratio", "Root_length_density_volume",
        "Root_mass_density", "Cortex...m.", "Stele...m.", "Fine.root.longevity..yr.", "Max_Rooting_Depth"
    )) %>%
    mutate(logStdValue = log(traitValue)) %>%
    group_by(traitName) %>%
    summarise(
        nspec = length(unique(gRoot_Accepted_Name)), nobs = n(), logmeanTrait = mean(logStdValue), expmean = exp(logmeanTrait),
        medianTrait = median(traitValue), firstQuantileTrait = quantile(traitValue, probs = c(0.25)),
        thirdQuantileTrait = quantile(traitValue, probs = c(0.75))
    )

# rbind summarys

GRoot_Indi_Summary_notnor$logmeanTrait <- NULL
colnames(GRoot_Indi_Summary_notnor)[4] <- "meanTrait"

# final summary containing number of observations per trait, species per trait, mean, median and quantiles

GRoot_Summary <- rbind(GRoot_Indi_Summary_nor, GRoot_Indi_Summary_notnor)


###############################################################################################################################
# PCA with 6 core traits + pairwise correlations (with residuals)
###############################################################################################################################


# clear workspace

rm(list = setdiff(ls(), c("GRootFull_Indi_ErrorRisk", "TRYFull_Indi_ErrorRisk")))

# Shortcut to load data from previous sections

# TRYFull_Indi_ErrorRisk <- read.csv("TRYFull_Indi_ErrorRisk.csv",header=T,sep=';',na.strings=c("","NA"))
# GRootFull_Indi_ErrorRisk <- read.csv('GRootFull_Indi_ErrorRisk.csv',header=T)

# change remaining synonyms

TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Fuscospora truncata")] <- "Nothofagus fusca"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Fuscospora solandri")] <- "Nothofagus solandri"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Lophozonia menziesii")] <- "Nothofagus menziesii"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Fuscospora fusca")] <- "Nothofagus fusca"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Psoralea bituminosa")] <- "Bituminaria bituminosa"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Andropogon gerardi")] <- "Andropogon gerardii"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Canarium tonkinense")] <- "Canarium album"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Bouteloua gracilis")] <- "Chondrosum gracile"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Chamerion angustifolium")] <- "Epilobium angustifolium"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Acacia albida")] <- "Faidherbia albida"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Helianthemum canum")] <- "Helianthemum oelandicum"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Chrysopsis villosa")] <- "Heterotheca villosa"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Michelia cavaleriei")] <- "Magnolia cavaleriei"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Microlaena stipoides")] <- "Ehrharta stipoides"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Nothofagus truncata")] <- "Nothofagus fusca"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Populus davidiana")] <- "Populus tremula"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Reaumuria songarica")] <- "Reaumuria soongarica"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Leontodon autumnalis")] <- "Scorzoneroides autumnalis"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Potentilla bifurca")] <- "Sibbaldianthe bifurca"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Thalictrum aquilegifolium")] <- "Thalictrum aquilegiifolium"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Rudbeckia serotina")] <- "Echinacea serotina"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Chenopodium glaucum")] <- "Oxybasis glauca"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Leucanthemum ircutianum")] <- "Leucanthemum vulgare"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Potentilla tabernaemontani")] <- "Potentilla neumanniana"
TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name[which(TRYFull_Indi_ErrorRisk$gRoot_Accepted_Name == "Quercus wutaishanica")] <- "Quercus mongolica"


GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Fuscospora" & GRootFull_Indi_ErrorRisk$speciesTNRS == "truncata")] <- "Nothofagus"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Nothofagus" & GRootFull_Indi_ErrorRisk$speciesTNRS == "truncata")] <- "fusca"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Fuscospora" & GRootFull_Indi_ErrorRisk$speciesTNRS == "solandri")] <- "Nothofagus"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Fuscospora" & GRootFull_Indi_ErrorRisk$speciesTNRS == "fusca")] <- "Nothofagus"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Lophozonia" & GRootFull_Indi_ErrorRisk$speciesTNRS == "menziesii")] <- "Nothofagus"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Psoralea" & GRootFull_Indi_ErrorRisk$speciesTNRS == "bituminosa")] <- "Bituminaria"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Andropogon" & GRootFull_Indi_ErrorRisk$speciesTNRS == "gerardi")] <- "gerardii"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Canarium" & GRootFull_Indi_ErrorRisk$speciesTNRS == "tonkinense")] <- "album"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Bouteloua" & GRootFull_Indi_ErrorRisk$speciesTNRS == "gracilis")] <- "Chondrosum"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Chondrosum" & GRootFull_Indi_ErrorRisk$speciesTNRS == "gracilis")] <- "gracile"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Chamerion" & GRootFull_Indi_ErrorRisk$speciesTNRS == "angustifolium")] <- "Epilobium"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Acacia" & GRootFull_Indi_ErrorRisk$speciesTNRS == "albida")] <- "Faidherbia"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Helianthemum" & GRootFull_Indi_ErrorRisk$speciesTNRS == "canum")] <- "oelandicum"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Chrysopsis" & GRootFull_Indi_ErrorRisk$speciesTNRS == "villosa")] <- "Heterotheca"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Michelia" & GRootFull_Indi_ErrorRisk$speciesTNRS == "cavaleriei")] <- "Magnolia"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Microlaena" & GRootFull_Indi_ErrorRisk$speciesTNRS == "stipoides")] <- "Ehrharta"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Nothofagus" & GRootFull_Indi_ErrorRisk$speciesTNRS == "truncata")] <- "fusca"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Populus" & GRootFull_Indi_ErrorRisk$speciesTNRS == "davidiana")] <- "tremula"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Reaumuria" & GRootFull_Indi_ErrorRisk$speciesTNRS == "songarica")] <- "soongarica"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Leontodon" & GRootFull_Indi_ErrorRisk$speciesTNRS == "autumnalis")] <- "Scorzoneroides"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Potentilla" & GRootFull_Indi_ErrorRisk$speciesTNRS == "bifurca")] <- "Sibbaldianthe"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Thalictrum" & GRootFull_Indi_ErrorRisk$speciesTNRS == "aquilegifolium")] <- "aquilegiifolium"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Rudbeckia" & GRootFull_Indi_ErrorRisk$speciesTNRS == "serotina")] <- "Echinacea"
GRootFull_Indi_ErrorRisk$genusTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Chenopodium" & GRootFull_Indi_ErrorRisk$speciesTNRS == "glaucum")] <- "Oxybasis"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Oxybasis" & GRootFull_Indi_ErrorRisk$speciesTNRS == "glaucum")] <- "glauca"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Leucanthemum" & GRootFull_Indi_ErrorRisk$speciesTNRS == "ircutianum")] <- "vulgare"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Potentilla" & GRootFull_Indi_ErrorRisk$speciesTNRS == "tabernaemontani")] <- "neumanniana"
GRootFull_Indi_ErrorRisk$speciesTNRS[which(GRootFull_Indi_ErrorRisk$genusTNRS == "Quercus" & GRootFull_Indi_ErrorRisk$speciesTNRS == "wutaishanica")] <- "mongolica"

# extract meta data (on myccorrhizal association, N-Fixation ability, woodiness) from GRooT

add_info1 <- GRootFull_Indi_ErrorRisk %>%
    mutate(gRoot_Accepted_Name = if_else(is.na(speciesTNRS), genusTNRS, paste(genusTNRS, speciesTNRS))) %>%
    dplyr::select("gRoot_Accepted_Name", "woodiness") %>%
    distinct() %>%
    drop_na()

add_info2 <- GRootFull_Indi_ErrorRisk %>%
    mutate(gRoot_Accepted_Name = if_else(is.na(speciesTNRS), genusTNRS, paste(genusTNRS, speciesTNRS))) %>%
    dplyr::select("gRoot_Accepted_Name", "nitrogenFixationNodDB") %>%
    distinct() %>%
    drop_na()

add_info3 <- GRootFull_Indi_ErrorRisk %>%
    mutate(gRoot_Accepted_Name = if_else(is.na(speciesTNRS), genusTNRS, paste(genusTNRS, speciesTNRS))) %>%
    dplyr::select("gRoot_Accepted_Name", "mycorrhizalAssociationTypeFungalRoot") %>%
    distinct() %>%
    drop_na()

add_info <- merge(add_info1, add_info2, by = "gRoot_Accepted_Name", all.x = T)
add_info <- merge(add_info, add_info3, by = "gRoot_Accepted_Name", all.x = T)

colnames(add_info)[3] <- "N_fixation"
colnames(add_info)[4] <- "Mycorrhizal_type"

add_info <- add_info[!duplicated(add_info$gRoot_Accepted_Name), ]

# merge meta data from GRooT with TRY data

TRY_INDI_CLEAN_META_ErrorRisk <- merge(TRYFull_Indi_ErrorRisk, add_info, by = "gRoot_Accepted_Name", all.x = T)

# in order to calculate residuals we need to rename all NA to unknown (third category)

TRY_INDI_CLEAN_META_ErrorRisk$Growth_conditions <- as.character(TRY_INDI_CLEAN_META_ErrorRisk$Growth_conditions)
TRY_INDI_CLEAN_META_ErrorRisk$Growth_conditions[is.na(TRY_INDI_CLEAN_META_ErrorRisk$Growth_conditions)] <- "Unknown"

### subseting individual data and calculate means per dateset

TRY_INDI_CLEAN_META_ErrorRisk$Individual_data <- as.numeric(as.character(TRY_INDI_CLEAN_META_ErrorRisk$Individual_data))

Indi_study_mean <- TRY_INDI_CLEAN_META_ErrorRisk[!is.na(TRY_INDI_CLEAN_META_ErrorRisk$Individual_data), ]
TRY_without_indi <- TRY_INDI_CLEAN_META_ErrorRisk[is.na(TRY_INDI_CLEAN_META_ErrorRisk$Individual_data), ]

Indi_study_mean <- Indi_study_mean %>%
    group_by(Reference, TraitName, gRoot_Accepted_Name) %>%
    mutate(studyMean = mean(StdValue))

Indi_study_mean$StdValue <- NULL
colnames(Indi_study_mean)[41] <- "StdValue"

TRY_Indi_ER <- bind_rows(TRY_without_indi, Indi_study_mean)

### I) TRY: Calculating residuals ########################################################################

# in order to correct the data we calculate log and z transform the data and calculate the residuals including studies and growth conditions
# as fixed or random factors (pot, field, unknown)

# subset data

Above_coreTraits <- TRY_Indi_ER %>%
    dplyr::select(
        gRoot_Accepted_Name, TraitName, StdValue, LastName, FirstName, Reference, Health_status,
        Growth_conditions, errorRiskEntries, errorRisk, woodiness, Mycorrhizal_type, N_fixation, ID_TRY_INDI
    ) %>%
    dplyr::filter(TraitName %in% c(
        "Leaf mass per area (LMA)",
        "Leaf nitrogen (N) content per leaf dry mass",
        "Plant height vegetative",
        "Leaf phosphorus (P) content per leaf dry mass",
        "Leaf_Lignin",
        "Leaf thickness",
        "Leaf density (leaf tissue density, leaf dry mass per leaf volume)",
        "Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)"
    ))

# log and z transformed core traits (for all values with errorRisk = 0 only the log and not the scale will be calculated)

Above_coreTraits$errorRisk[is.na(Above_coreTraits$errorRisk)] <- 0

Above_coreTraits_trans <- Above_coreTraits %>%
    dplyr::filter(between(errorRisk, -4, 4)) %>%
    dplyr::select(
        gRoot_Accepted_Name, TraitName, StdValue, LastName, FirstName, Reference, Health_status,
        Growth_conditions, errorRiskEntries, errorRisk, woodiness, Mycorrhizal_type, N_fixation, ID_TRY_INDI
    ) %>%
    group_by(TraitName) %>%
    mutate(Log_StdValue = log(StdValue)) %>%
    mutate(scaled_LogStdValue = scale(Log_StdValue))

# simplify trait names

Above_coreTraits_trans$TraitName <- as.character(Above_coreTraits_trans$TraitName)

Above_coreTraits_trans$TraitName[which(Above_coreTraits_trans$TraitName == "Leaf mass per area (LMA)")] <- "LMA"
Above_coreTraits_trans$TraitName[which(Above_coreTraits_trans$TraitName == "Leaf nitrogen (N) content per leaf dry mass")] <- "Leaf_N"
Above_coreTraits_trans$TraitName[which(Above_coreTraits_trans$TraitName == "Plant height vegetative")] <- "Height"
Above_coreTraits_trans$TraitName[which(Above_coreTraits_trans$TraitName == "Leaf phosphorus (P) content per leaf dry mass")] <- "Leaf_P"
Above_coreTraits_trans$TraitName[which(Above_coreTraits_trans$TraitName == "Leaf_Lignin")] <- "Leaf_L"
Above_coreTraits_trans$TraitName[which(Above_coreTraits_trans$TraitName == "Leaf thickness")] <- "Leaf_thick"
Above_coreTraits_trans$TraitName[which(Above_coreTraits_trans$TraitName == "Leaf density (leaf tissue density, leaf dry mass per leaf volume)")] <- "LTD"
Above_coreTraits_trans$TraitName[which(Above_coreTraits_trans$TraitName == "Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)")] <- "SSD"

# reshape data

Above_trans_reshaped <- dcast(Above_coreTraits_trans, gRoot_Accepted_Name + Reference +
    Growth_conditions + woodiness + Mycorrhizal_type + N_fixation + ID_TRY_INDI ~ TraitName, value.var = "scaled_LogStdValue")
Above_trans_reshaped$ID_TRY_INDI <- NULL

# calculate residuals for LMA

model1 <- lmer(LMA ~ Growth_conditions +
    (1 | Reference), data = Above_trans_reshaped[complete.cases(Above_trans_reshaped[, c(12, 2, 3)]), ])

Above_trans_reshaped$LMA_corrected <- rep(NA, nrow(Above_trans_reshaped))
Above_trans_reshaped$LMA_corrected[complete.cases(Above_trans_reshaped[, c(12, 2, 3)])] <- residuals(model1)

# calculate residuals for Leaf Nitrogen

model2 <- lmer(Leaf_N ~ Growth_conditions +
    (1 | Reference), data = Above_trans_reshaped[complete.cases(Above_trans_reshaped[, c(9, 2, 3)]), ])

Above_trans_reshaped$Leaf_N_corrected <- rep(NA, nrow(Above_trans_reshaped))
Above_trans_reshaped$Leaf_N_corrected[complete.cases(Above_trans_reshaped[, c(9, 2, 3)])] <- residuals(model2)

# calculate residuals for height

model3 <- lmer(Height ~ Growth_conditions +
    (1 | Reference), data = Above_trans_reshaped[complete.cases(Above_trans_reshaped[, c(7, 2, 3)]), ])

Above_trans_reshaped$Height_corrected <- rep(NA, nrow(Above_trans_reshaped))
Above_trans_reshaped$Height_corrected[complete.cases(Above_trans_reshaped[, c(7, 2, 3)])] <- residuals(model3)

# calculate residuals for Leaf Phosphorus

model4 <- lmer(Leaf_P ~ Growth_conditions +
    (1 | Reference), data = Above_trans_reshaped[complete.cases(Above_trans_reshaped[, c(10, 2, 3)]), ])

Above_trans_reshaped$Leaf_P_corrected <- rep(NA, nrow(Above_trans_reshaped))
Above_trans_reshaped$Leaf_P_corrected[complete.cases(Above_trans_reshaped[, c(10, 2, 3)])] <- residuals(model4)

# calculate residuals for Leaf Lignin

model5 <- lmer(Leaf_L ~ Growth_conditions +
    (1 | Reference), data = Above_trans_reshaped[complete.cases(Above_trans_reshaped[, c(8, 2, 3)]), ])

Above_trans_reshaped$Leaf_L_corrected <- rep(NA, nrow(Above_trans_reshaped))
Above_trans_reshaped$Leaf_L_corrected[complete.cases(Above_trans_reshaped[, c(8, 2, 3)])] <- residuals(model5)

# calculate residuals for Leaf thickness

model6 <- lmer(Leaf_thick ~ Growth_conditions +
    (1 | Reference), data = Above_trans_reshaped[complete.cases(Above_trans_reshaped[, c(11, 2, 3)]), ])

Above_trans_reshaped$Leaf_thick_corrected <- rep(NA, nrow(Above_trans_reshaped))
Above_trans_reshaped$Leaf_thick_corrected[complete.cases(Above_trans_reshaped[, c(11, 2, 3)])] <- residuals(model6)

# calculate residuals for LTD

model7 <- lmer(LTD ~ Growth_conditions +
    (1 | Reference), data = Above_trans_reshaped[complete.cases(Above_trans_reshaped[, c(13, 2, 3)]), ])

Above_trans_reshaped$LTD_corrected <- rep(NA, nrow(Above_trans_reshaped))
Above_trans_reshaped$LTD_corrected[complete.cases(Above_trans_reshaped[, c(13, 2, 3)])] <- residuals(model7)

# calculate residuals for SSD

model8 <- lmer(SSD ~ Growth_conditions +
    (1 | Reference), data = Above_trans_reshaped[complete.cases(Above_trans_reshaped[, c(14, 2, 3)]), ])

Above_trans_reshaped$SSD_corrected <- rep(NA, nrow(Above_trans_reshaped))
Above_trans_reshaped$SSD_corrected[complete.cases(Above_trans_reshaped[, c(14, 2, 3)])] <- residuals(model8)

# reduce data frame

Above_trans_reshaped <- Above_trans_reshaped[c(1, 15:22)]


### II) GRoot: Calculating residuals (BLUPs) ###################################################################################

speciesGRooT <- dplyr::filter(GRootFull_Indi_ErrorRisk, !is.na(speciesTNRS))

# subset Only for Fine roots

speciesGRooT$errorRisk[is.na(speciesGRooT$errorRisk)] <- 0
speciesGRooT$errorRisk[speciesGRooT$errorRisk == "NaN"] <- 0

speciesGRooT$belowgroundEntities[speciesGRooT$traitName == "Max_Rooting_Depth"] <- "FR"
speciesGRooT$measurementProvenance[speciesGRooT$traitName == "Max_Rooting_Depth"] <- "Field"

GRooTAggregateSpeciesVersion1 <- speciesGRooT %>%
    dplyr::filter(between(errorRisk, -4, 4)) %>%
    dplyr::select(
        belowgroundEntities, genusTNRS, speciesTNRS, traitName,
        traitValue, growthForm, woodiness, vitality, references, errorRisk, measurementProvenance
    ) %>%
    dplyr::filter(belowgroundEntities == "FR") %>%
    dplyr::filter(traitName %in% c(
        "Root_N_concentration",
        "Mean_Root_diameter",
        "Root_tissue_density",
        "Specific_root_length",
        "Root_mycorrhizal colonization",
        "Max_Rooting_Depth",
        "Root_P_concentration",
        "Root_lignin_concentration",
        "cortex_fraction"
    ))

# Exclude tissue density > 1 (logical error = not possible values) and ferns

GRooTAggregateSpeciesVersion2 <- GRooTAggregateSpeciesVersion1[-which(GRooTAggregateSpeciesVersion1$traitName == "Root_tissue_density" & GRooTAggregateSpeciesVersion1$traitValue > 1), ]

GRooTAggregateSpeciesVersion3 <- GRooTAggregateSpeciesVersion2[-which(GRooTAggregateSpeciesVersion2$growthForm == "fern"), ]

# simplify trait names

GRooTAggregateSpeciesVersion3$traitName <- as.character(GRooTAggregateSpeciesVersion3$traitName)

GRooTAggregateSpeciesVersion3$traitName[which(GRooTAggregateSpeciesVersion3$traitName == "Root_N_concentration")] <- "RN"
GRooTAggregateSpeciesVersion3$traitName[which(GRooTAggregateSpeciesVersion3$traitName == "Mean_Root_diameter")] <- "MRD"
GRooTAggregateSpeciesVersion3$traitName[which(GRooTAggregateSpeciesVersion3$traitName == "Root_tissue_density")] <- "RTD"
GRooTAggregateSpeciesVersion3$traitName[which(GRooTAggregateSpeciesVersion3$traitName == "Specific_root_length")] <- "SRL"
GRooTAggregateSpeciesVersion3$traitName[which(GRooTAggregateSpeciesVersion3$traitName == "Root_mycorrhizal colonization")] <- "Myc_col"
GRooTAggregateSpeciesVersion3$traitName[which(GRooTAggregateSpeciesVersion3$traitName == "Max_Rooting_Depth")] <- "R_depth"
GRooTAggregateSpeciesVersion3$traitName[which(GRooTAggregateSpeciesVersion3$traitName == "Root_P_concentration")] <- "Root_P"
GRooTAggregateSpeciesVersion3$traitName[which(GRooTAggregateSpeciesVersion3$traitName == "Root_lignin_concentration")] <- "Root_L"
GRooTAggregateSpeciesVersion3$traitName[which(GRooTAggregateSpeciesVersion3$traitName == "cortex_fraction")] <- "CF"

# calculating the mean and z transformation (GRoot) -> traits CF and Myc_col were transformed seperately using arcsin square root transformation + Myc_col was additionally devided by 100
# despite CF and Myc_col were transformed differently we saved the transformed values in the same calumns as the log transformed values to simplify the process

GRooTAggregateSpeciesVersion4 <- GRooTAggregateSpeciesVersion3[!GRooTAggregateSpeciesVersion3$traitValue == 0, ]

GRooTAggregateSpeciesVersion4$traitValue[which(GRooTAggregateSpeciesVersion4$traitName == "Myc_col")] <- GRooTAggregateSpeciesVersion4$traitValue[which(GRooTAggregateSpeciesVersion4$traitName == "Myc_col")] / 100

GRoot_coreTraits_trans1 <- GRooTAggregateSpeciesVersion4 %>%
    dplyr::select(genusTNRS, speciesTNRS, traitName, traitValue, references, measurementProvenance) %>%
    dplyr::filter(traitName %in% c(
        "RN",
        "MRD",
        "RTD",
        "SRL",
        "R_depth",
        "Root_P",
        "Root_L"
    )) %>%
    group_by(traitName) %>%
    mutate(Log_StdValue = log(traitValue)) %>%
    mutate(scaled_LogStdValue = scale(Log_StdValue))

GRoot_coreTraits_trans2 <- GRooTAggregateSpeciesVersion4 %>%
    dplyr::select(genusTNRS, speciesTNRS, traitName, traitValue, references, measurementProvenance) %>%
    dplyr::filter(traitName %in% c(
        "Myc_col",
        "CF"
    )) %>%
    group_by(traitName) %>%
    mutate(Log_StdValue = asin(sqrt(traitValue))) %>%
    mutate(scaled_LogStdValue = scale(Log_StdValue))

GRoot_coreTraits_trans <- rbind(GRoot_coreTraits_trans1, GRoot_coreTraits_trans2)

# Collapsing species and genus names

GRoot_coreTraits_trans$gRoot_Accepted_Name <- paste(GRoot_coreTraits_trans$genusTNRS, GRoot_coreTraits_trans$speciesTNRS)
colnames(GRoot_coreTraits_trans)[3] <- "TraitName"
GRoot_coreTraits_trans$genusTNRS <- NULL
GRoot_coreTraits_trans$speciesTNRS <- NULL

# create unique ID for reshaping data

GRoot_coreTraits_trans$unique_ID <- seq.int(nrow(GRoot_coreTraits_trans))

# reshape data

GRoot_trans_reshaped <- dcast(GRoot_coreTraits_trans, gRoot_Accepted_Name + references + measurementProvenance + unique_ID ~ TraitName, value.var = "scaled_LogStdValue")

GRoot_trans_reshaped$unique_ID <- NULL

# aggregate unclear measurement Provenances to either  Field or Pot

GRoot_trans_reshaped$measurementProvenance[which(GRoot_trans_reshaped$measurementProvenance == "field")] <- "Field"
GRoot_trans_reshaped$measurementProvenance[which(GRoot_trans_reshaped$measurementProvenance == "potted")] <- "Pot"
GRoot_trans_reshaped$measurementProvenance[which(GRoot_trans_reshaped$measurementProvenance == "hydroponic")] <- "Field"

# calculate residuals for Root Nitrogen

model10 <- lmer(RN ~ measurementProvenance + (1 | references), data = GRoot_trans_reshaped[complete.cases(GRoot_trans_reshaped[, c(8, 2, 3)]), ])

GRoot_trans_reshaped$RN_corrected <- rep(NA, nrow(GRoot_trans_reshaped))
GRoot_trans_reshaped$RN_corrected[complete.cases(GRoot_trans_reshaped[, c(8, 2, 3)])] <- residuals(model10)

# calculate residuals for Mean Root Diameter

model11 <- lmer(MRD ~ measurementProvenance + (1 | references), data = GRoot_trans_reshaped[complete.cases(GRoot_trans_reshaped[, c(5, 2, 3)]), ])

GRoot_trans_reshaped$MRD_corrected <- rep(NA, nrow(GRoot_trans_reshaped))
GRoot_trans_reshaped$MRD_corrected[complete.cases(GRoot_trans_reshaped[, c(5, 2, 3)])] <- residuals(model11)

# calculate residuals for RTD

model12 <- lmer(RTD ~ measurementProvenance + (1 | references), data = GRoot_trans_reshaped[complete.cases(GRoot_trans_reshaped[, c(11, 2, 3)]), ])

GRoot_trans_reshaped$RTD_corrected <- rep(NA, nrow(GRoot_trans_reshaped))
GRoot_trans_reshaped$RTD_corrected[complete.cases(GRoot_trans_reshaped[, c(11, 2, 3)])] <- residuals(model12)

# calculate residuals for SRL

model13 <- lmer(SRL ~ measurementProvenance + (1 | references), data = GRoot_trans_reshaped[complete.cases(GRoot_trans_reshaped[, c(12, 2, 3)]), ])

GRoot_trans_reshaped$SRL_corrected <- rep(NA, nrow(GRoot_trans_reshaped))
GRoot_trans_reshaped$SRL_corrected[complete.cases(GRoot_trans_reshaped[, c(12, 2, 3)])] <- residuals(model13)

# calculate residuals for Micorhizal colonisation (in %)

model14 <- lmer(Myc_col ~ measurementProvenance + (1 | references), data = GRoot_trans_reshaped[complete.cases(GRoot_trans_reshaped[, c(6, 2, 3)]), ])

GRoot_trans_reshaped$Myc_col_corrected <- rep(NA, nrow(GRoot_trans_reshaped))
GRoot_trans_reshaped$Myc_col_corrected[complete.cases(GRoot_trans_reshaped[, c(6, 2, 3)])] <- residuals(model14)

# calculate residuals for Root_depth (here only reference since all maximum rooting depth measurements were performed in the field!)

model15 <- lm(R_depth ~ references, data = GRoot_trans_reshaped[complete.cases(GRoot_trans_reshaped[, c(2, 7)]), ])

GRoot_trans_reshaped$Root_depth_corrected <- rep(NA, nrow(GRoot_trans_reshaped))
GRoot_trans_reshaped$Root_depth_corrected[complete.cases(GRoot_trans_reshaped[, c(2, 7)])] <- residuals(model15)

# calculate residuals for Root Lignin

model16 <- lmer(Root_L ~ measurementProvenance + (1 | references), data = GRoot_trans_reshaped[complete.cases(GRoot_trans_reshaped[, c(2, 3, 9)]), ])

GRoot_trans_reshaped$Root_L_corrected <- rep(NA, nrow(GRoot_trans_reshaped))
GRoot_trans_reshaped$Root_L_corrected[complete.cases(GRoot_trans_reshaped[, c(2, 3, 9)])] <- residuals(model16)

# calculate residuals for Root_P

model17 <- lmer(Root_P ~ measurementProvenance + (1 | references), data = GRoot_trans_reshaped[complete.cases(GRoot_trans_reshaped[, c(2, 3, 10)]), ])

GRoot_trans_reshaped$Root_P_corrected <- rep(NA, nrow(GRoot_trans_reshaped))
GRoot_trans_reshaped$Root_P_corrected[complete.cases(GRoot_trans_reshaped[, c(2, 3, 10)])] <- residuals(model17)

# calculate residuals for CF

model18 <- lmer(CF ~ measurementProvenance + (1 | references), data = GRoot_trans_reshaped[complete.cases(GRoot_trans_reshaped[, c(2, 3, 4)]), ])

GRoot_trans_reshaped$CF_corrected <- rep(NA, nrow(GRoot_trans_reshaped))
GRoot_trans_reshaped$CF_corrected[complete.cases(GRoot_trans_reshaped[, c(2, 3, 4)])] <- residuals(model18)


GRoot_trans_reshaped <- GRoot_trans_reshaped[c(1, 13:21)]

### III) Combine above with belowground ##########################################################################################

# Aboveground: calcuate species averages

colnames(Above_trans_reshaped)[2] <- "LMA"
colnames(Above_trans_reshaped)[3] <- "LN"
colnames(Above_trans_reshaped)[4] <- "Height"
colnames(Above_trans_reshaped)[5] <- "LP"
colnames(Above_trans_reshaped)[6] <- "LL"
colnames(Above_trans_reshaped)[7] <- "Lth"
colnames(Above_trans_reshaped)[8] <- "LTD"
colnames(Above_trans_reshaped)[9] <- "SSD"

Above_corrected_means <- aggregate(
    cbind(LMA, LN, Height, LP, LL, Lth, LTD, SSD) ~
        gRoot_Accepted_Name,
    data = Above_trans_reshaped,
    FUN = mean, na.rm = TRUE, na.action = NULL
)

# Belowground: calcuate species averages

colnames(GRoot_trans_reshaped)[2] <- "RN"
colnames(GRoot_trans_reshaped)[3] <- "MRD"
colnames(GRoot_trans_reshaped)[9] <- "RP"
colnames(GRoot_trans_reshaped)[4] <- "RTD"
colnames(GRoot_trans_reshaped)[5] <- "SRL"
colnames(GRoot_trans_reshaped)[6] <- "Myc_col"
colnames(GRoot_trans_reshaped)[7] <- "Rdep"
colnames(GRoot_trans_reshaped)[8] <- "RL"
colnames(GRoot_trans_reshaped)[10] <- "CF"

GRoot_corrected_means <- aggregate(
    cbind(RN, MRD, RP, RTD, SRL, Myc_col, Rdep, RL, CF) ~
        gRoot_Accepted_Name,
    data = GRoot_trans_reshaped,
    FUN = mean, na.rm = TRUE, na.action = NULL
)


# in order to rbind root and leaf core traits

Above_corrected_means <- as.data.frame(Above_corrected_means)
GRoot_corrected_means <- as.data.frame(GRoot_corrected_means)

Core_combined <- merge(Above_corrected_means, GRoot_corrected_means, by = "gRoot_Accepted_Name")

# merge data with meta data (on myccorrhizal association, N-Fixation ability, woodiness) from GRooT

Core_combined_meta <- merge(Core_combined, add_info, by = "gRoot_Accepted_Name", all.x = TRUE)

# adding some information on growth Form and woodiness

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Cornus bretschneideri")] <- "woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Millettia leptobotrya")] <- "woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Tanaecium pyramidatum")] <- "woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Chondrosum gracile")] <- "non-woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Chrysopsis villosa")] <- "non-woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Dasiphora fruticosa")] <- "woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Persicaria vivipara")] <- "non-woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Sibbaldianthe bifurca")] <- "non-woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Stipa tianschanica")] <- "non-woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Taraxacum campylodes")] <- "non-woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Magnolia cavaleriei")] <- "non-woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Bassia prostrata")] <- "woody"

Core_combined_meta$woodiness[which(Core_combined_meta$woodiness == "unknown" &
    Core_combined_meta$gRoot_Accepted_Name == "Lespedeza davurica")] <- "non-woody"

# rename species name and change space to underscore

colnames(Core_combined_meta)[1] <- "full_species"
Core_combined_meta$full_species <- gsub(" ", "_", Core_combined_meta$full_species)

# FINAL dataset for PCA with core traits and meta data

Core_combined_meta[is.na(Core_combined_meta)] <- as.numeric(NA)

# delete species that don't have species names and remove ferns

Core_combined_meta <- Core_combined_meta %>%
    dplyr::filter(!full_species %in% c(
        "Salsola_spp.a", "Salsola_spp.b", "Salsola_spp.c", "Lonicera_x", "Lycopodium_complanatum", "Euterpe_precatoria",
        "Hemerocallis_citrina", "Podocarpium_podocarpum", "Cystopteris_fragilis", "Diplopterygium_glaucum",
        "Equisetum_arvense", "Equisetum_fluviatile",
        "Equisetum_palustre", "Equisetum_sylvaticum", "Botrychium_lunaria", "Cycas_revoluta", "Lycopodium_annotinum",
        "Selaginella_selaginoides", "Pteridium_aquilinum", "Polystichum_chilense", "Athyrium_brevifrons", "Blechnum_novae-zelandiae",
        "Huperzia_selago"
    ))

# save file

write.table(Core_combined_meta, file = "Weigelt_et_al_2021_Main.PCA.Matrix.csv", sep = ";")


###############################################################################################################################
# Individual PES data
###############################################################################################################################

rm(list = ls())

# load data

Indi_PES <- read.csv("Indi_PES.csv", header = T, sep = ";")

FungalRoot_db <- read.csv("FungalRoot_database_17_06_2020.csv", sep = ";", header = T, na.strings = c("", "NA")) # Fungal root database
nodDB_20_10_2020 <- read.csv("nodDB_20_10_2020.csv", header = T, sep = ";", na.strings = c("", "NA")) # N-fixation database

### I) Processing data ###############################################################################################

# remove columns with no information

Indi_PES <- Indi_PES %>%
    dplyr::select(Row.ID, Reference, growth_conditions, full_species, Woodiness, Root.Entity, LMA, Leaf_N, SRL, RTD, Root_N, Root_diam)

### II) Correct data #################################################################################################

# filter for RTD > 1

Indi_PES <- Indi_PES[-which(Indi_PES$RTD > 1), ]

# reshape data

Indi_core_reshaped <- melt(Indi_PES, id.vars = c("Row.ID", "growth_conditions", "full_species", "Reference", "Woodiness", "Root.Entity"))
colnames(Indi_core_reshaped)[7] <- "TraitName"
colnames(Indi_core_reshaped)[8] <- "StdValue"

# calculate error risk

speciesIndi <- Indi_core_reshaped[which(!grepl("^\\w+$", Indi_core_reshaped$full_species)), ]

### Error risk calculation ###

speciesIndi <- speciesIndi[!is.na(speciesIndi$StdValue), ]

# not normally distirbuted data

speciesIndilog <- speciesIndi %>%
    dplyr::select(Row.ID, full_species, TraitName, StdValue, growth_conditions, Woodiness, Root.Entity, Reference) %>%
    group_by(full_species, TraitName) %>%
    dplyr::filter(TraitName %in% c("LMA", "RTD", "Root_diam", "Root_N", "SRL")) %>%
    mutate(errorRiskEntries = n()) %>%
    mutate(StdValuelog2 = log2(StdValue)) %>%
    mutate(meanSpp = mean(StdValuelog2), sdSpp = sd(StdValuelog2)) %>%
    group_by(TraitName) %>%
    mutate(SDSppAvg = mean(sdSpp, na.rm = T)) %>%
    mutate(errorRisk = ((meanSpp - StdValuelog2) / SDSppAvg)) %>%
    dplyr::select(Row.ID, full_species, TraitName, StdValue, errorRiskEntries, errorRisk, growth_conditions, Woodiness, Root.Entity, Reference)

speciesIndilog$StdValuelog2 <- NULL

# normally distributed data

speciesIndinorm <- speciesIndi %>%
    dplyr::select(Row.ID, full_species, TraitName, StdValue, growth_conditions, Woodiness, Root.Entity, Reference) %>%
    group_by(full_species, TraitName) %>%
    dplyr::filter(TraitName %in% c("Leaf_N")) %>%
    mutate(errorRiskEntries = n()) %>%
    mutate(meanSpp = mean(StdValue), sdSpp = sd(StdValue)) %>%
    group_by(TraitName) %>%
    mutate(SDSppAvg = mean(sdSpp, na.rm = T)) %>%
    mutate(errorRisk = ((meanSpp - StdValue) / SDSppAvg)) %>%
    dplyr::select(Row.ID, full_species, TraitName, StdValue, errorRiskEntries, errorRisk, growth_conditions, Woodiness, Root.Entity, Reference)


speciesRisk <- rbind(speciesIndilog, speciesIndinorm)


# log and z transformed core traits

Above_coreTraits_trans <- speciesRisk %>%
    dplyr::filter(between(errorRisk, -4, 4)) %>%
    dplyr::select(Row.ID, full_species, TraitName, StdValue, errorRiskEntries, errorRisk, growth_conditions, Woodiness, Root.Entity, Reference) %>%
    dplyr::filter(Root.Entity == "FR") %>%
    group_by(TraitName) %>%
    mutate(Log_StdValue = log(StdValue)) %>%
    mutate(scaled_LogStdValue = scale(Log_StdValue))


# add a unique ID for reshape process

Above_coreTraits_trans$Unique_ID <- seq.int(nrow(Above_coreTraits_trans))

# reshape data

Above_trans_reshaped <- dcast(Above_coreTraits_trans, Row.ID + full_species +
    growth_conditions + Woodiness + Reference +
    Root.Entity ~ TraitName, value.var = "scaled_LogStdValue")

# remove NA's from the dataset (755 spec.)

Indi_PCA_final <- Above_trans_reshaped
Indi_PCA_final <- Indi_PCA_final[complete.cases(Indi_PCA_final[, c(7:12)]), ]

### III) Calculate residuals (BLUPs) #########################################################################################

# calculate residuals for LMA

model1 <- lmer(LMA ~ growth_conditions +
    (1 | Reference), data = Indi_PCA_final[complete.cases(Indi_PCA_final[, c(7, 5, 3)]), ])

Indi_PCA_final$LMA_corrected <- rep(NA, nrow(Indi_PCA_final))
Indi_PCA_final$LMA_corrected[complete.cases(Indi_PCA_final[, c(7, 5, 3)])] <- residuals(model1)

# calculate residuals for Leaf_N

model2 <- lmer(Leaf_N ~ growth_conditions +
    (1 | Reference), data = Indi_PCA_final[complete.cases(Indi_PCA_final[, c(8, 5, 3)]), ])

Indi_PCA_final$Leaf_N_corrected <- rep(NA, nrow(Indi_PCA_final))
Indi_PCA_final$Leaf_N_corrected[complete.cases(Indi_PCA_final[, c(8, 5, 3)])] <- residuals(model2)

# calculate residuals for Root_N

model3 <- lmer(Root_N ~ growth_conditions +
    (1 | Reference), data = Indi_PCA_final[complete.cases(Indi_PCA_final[, c(11, 5, 3)]), ])

Indi_PCA_final$Root_N_corrected <- rep(NA, nrow(Indi_PCA_final))
Indi_PCA_final$Root_N_corrected[complete.cases(Indi_PCA_final[, c(11, 5, 3)])] <- residuals(model3)

# calculate residuals for SRL

model4 <- lmer(SRL ~ growth_conditions +
    (1 | Reference), data = Indi_PCA_final[complete.cases(Indi_PCA_final[, c(9, 5, 3)]), ])

Indi_PCA_final$SRL_corrected <- rep(NA, nrow(Indi_PCA_final))
Indi_PCA_final$SRL_corrected[complete.cases(Indi_PCA_final[, c(9, 5, 3)])] <- residuals(model4)

# calculate residuals for RTD

model5 <- lmer(RTD ~ growth_conditions +
    (1 | Reference), data = Indi_PCA_final[complete.cases(Indi_PCA_final[, c(10, 5, 3)]), ])

Indi_PCA_final$RTD_corrected <- rep(NA, nrow(Indi_PCA_final))
Indi_PCA_final$RTD_corrected[complete.cases(Indi_PCA_final[, c(10, 5, 3)])] <- residuals(model5)

# calculate residuals for Root_diam

model6 <- lmer(Root_diam ~ growth_conditions +
    (1 | Reference), data = Indi_PCA_final[complete.cases(Indi_PCA_final[, c(12, 5, 3)]), ])

Indi_PCA_final$Root_diam_corrected <- rep(NA, nrow(Indi_PCA_final))
Indi_PCA_final$Root_diam_corrected[complete.cases(Indi_PCA_final[, c(12, 5, 3)])] <- residuals(model6)

# select only corrected trait values

Indi_PCA <- Indi_PCA_final[, c(2, 4, 13:18)]

# rename colnames

colnames(Indi_PCA)[1] <- "Species"
colnames(Indi_PCA)[2] <- "Woodiness"
colnames(Indi_PCA)[3] <- "LMA"
colnames(Indi_PCA)[4] <- "LN"
colnames(Indi_PCA)[5] <- "RN"
colnames(Indi_PCA)[6] <- "SRL"
colnames(Indi_PCA)[7] <- "RTD"
colnames(Indi_PCA)[8] <- "D"

# add Mycorrhizal meta information

Indi_PCA$Genus <- word(Indi_PCA$Species, start = 1, end = 1)
Indi_PCA <- merge(Indi_PCA, FungalRoot_db, by = "Genus", all.x = TRUE)
Indi_PCA$Genus <- NULL

Indi_PCA$Mycorrhizal.type[Indi_PCA$Mycorrhizal.type == "EcM"] <- "EM"
Indi_PCA$Mycorrhizal.type[Indi_PCA$Mycorrhizal.type == "EcM-AM"] <- "EM+AM"
Indi_PCA$Mycorrhizal.type[Indi_PCA$Mycorrhizal.type == "NM-AM"] <- "NM"
Indi_PCA$Mycorrhizal.type[is.na(Indi_PCA$Mycorrhizal.type)] <- "unknown"
Indi_PCA$Mycorrhizal.type[Indi_PCA$Mycorrhizal.type == "NM-AM, rarely EcM"] <- "NM"
Indi_PCA$Mycorrhizal.type[Indi_PCA$Mycorrhizal.type == "species-specific: AM or rarely EcM-AM or AM"] <- "AM"

Indi_PCA$Mycorrhizal.type[which(Indi_PCA$Species == "Lamiophlomis rotata")] <- "AM"
Indi_PCA$Mycorrhizal.type[which(Indi_PCA$Species == "launaea arborescens")] <- "AM"
Indi_PCA$Mycorrhizal.type[which(Indi_PCA$Species == "Mezzettiopsis creaghii")] <- "AM"
Indi_PCA$Mycorrhizal.type[which(Indi_PCA$Species == "Saussvrea tibetica")] <- "AM"
Indi_PCA$Mycorrhizal.type[which(Indi_PCA$Species == "Staurachanthus genistoides")] <- "AM"
Indi_PCA$Mycorrhizal.type[which(Indi_PCA$Species == "Melissitus ruthenica")] <- "AM"

Indi_PCA$Species <- gsub(" ", "_", Indi_PCA$Species)

# add N-Fixation meta information

Indi_PCA$Species <- gsub("_", " ", Indi_PCA$Species)
Indi_PCA$genus <- word(Indi_PCA$Species, start = 1, end = 1)
Indi_PCA <- merge(Indi_PCA, nodDB_20_10_2020[, c(4, 5)], by = "genus", all.x = T)
Indi_PCA$N_fixation[Indi_PCA$Consensus.estimate == "Rhizobia"] <- "N-fixing"
Indi_PCA$N_fixation[Indi_PCA$N_fixation == "unknown"] <- "Non-N-fixing"
Indi_PCA$genus <- NULL
Indi_PCA$Consensus.estimate <- NULL
Indi_PCA$Species <- gsub(" ", "_", Indi_PCA$Species)

Indi_PCA$N_fixation[is.na(Indi_PCA$N_fixation)] <- "None"

### IV) Species selection from individual PES data #################################################################################################

# Individuals were selected either randomly (if there were only 2 individuals per species) or using the 'clhs' function of the 'clhs' package
# (if there where more than 2 individuals per species; Roudier et al. 2011, Version 0.7.3) which uses a stratified random procedure and provides
# an efficient way of sampling variables from their multivariate distributions. Thus, when having more than 2 individuals per species,
# an individual was selected based on the proximity of it's individual trait values relative to the mean species trait values.

rm(list = setdiff(ls(), "Indi_PCA"))

# Create data frames with: 1. only one individual per species, 2. two individuals per species, and 3. more than two individuals per species

IndiSpecies_dup <- unique(Indi_PCA$Species[duplicated(Indi_PCA$Species)])

Indi_PCA_dup <- Indi_PCA %>%
    dplyr::filter(Species %in% IndiSpecies_dup)

For_clhs <- Indi_PCA_dup %>%
    group_by(Species) %>%
    dplyr::filter(n() > 2)

IDs1 <- unique(For_clhs$Species)

Frequ_2 <- Indi_PCA_dup %>%
    group_by(Species) %>%
    dplyr::filter(n() < 3)

IDs2 <- unique(Frequ_2$Species)

Indi_PCA_single <- Indi_PCA %>%
    group_by(Species) %>%
    dplyr::filter(n() < 2)
Indi_PCA_single <- data.frame(Indi_PCA_single)

# cLHS approach for species with more than 2 individuals

final.list <- list()

pb <- txtProgressBar(min = 0, max = length(IDs1), style = 3)
for (i in 1:length(IDs1)) {
    temp <- For_clhs[For_clhs$Species == IDs1[i], ]
    set.seed(001)
    res <- clhs(temp, size = 1, progress = FALSE, iter = 1000)
    print(i)
    final.list[[i]] <- temp[res, ]
    setTxtProgressBar(pb, i)
}

close(pb)

final <- do.call(rbind, final.list)
final <- data.frame(final)

# draw random samples from species with only two individuals

final.list2 <- list()

pb <- txtProgressBar(min = 0, max = length(IDs2), style = 3)
for (i in 1:length(IDs2)) {
    temp2 <- Frequ_2[Frequ_2$Species == IDs2[i], ]
    set.seed(001)
    res2 <- temp2[sample(nrow(temp2), 1), ]
    print(i)
    final.list2[[i]] <- unlist(res2)
    setTxtProgressBar(pb, i)
}

close(pb)

final2 <- do.call(rbind, final.list2)

list(final2)
final2 <- data.frame(final2)
final2 <- final2 %>% drop_na()

# rbind data frames

Indi_PCA_sel <- rbind(Indi_PCA_single, final, final2)

# reorder data set

Indi_PCA <- Indi_PCA_sel
Indi_PCA <- Indi_PCA %>% dplyr::select(Species, LMA, LN, RN, SRL, RTD, D, Woodiness, Mycorrhizal.type, N_fixation)

### V) Perform PCA #################################################################################################

# Correct Species names (typo, synonym)

Indi_PCA$Species <- gsub("Parakmeria_lotungensis", "Magnolia_lotungensis", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Parakmeria_yunnanensis", "Magnolia_yunnanensis", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Lamiophlomis_rotata", "Phlomoides_rotata", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Paramichelia_baillonii", "Magnolia_baillonii", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Saussvrea_tibetica", "Saussurea_tibetica", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Staurachanthus_genistoides", "Stauracanthus_genistoides", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Aporusa_dioica", "Aporosa_octandra", Indi_PCA$Species)
Indi_PCA$Species <- gsub("launaea_arborescens", "Launaea_arborescens", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Mezzettiopsis_creaghii", "Orophea_creaghii", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Cyclobalanopsis_bambusaefolia", "Quercus_myrsinifolia", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Cyclobalanopsis_chungii", "Quercus_delavayi", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Cyclobalanopsis_patelliformis", "Quercus_patelliformis", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Kochia_prostrata", "Bassia_prostrata", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Manglietia_dandyi", "Magnolia_dandyi", Indi_PCA$Species)
Indi_PCA$Species <- gsub("Melissitus_ruthenica", "Medicago_ruthenica", Indi_PCA$Species)

# extract phylogenetic information on family names for all species in the PCA data set from ncbi (see https://www.ncbi.nlm.nih.gov/)
# this approach was recommended from the author of the package "brranching" after reporting errors related to the function "phylomatic" that is usally used to
# extract phylogeny (see GitHub https://github.com/ropensci/brranching/issues/42)

set_entrez_key("YOUR_KEY") # enter your API key
Sys.getenv("ENTREZ_KEY")
# API key from ncbi (see https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)

Indi_names_phyl <- phylomatic_names(Indi_PCA$Species, db = "ncbi")

Indi_PhylomaticNames <- data.frame(Indi_names_phyl)
colnames(Indi_PhylomaticNames)[1] <- "Species"
Indi_PhylomaticNames$ID <- seq.int(nrow(Indi_PhylomaticNames))
Indi_PCA$ID <- seq.int(nrow(Indi_PCA))
colnames(Indi_PhylomaticNames)[1] <- "Species_ncbi"
Indi_PCA_2 <- merge(Indi_PCA, Indi_PhylomaticNames, by = "ID")
Indi_PCA_2$ID <- NULL
Indi_PCA <- Indi_PCA_2

# change trait columns to numeric

Indi_PCA[c(2:7)] <- sapply(Indi_PCA[c(2:7)], as.numeric)


# save file

write.table(Indi_PCA, file = "Weigelt_et_al_2021_Individal.PCA.Matrix.csv", sep = ";")


###############################################################################################################################
# Literature list
###############################################################################################################################

# load data

TRYFull_Indi_ErrorRisk <- read.csv("TRYFull_Indi_ErrorRisk.csv", header = T, sep = ";", na.strings = c("", "NA"))

### Literature list for all trait data which was extracted from single publications #############################

Literature_list_TRY_singlePub <- TRYFull_Indi_ErrorRisk %>%
    dplyr::filter(Data_source == "Single publication") %>%
    dplyr::filter(TraitName %in% c(
        "Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)",
        "Leaf nitrogen (N) content per leaf dry mass",
        "Leaf mass per area (LMA)",
        "Leaf dry mass per leaf fresh mass (leaf dry matter content, LDMC)",
        "Leaf density (leaf tissue density, leaf dry mass per leaf volume)",
        "Leaf thickness",
        "Leaf phosphorus (P) content per leaf dry mass",
        "Plant height vegetative",
        "Leaf carbon (C) content per leaf dry mass",
        "Leaf_Lignin"
    )) %>%
    dplyr::filter(!Data_source == "TRY") %>%
    dplyr::select(Reference, Data_source) %>%
    distinct(Reference, .keep_all = TRUE)

write.csv(Literature_list_TRY_singlePub, "References_Additional_Publications.csv")

### Literature list for all individual based trait data #############################

Literature_list_TRY_INDI_PES <- TRYFull_Indi_ErrorRisk %>%
    dplyr::filter(!is.na(Individual_data)) %>%
    dplyr::filter(TraitName %in% c(
        "Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)",
        "Leaf nitrogen (N) content per leaf dry mass",
        "Leaf mass per area (LMA)",
        "Leaf dry mass per leaf fresh mass (leaf dry matter content, LDMC)",
        "Leaf density (leaf tissue density, leaf dry mass per leaf volume)",
        "Leaf thickness",
        "Leaf phosphorus (P) content per leaf dry mass",
        "Plant height vegetative",
        "Leaf carbon (C) content per leaf dry mass",
        "Leaf_Lignin"
    )) %>%
    dplyr::select(Reference) %>%
    distinct(Reference)

write.csv(Literature_list_TRY_INDI_PES, "References_Individual_based_Trait_data.csv")
