#!/usr/bin/env Rscript

# to be run on the VM
library("ape")
library("phytools")
library("readxl")

phylo <- ape::read.tree("./../data/chapter2/uphylomaker/FRED4_1301.tre")
if(!ape::is.binary(phylo)) phylo <- ape::multi2di(phylo, random = FALSE) # if not binary make it binary
phylo$edge.length[phylo$edge.length <= 0] <- rnorm(n = sum(phylo$edge.length == 0), mean = 1e-6, sd = 1e-8) # replace the 0 length branches with small values
if (!ape::is.ultrametric(phylo)) phylo <- phytools::force.ultrametric(tree = phylo, method = "extend") # if not ultrametric make it ultrametric
stopifnot(ape::is.binary(phylo) && ape::is.ultrametric(phylo))

fred4 <- read.csv("./../data/chapter2/FRED/subsets/continuous_raw.csv") # data with repeated records for species
fred4[, 5:7] <- scale(log(fred4[, 5:7])) # log transform and standardize the continuous traits

RD <- phytools::phylANOVA(tree = phylo, y = setNames(fred4$F00679, fred4$binominal), x = setNames(fred4$binominal, fred4$binominal))
SRL <- phytools::phylANOVA(tree = phylo, y = setNames(fred4$F00727, fred4$binominal), x = setNames(fred4$binominal, fred4$binominal))
RTD <- phytools::phylANOVA(tree = phylo, y = setNames(fred4$F00709, fred4$binominal), x = setNames(fred4$binominal, fred4$binominal))

save(RD, SRL, RTD, file = "./ScratchData/ANOVA.RData")
