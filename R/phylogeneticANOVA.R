#!/usr/bin/env Rscript

library("ape")
library("phytools")
library("readxl")

phylo <- ape::read.tree("./../data/chapter2/uphylomaker/FRED4_1301.tre")
if(!ape::is.binary(phylo)) phylo <- ape::multi2di(phylo, random = FALSE)
phylo$edge.length[phylo$edge.length <= 0] <- rnorm(n = sum(phylo$edge.length == 0), mean = 1e-6, sd = 1e-8)
if (!ape::is.ultrametric(phylo)) phylo <- phytools::force.ultrametric(tree = phylo, method = "extend")
stopifnot(ape::is.binary(phylo) && ape::is.ultrametric(phylo))

fred4 <- read.csv("./../data/chapter2/FRED/subsets/continuous_raw.csv")
fred4[, 5:7] <- scale(log(fred4[, 5:7]))

RD <- phytools::phylANOVA(tree = phylo, y = setNames(fred4$F00679, fred4$binominal), x = setNames(fred4$binominal, fred4$binominal))
SRL <- phytools::phylANOVA(tree = phylo, y = setNames(fred4$F00727, fred4$binominal), x = setNames(fred4$binominal, fred4$binominal))
RTD <- phytools::phylANOVA(tree = phylo, y = setNames(fred4$F00709, fred4$binominal), x = setNames(fred4$binominal, fred4$binominal))

save(RD, SRL, RTD, file = "./ScratchData/ANOVA.RData")
