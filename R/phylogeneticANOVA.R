#!/usr/bin/env Rscript

library("ape")
library("phytools")
library("readxl")
library("RRPP")

phylo <- ape::read.tree("./../data/chapter2/uphylomaker/FRED4_1301.tre")
if(!ape::is.binary(phylo)) phylo <- ape::multi2di(phylo, random = FALSE)
phylo$edge.length[phylo$edge.length <= 0] <- rnorm(n = sum(phylo$edge.length == 0), mean = 1e-6, sd = 1e-8)

if (!ape::is.ultrametric(phylo)) phylo <- phytools::force.ultrametric(tree = phylo, method = "extend")
stopifnot(ape::is.binary(phylo) && ape::is.ultrametric(phylo))

fred4_raw <- read.csv("./../data/chapter2/FRED/subsets/continuous_raw.csv")
fred4_raw[, 5:7] <- scale(log(fred4_raw[, 5:7]))
varcov <- ape::vcv.phylo(phylo)
varcov <- varcov[fred4_raw$binominal, fred4_raw$binominal]

rrpp_df <- RRPP::rrpp.data.frame(Y = fred4_raw[, c("F00679", "F00727", "F00709")], taxa = fred4_raw$binominal, Cov = varcov)
tm <- Sys.time()
lmod_aov <- RRPP::lm.rrpp(Y ~ taxa, Cov = varcov, data = rrpp_df, iter = 1000, print.progress = FALSE, parallel = 12)
tm <- Sys.time() - tm

stats::anova(lmod_aov)

saveRDS(lmod_aov, file = "./ScratchData/RRPPanova.Rds")
