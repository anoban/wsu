# https://paulbuerkner.com/brms/articles/brms_multivariate.html
# https://paulbuerkner.com/brms/articles/brms_phylogenetics.html

library("brms")
library("loo")
library("bayesplot")
library("ggplot2")
library("ggpubr")
library("rstan")
library("cmdstanr")
library("posterior")

options(mc.cores = 10)

#--------------------------------------
# model as defined in the manuscript
#--------------------------------------

# M1 - (RD,SRL,RTD) ~ state + (1│gr(binominal, cov=vcv_phy)) + (1|taxa)
M1 <- readRDS("../data/chapter2/rdata/hie-general/brms/brms_phylo.Rds")

# M2 - (RD,SRL,RTD) ~ state + (1|taxa)
M2 <- readRDS("../data/chapter2/rdata/hie-general/brms/brms_taxa.Rds")

# M3 - (RD,SRL,RTD) ~ state + (1│p|gr(binominal, cov=vcv_phy)) + (1|q|taxa)


# M4 - (RD,SRL,RTD) ~ state + (1|q|taxa)
M4 <- readRDS("../data/chapter2/rdata/hie-general/brms/brms_taxa_corr.Rds")

# all of these models have been fitted with 8 chains parallelized across 8 CPU cores on an Ubuntu cloud VM
# the difference between M2 and M4 is the cross trait correlations


#--------------------
# pp_check()
#--------------------

p1 <- brms::pp_check(M2, resp = "F00679")
p2 <- brms::pp_check(M2, resp = "F00727")
p3 <- brms::pp_check(M2, resp = "F00709")

combined <- ggpubr::ggarrange(p1, p2, p3, ncol = 3, labels = c("RD", "SRL", "RTD"))
ggplot2::ggsave("../plots/pp_check_M2.png", plot = combined, device = "png", width = 30, height = 10, units = "in", dpi = 500, bg = "white")


# this looks good too :)
p1 <- brms::pp_check(M4, resp = "F00679")
p2 <- brms::pp_check(M4, resp = "F00727")
p3 <- brms::pp_check(M4, resp = "F00709")

combined <- ggpubr::ggarrange(p1, p2, p3, ncol = 3, labels = c("RD", "SRL", "RTD"))
ggplot2::ggsave("../plots/pp_check_M4.png", plot = combined, device = "png", width = 30, height = 10, units = "in", dpi = 500, bg = "white")


p1 <- brms::pp_check(M1, resp = "F00679")
p2 <- brms::pp_check(M1, resp = "F00727")
p3 <- brms::pp_check(M1, resp = "F00709")

combined <- ggpubr::ggarrange(p1, p2, p3, ncol = 3, labels = c("RD", "SRL", "RTD"))
ggplot2::ggsave("../plots/pp_check_M1.png", plot = combined, device = "png", width = 30, height = 10, units = "in", dpi = 500, bg = "white")


#--------------------
# plot()
#--------------------

plots_4 <- plot(M1, ask = FALSE) # without ask = FALSE, it will prompt you tu hit enter to view each one of the 4 plots
p4 <- ggpubr::ggarrange(plotlist = plots_4)
ggplot2::ggsave("../plots/M1.png", plot = p4, device = "png", width = 32, height = 18, units = "in", dpi = 600, bg = "white")

plots_4 <- plot(M2, ask = FALSE)
p4 <- ggpubr::ggarrange(plotlist = plots_4)
ggplot2::ggsave("../plots/M2.png", plot = p4, device = "png", width = 32, height = 18, units = "in", dpi = 600, bg = "white")

plots_4 <- plot(M4, ask = FALSE)
p4 <- ggpubr::ggarrange(plotlist = plots_4)
ggplot2::ggsave("../plots/M4.png", plot = p4, device = "png", width = 32, height = 18, units = "in", dpi = 600, bg = "white")


#-----------------------
# conditional_effects()
#-----------------------

# to get the posterior samples from a fitted brms model - https://github.com/santiagobarreda/bmmb/blob/main/R/get_samples.R
# in our case the conditional_effects will always return three plots because we have three different response variables
# for all our fits brms says the only valid effect is the fixed effect -> state

coneffs <- plot(brms::conditional_effects(M4), ask = FALSE, points = TRUE, plot = FALSE)
ceffgrid <- ggpubr::ggarrange(coneffs[[1]], coneffs[[2]], coneffs[[3]], nrow = 1, ncol = 3, labels = c("SRL", "RD", "RTD"))
ggplot2::ggsave("../plots/condeffs_M4.png", plot = ceffgrid, device = "png", width = 15, height = 6, units = "in", dpi = 600, bg = "white")


coneffs <- plot(brms::conditional_effects(M2), ask = FALSE, points = TRUE, plot = FALSE)
ceffgrid <- ggpubr::ggarrange(coneffs[[1]], coneffs[[2]], coneffs[[3]], nrow = 1, ncol = 3, labels = c("SRL", "RD", "RTD"))
ggplot2::ggsave("../plots/condeffs_M2.png", plot = ceffgrid, device = "png", width = 15, height = 6, units = "in", dpi = 600, bg = "white")


coneffs <- plot(brms::conditional_effects(M1), ask = FALSE, points = TRUE, plot = FALSE)
ceffgrid <- ggpubr::ggarrange(coneffs[[1]], coneffs[[2]], coneffs[[3]], nrow = 1, ncol = 3, labels = c("SRL", "RD", "RTD"))
ggplot2::ggsave("../plots/condeffs_M1.png", plot = ceffgrid, device = "png", width = 15, height = 6, units = "in", dpi = 600, bg = "white")



#-----------------
# Bayesian R2
#-----------------

brms::bayes_R2(M2)
#           Estimate   Est.Error      Q2.5     Q97.5
# R2F00727 0.3891677 0.013460267 0.3622644 0.4151264
# R2F00679 0.3725701 0.012700322 0.3474079 0.3973264
# R2F00709 0.5887040 0.008510461 0.5715858 0.6052387

brms::bayes_R2(M4)
#           Estimate   Est.Error      Q2.5     Q97.5
# R2F00727 0.5816603 0.007501405 0.5665312 0.5959486
# R2F00679 0.5202500 0.008482579 0.5031848 0.5364784
# R2F00709 0.6504895 0.006012502 0.6384767 0.6619459

# the model that allowed cross trait correlation (M4) was able to explain more variation in all traits compared to the model that did not (M2)
# based on the Bayesian R2 values

brms::bayes_R2(M1)
#           Estimate   Est.Error      Q2.5     Q97.5
# R2F00727 0.4601284 0.009358918 0.4414945 0.4784119
# R2F00679 0.4022071 0.010203081 0.3822346 0.4218134
# R2F00709 0.6071736 0.006945364 0.5930534 0.6203351




#-------
# loo
#-------

# to compare M2 and M4
mrpmm_1 <- brms::add_criterion(M2, criterion = "loo", moment_match = TRUE)
mrpmm_3 <- brms::add_criterion(M4, criterion = "loo", moment_match = TRUE)

loo::loo(mrpmm_1, mrpmm_3)
