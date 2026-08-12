# https://paulbuerkner.com/brms/articles/brms_multivariate.html
# https://paulbuerkner.com/brms/articles/brms_phylogenetics.html

library("brms")
library("loo")
library("bayesplot")
library("ggplot2")
library("gridExtra")
library("ggpubr")

options(mc.cores = 10)

# model as defined in the manuscript
# M1 - (RD,SRL,RTD) ~ states + (1│gr(species, cov=vcv_phy)) + (1|species)

# M2 - (RD,SRL,RTD) ~ states + (1|species)
M2 <- readRDS("../data/chapter2/rdata/hie-general/brms/brms_taxa.Rds")

# M3 - (RD,SRL,RTD) ~ states + (1│p|gr(species, cov=vcv_phy)) + (1|q|species)

# M4 - (RD,SRL,RTD) ~ states + (1|q|species)
M4 <- readRDS("../data/chapter2/rdata/hie-general/brms/brms_taxa_corr.Rds")


# all of these models have been fitted with 8 chains parallelized across 8 CPU cores on an Ubuntu cloud VM

# the difference between model 1 and 3 is the cross trait correlations
# these are non-phylogenetic models
# posterior prediction tests to evaluate model fits
# looks good :)

p1 <- brms::pp_check(M2, resp = "F00679")
p2 <- brms::pp_check(M2, resp = "F00727")
p3 <- brms::pp_check(M2, resp = "F00709")

combined <- ggpubr::ggarrange(p1, p2, p3, ncol = 3, labels = c("RD", "SRL", "RTD"))
ggplot2::ggsave("../plots/pp_check_M2.png", device = "png", width = 30, height = 10, units = "in", dpi = 500, bg = "white")




# this looks good too :)
p1 <- brms::pp_check(M4, resp = "F00679")
p2 <- brms::pp_check(M4, resp = "F00727")
p3 <- brms::pp_check(M4, resp = "F00709")

combined <- ggpubr::ggarrange(p1, p2, p3, ncol = 3, labels = c("RD", "SRL", "RTD"))
ggplot2::ggsave("../plots/pp_check_M4.png", device = "png", width = 30, height = 10, units = "in", dpi = 500, bg = "white")


brms::bayes_R2(mrpmm_1)
#          Estimate   Est.Error      Q2.5     Q97.5
# R2F00727 0.5815352 0.007522336 0.5663746 0.5959299
# R2F00679 0.5202709 0.008479031 0.5032777 0.5365578
# R2F00709 0.6504595 0.006031641 0.6383486 0.6619273

brms::bayes_R2(mrpmm_3)
#           Estimate   Est.Error      Q2.5     Q97.5
# R2F00727 0.3889979 0.013610280 0.3619941 0.4154351
# R2F00679 0.3723803 0.012788278 0.3472163 0.3970243
# R2F00709 0.5885554 0.008566066 0.5711901 0.6050101

# the model that allowed cross trait correlation (model 1) was able to explain more variation in all traits compared to the model that did not
# based on the Bayesian R2 values

# to compare model 1 and model 3
mrpmm_1 <- brms::add_criterion(mrpmm_1, criterion = "loo", moment_match = TRUE)
mrpmm_3 <- brms::add_criterion(mrpmm_3, criterion = "loo", moment_match = TRUE)

loo::loo(mrpmm_1, mrpmm_3)
