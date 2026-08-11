# https://paulbuerkner.com/brms/articles/brms_multivariate.html
# https://paulbuerkner.com/brms/articles/brms_phylogenetics.html

library("brms")
library("loo")
library("bayesplot")
library("ggplot2")
library("gridExtra")

options(mc.cores = 10)

# let's analyze the brms model fits

# (F00679, F00727, F00709) ~ state + (1|q|taxa)
mrpmm_1 <- readRDS("../data/chapter2/rdata/hie-general/brms_model_1.Rds")

# (F00679, F00727, F00709) ~ state + (1|taxa)
mrpmm_3 <- readRDS("../data/chapter2/rdata/hie-general/brms_model_3.Rds")

# (F00679, F00727, F00709) ~ state + (1|gr(binominal, cov = corrmat)) + (1|taxa)
mrpmm_2 <- readRDS("../data/chapter2/rdata/hie-general/brms_model_2.Rds")



# the difference between model 1 and 3 is the cross trait correlations
# these are non-phylogenetic models
# posterior prediction tests to evaluate model fits
# looks good :)
gridExtra::grid.arrange(brms::pp_check(mrpmm_1, resp = "F00679"),
                        brms::pp_check(mrpmm_1, resp = "F00727"),
                        brms::pp_check(mrpmm_1, resp = "F00709"),
                        nrow = 1)

# this looks good too :)
gridExtra::grid.arrange(brms::pp_check(mrpmm_3, resp = "F00679"),
                        brms::pp_check(mrpmm_3, resp = "F00727"),
                        brms::pp_check(mrpmm_3, resp = "F00709"),
                        nrow = 1)

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
