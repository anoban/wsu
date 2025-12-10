suppressPackageStartupMessages({
    library("ape")
    library("phytools")
    library("nlme")
    library("corHMM")
    library("geiger")
    library("mkcor")
    library("OUwie")
    library("reshape2")
    library("ggplot2")
})

load("./OU_CD.RData") # CD?? models
load("./OU_CIDp.RData") # CID+ models

# find out the AIC and AICc of all the models
# CD models
models <- list(EROUM=ER_OUM_RD, EROUMA=ER_OUMA_RD, EROUMV=ER_OUMV_RD, EROUMVA=ER_OUMVA_RD,
               ARDOUM=ARD_OUM_RD, ARDOUMA=ARD_OUMA_RD, ARDOUMV=ARD_OUMV_RD, ARDOUMVA=ARD_OUMVA_RD,
               SYMOUM=SYM_OUM_RD, SYMOUMA=SYM_OUMA_RD, SYMOUMV=SYM_OUMV_RD, SYMOUMVA=SYM_OUMVA_RD)
# CID+ models
models_CIDp <- list(EROUM=ER_OUM_RD_CIDp, EROUMA=ER_OUMA_RD_CIDp, EROUMV=ER_OUMV_RD_CIDp, EROUMVA=ER_OUMVA_RD_CIDp,
                    ARDOUM=ARD_OUM_RD_CIDp, ARDOUMA=ARD_OUMA_RD_CIDp, ARDOUMV=ARD_OUMV_RD_CIDp, ARDOUMVA=ARD_OUMVA_RD_CIDp,
                    SYMOUM=SYM_OUM_RD_CIDp, SYMOUMA=SYM_OUMA_RD_CIDp, SYMOUMV=SYM_OUMV_RD_CIDp, SYMOUMVA=SYM_OUMVA_RD_CIDp)

lapply(models, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame()
lapply(models_CIDp, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame()

# model averages
# type - one of AIC, BIC, or AICc for use during evaluation of relative model fit.
# AICc is the best option for datasets with a few number of species.
# force - a boolean indicating whether to force potentially failed model fits to be included in the model averaging.
avg_models_CD <- OUwie::getModelAvgParams(models, type = "AICc", force = FALSE)
avg_models_CIDp <- OUwie::getModelAvgParams(models_CIDp, type = "AICc", force = FALSE)

# look up to see what the BIC stuff is about
# https://www.rdocumentation.org/packages/AICcmodavg/versions/2.3-4/topics/bictabCustom
# AIC vs BIC
# https://fiveable.me/bayesian-statistics/unit-11/bayesian-information-criterion/study-guide/o3iS2biLgz7mcyuv

# plot the results
plot_df <- reshape2::melt(avg_models_CD)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_IDKmaybeCD.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)


plot_df <- reshape2::melt(avg_models_CIDp)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_CIDp.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)

avg_models_CD[, c("alpha", "sigma.sq", "theta", "tip_state")] |> split(~tip_state)


# follwing gives you the state shift info in the phylogeny
transition_mat <- ER_OUM_RD$hOUwie.dat$StateMats[[1]]
# how to interpret the columns and rows
ER_OUM_RD$hOUwie.dat$ObservedTraits

# rename the R1 to Rn to actual mycorrhizal states
renames <- setNames(ER_OUM_RD$hOUwie.dat$ObservedTraits, nm=paste0("R", names(ER_OUM_RD$hOUwie.dat$ObservedTraits)))
rownames(transition_mat) <- renames
colnames(transition_mat) <- renames

stderr <- function(df) { lapply(X=df, FUN=function(column) {sd(column) / sqrt(length(column))}) |> unlist() }

avg_models_CD |> split(~tip_state)
avg_models_CD |> split(~tip_state) |> lapply(function(df) { colMeans(df[, c("rates", "alpha", "sigma.sq", "theta")]) })
avg_models_CD |> split(~tip_state) |> lapply(function(df) { stderr(df[, c("rates", "alpha", "sigma.sq", "theta")]) })
