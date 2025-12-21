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

#--------------------
# ROOT DIAMETER
#--------------------


# load("./rdata/OU_CD.RData") # rate.cat=2, null.model=FALSE - WHAT'S THE POINT OF THIS???
load("./rdata/OU_CIDp.RData") # rate.cat=2, null.model=TRUE
load("./rdata/OU_CD_.RData") # rate.cat=1, null.model=FALSE

# find out the AIC and AICc of all the models

# rate.cat=1, null.model=FALSE
models_CD_RD <- list(EROUM=ER_OUM_RD_CD, EROUMA=ER_OUMA_RD_CD, EROUMV=ER_OUMV_RD_CD, EROUMVA=ER_OUMVA_RD_CD, ARDOUM=ARD_OUM_RD_CD,
                  ARDOUMA=ARD_OUMA_RD_CD, ARDOUMV=ARD_OUMV_RD_CD, ARDOUMVA=ARD_OUMVA_RD_CD, SYMOUM=SYM_OUM_RD_CD, SYMOUMA=SYM_OUMA_RD_CD,
                  SYMOUMV=SYM_OUMV_RD_CD, SYMOUMVA=SYM_OUMVA_RD_CD)

# rate.cat=2, null.model=TRUE
models_CIDp_RD <- list(EROUM=ER_OUM_RD_CIDp, EROUMA=ER_OUMA_RD_CIDp, EROUMV=ER_OUMV_RD_CIDp, EROUMVA=ER_OUMVA_RD_CIDp,
                    ARDOUM=ARD_OUM_RD_CIDp, ARDOUMA=ARD_OUMA_RD_CIDp, ARDOUMV=ARD_OUMV_RD_CIDp, ARDOUMVA=ARD_OUMVA_RD_CIDp,
                    SYMOUM=SYM_OUM_RD_CIDp, SYMOUMA=SYM_OUMA_RD_CIDp, SYMOUMV=SYM_OUMV_RD_CIDp, SYMOUMVA=SYM_OUMVA_RD_CIDp)


# lapply(models, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))
lapply(models_CIDp_RD, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))
lapply(models_CD_RD, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))

# model averages
# type - one of AIC, BIC, or AICc for use during evaluation of relative model fit.
# AICc is the best option for datasets with a few number of species.
# force - a boolean indicating whether to force potentially failed model fits to be included in the model averaging.

avg_models_CIDp_RD <- OUwie::getModelAvgParams(models_CIDp_RD, type = "AICc", force = FALSE)
avg_models_CD_RD <- OUwie::getModelAvgParams(models_CD_RD, type = "AICc", force = FALSE)

# look up to see what the BIC stuff is about
# https://www.rdocumentation.org/packages/AICcmodavg/versions/2.3-4/topics/bictabCustom
# AIC vs BIC
# https://fiveable.me/bayesian-statistics/unit-11/bayesian-information-criterion/study-guide/o3iS2biLgz7mcyuv

plot_df <- reshape2::melt(avg_models_CIDp_RD)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_RD_CIDp.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)


plot_df <- reshape2::melt(avg_models_CD_RD)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_RD_CD.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)


avg_models_CD_RD[, c("alpha", "sigma.sq", "theta", "tip_state")] |> split(~tip_state)


# there's already a function named stderr in base R ???
stderr_ <- function(df) { lapply(X=df, FUN=function(column) {sd(column) / sqrt(length(column))}) |> unlist() }

avg_models_CD_RD |> split(~tip_state)
avg_models_CD_RD |> split(~tip_state) |> lapply(function(df) { colMeans(df[, c("rates", "alpha", "sigma.sq", "theta")]) })
avg_models_CD_RD |> split(~tip_state) |> lapply(function(df) { stderr_(df[, c("rates", "alpha", "sigma.sq", "theta")]) })


#--------------------------
# SPECIFIC ROOT LENGTH
#--------------------------

load("./rdata/OU_SRL_CD.RData")
load("./rdata/OU_SRL_CIDp.RData")


models_CD_SRL <- lit(ER_OUM_SRL_CD, ER_OUMA_SRL_CD, ER_OUMV_SRL_CD, ER_OUMVA_SRL_CD, ARD_OUM_SRL_CD, ARD_OUMA_SRL_CD, ARD_OUMV_SRL_CD,
                   ARD_OUMVA_SRL_CD, SYM_OUM_SRL_CD, SYM_OUMA_SRL_CD, SYM_OUMV_SRL_CD, SYM_OUMVA_SRL_CD)

models_CIDp_SRL <- list(ER_OUM_SRL_CIDp, ER_OUMA_SRL_CIDp, ER_OUMV_SRL_CIDp, ER_OUMVA_SRL_CIDp, ARD_OUM_SRL_CIDp, ARD_OUMA_SRL_CIDp,
                     ARD_OUMV_SRL_CIDp, ARD_OUMVA_SRL_CIDp, SYM_OUM_SRL_CIDp, SYM_OUMA_SRL_CIDp, SYM_OUMV_SRL_CIDp, SYM_OUMVA_SRL_CIDp)
