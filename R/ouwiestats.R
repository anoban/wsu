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


load("./rdata/OU_RD_CD.RData") # rate.cat=1, null.model=FALSE
load("./rdata/OU_RD_CID.RData") # rate.cat=2, null.model=TRUE

# find out the AIC and AICc of all the models

# rate.cat=1, null.model=FALSE
models_CD_RD <- list(EROUM=ER_OUM_RD_CD, EROUMA=ER_OUMA_RD_CD, EROUMV=ER_OUMV_RD_CD, EROUMVA=ER_OUMVA_RD_CD, ARDOUM=ARD_OUM_RD_CD,
                  ARDOUMA=ARD_OUMA_RD_CD, ARDOUMV=ARD_OUMV_RD_CD, ARDOUMVA=ARD_OUMVA_RD_CD, SYMOUM=SYM_OUM_RD_CD, SYMOUMA=SYM_OUMA_RD_CD,
                  SYMOUMV=SYM_OUMV_RD_CD, SYMOUMVA=SYM_OUMVA_RD_CD)

# rate.cat=2, null.model=TRUE
models_CID_RD <- list(EROUM=ER_OUM_RD_CID, EROUMA=ER_OUMA_RD_CID, EROUMV=ER_OUMV_RD_CID, EROUMVA=ER_OUMVA_RD_CID,
                    ARDOUM=ARD_OUM_RD_CID, ARDOUMA=ARD_OUMA_RD_CID, ARDOUMV=ARD_OUMV_RD_CID, ARDOUMVA=ARD_OUMVA_RD_CID,
                    SYMOUM=SYM_OUM_RD_CID, SYMOUMA=SYM_OUMA_RD_CID, SYMOUMV=SYM_OUMV_RD_CID, SYMOUMVA=SYM_OUMVA_RD_CID)


lapply(models_CID_RD, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))
lapply(models_CD_RD, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))

# model averages
# type - one of AIC, BIC, or AICc for use during evaluation of relative model fit.
# AICc (small sample size corrected AIC) is the best option for datasets with a few number of species.
# force - a boolean indicating whether to force potentially failed model fits to be included in the model averaging.

avg_models_CD_RD <- OUwie::getModelAvgParams(models_CD_RD, type = "AICc", force = FALSE)
avg_models_CID_RD <- OUwie::getModelAvgParams(models_CID_RD, type = "AICc", force = FALSE)

# look up to see what the BIC stuff is about
# https://www.rdocumentation.org/packages/AICcmodavg/versions/2.3-4/topics/bictabCustom
# AIC vs BIC
# https://fiveable.me/bayesian-statistics/unit-11/bayesian-information-criterion/study-guide/o3iS2biLgz7mcyuv

plot_df <- reshape2::melt(avg_models_CID_RD)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_RD_CID.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)


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

avg_models_CD_RD |> split(~tip_state) |> lapply(function(df) { colMeans(df[, c("rates", "alpha", "sigma.sq", "theta")]) }) |> as.data.frame()
avg_models_CID_RD |> split(~tip_state) |> lapply(function(df) { colMeans(df[, c("rates", "alpha", "sigma.sq", "theta")]) }) |> as.data.frame()


avg_models_CD_RD |> split(~tip_state) |> lapply(function(df) { stderr_(df[, c("rates", "alpha", "sigma.sq", "theta")]) })



#--------------------------
# SPECIFIC ROOT LENGTH
#--------------------------


load("./rdata/OU_SRL_CD.RData")
load("./rdata/OU_SRL_CID.RData")

models_CD_SRL <- list(EROUM=ER_OUM_SRL_CD, EROUMA=ER_OUMA_SRL_CD, EROUMV=ER_OUMV_SRL_CD, EROUMVA=ER_OUMVA_SRL_CD, ARDOUM=ARD_OUM_SRL_CD,
                     ARDOUMA=ARD_OUMA_SRL_CD, ARDOUMV=ARD_OUMV_SRL_CD, ARDOUMVA=ARD_OUMVA_SRL_CD, SYMOUM=SYM_OUM_SRL_CD, SYMOUMA=SYM_OUMA_SRL_CD,
                     SYMOUMV=SYM_OUMV_SRL_CD, SYMOUMVA=SYM_OUMVA_SRL_CD)

models_CID_SRL <- list(EROUM=ER_OUM_SRL_CID, EROUMA=ER_OUMA_SRL_CID, EROUMV=ER_OUMV_SRL_CID, EROUMVA=ER_OUMVA_SRL_CID, ARDOUM=ARD_OUM_SRL_CID,
                        ARDOUMA=ARD_OUMA_SRL_CID, ARDOUMV=ARD_OUMV_SRL_CID, ARDOUMVA=ARD_OUMVA_SRL_CID, SYMOUM=SYM_OUM_SRL_CID, SYMOUMA=SYM_OUMA_SRL_CID,
                        SYM_OUMV=SYM_OUMV_SRL_CID, SYMOUMVA=SYM_OUMVA_SRL_CID)

lapply(models_CID_SRL, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))
lapply(models_CD_SRL, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))

avg_models_CID_SRL <- OUwie::getModelAvgParams(models_CID_SRL, type = "AICc", force = FALSE)
avg_models_CD_SRL <- OUwie::getModelAvgParams(models_CD_SRL, type = "AICc", force = FALSE)

plot_df <- reshape2::melt(avg_models_CD_SRL)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_SRL_CD.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)

plot_df <- reshape2::melt(avg_models_CID_SRL)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_SRL_CID.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)





#-----------------------------------------------------------
# AFTER STATE CHANGES (AM/NM TO NM AND REMOVING ErM)
#-----------------------------------------------------------


#------------------
# ROOT DIAMETER
#------------------


load("./rdata/OU_RD_CID_4states.RData")
load("./rdata/OU_RD_CD_4states.RData")


models_CD_RD_4states <- list(EROUM=ER_OUM_RD_CD, EROUMA=ER_OUMA_RD_CD, EROUMV=ER_OUMV_RD_CD, EROUMVA=ER_OUMVA_RD_CD, ARDOUM=ARD_OUM_RD_CD,
                     ARDOUMA=ARD_OUMA_RD_CD, ARDOUMV=ARD_OUMV_RD_CD, ARDOUMVA=ARD_OUMVA_RD_CD, SYMOUM=SYM_OUM_RD_CD, SYMOUMA=SYM_OUMA_RD_CD,
                     SYMOUMV=SYM_OUMV_RD_CD, SYMOUMVA=SYM_OUMVA_RD_CD)

# rate.cat=2, null.model=TRUE
models_CID_RD_4states <- list(EROUM=ER_OUM_RD_CID, EROUMA=ER_OUMA_RD_CID, EROUMV=ER_OUMV_RD_CID, EROUMVA=ER_OUMVA_RD_CID,
                      ARDOUM=ARD_OUM_RD_CID, ARDOUMA=ARD_OUMA_RD_CID, ARDOUMV=ARD_OUMV_RD_CID, ARDOUMVA=ARD_OUMVA_RD_CID,
                      SYMOUM=SYM_OUM_RD_CID, SYMOUMA=SYM_OUMA_RD_CID, SYMOUMV=SYM_OUMV_RD_CID, SYMOUMVA=SYM_OUMVA_RD_CID)


lapply(models_CID_RD_4states, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))
lapply(models_CD_RD_4states, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))

avg_models_CD_RD_4states <- OUwie::getModelAvgParams(models_CD_RD_4states, type = "AICc", force = FALSE)
avg_models_CID_RD_4states <- OUwie::getModelAvgParams(models_CID_RD_4states, type = "AICc", force = FALSE)

plot_df <- reshape2::melt(avg_models_CID_RD_4states)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_RD_CID_4states.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)


plot_df <- reshape2::melt(avg_models_CD_RD_4states)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_RD_CD_4states.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)


stderr_ <- function(df) { lapply(X=df, FUN=function(column) {sd(column) / sqrt(length(column))}) |> unlist() }

avg_models_CD_RD_4states |> split(~tip_state) |> lapply(function(df) { colMeans(df[, c("rates", "alpha", "sigma.sq", "theta")]) }) |> as.data.frame()
avg_models_CID_RD_4states |> split(~tip_state) |> lapply(function(df) { colMeans(df[, c("rates", "alpha", "sigma.sq", "theta")]) }) |> as.data.frame()


avg_models_CD_RD_4states |> split(~tip_state) |> lapply(function(df) { stderr_(df[, c("rates", "alpha", "sigma.sq", "theta")]) })
avg_models_CID_RD_4states |> split(~tip_state) |> lapply(function(df) { stderr_(df[, c("rates", "alpha", "sigma.sq", "theta")]) })


#--------------------------
# SPECIFIC ROOT LENGTH
#--------------------------

load("./rdata/OU_SRL_CID_4states.RData")
load("./rdata/OU_SRL_CD_4states.RData")


models_CD_SRL_4states <- list(EROUM=ER_OUM_SRL_CD, EROUMA=ER_OUMA_SRL_CD, EROUMV=ER_OUMV_SRL_CD, EROUMVA=ER_OUMVA_SRL_CD, ARDOUM=ARD_OUM_SRL_CD,
                              ARDOUMA=ARD_OUMA_SRL_CD, ARDOUMV=ARD_OUMV_SRL_CD, ARDOUMVA=ARD_OUMVA_SRL_CD, SYMOUM=SYM_OUM_SRL_CD, SYMOUMA=SYM_OUMA_SRL_CD,
                              SYMOUMV=SYM_OUMV_SRL_CD, SYMOUMVA=SYM_OUMVA_SRL_CD)

models_CID_SRL_4states <- list(EROUM=ER_OUM_SRL_CID, EROUMA=ER_OUMA_SRL_CID, EROUMV=ER_OUMV_SRL_CID, EROUMVA=ER_OUMVA_SRL_CID, ARDOUM=ARD_OUM_SRL_CID,
                               ARDOUMA=ARD_OUMA_SRL_CID, ARDOUMV=ARD_OUMV_SRL_CID, ARDOUMVA=ARD_OUMVA_SRL_CID, SYMOUM=SYM_OUM_SRL_CID, SYMOUMA=SYM_OUMA_SRL_CID,
                               SYM_OUMV=SYM_OUMV_SRL_CID, SYMOUMVA=SYM_OUMVA_SRL_CID)

lapply(models_CID_SRL_4states, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))
lapply(models_CD_SRL_4states, function(mod){c(mod$loglik, mod$AIC, mod$AICc)}) |> as.data.frame(row.names = c("lnLik", "AIC", "AICc"))

avg_models_CD_SRL_4states <- OUwie::getModelAvgParams(models_CD_SRL_4states, type = "AICc", force = FALSE)
avg_models_CID_SRL_4states <- OUwie::getModelAvgParams(models_CID_SRL_4states, type = "AICc", force = FALSE)

plot_df <- reshape2::melt(avg_models_CID_SRL_4states)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_RD_CID_4states.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)


plot_df <- reshape2::melt(avg_models_CD_SRL_4states)
plot <- ggplot(plot_df, aes(x = tip_state, y = value, color = tip_state)) +
    geom_point(size = 5, shape = 21) +
    stat_summary(fun = mean, geom = "point", aes(group = 1, size = 2)) +
    stat_summary(fun.data = "mean_se", geom = "errorbar", aes(group = 1), width = 0.15, color = "black") +
    theme_classic(base_size = 22) + facet_wrap(~variable, scales = "free")
ggplot2::ggsave(plot = plot, filename = "../plots/hOUwie_RD_CD_4states.png", device = "png", width = 22, height = 12, units = "in", dpi = 750)


stderr_ <- function(df) { lapply(X=df, FUN=function(column) {sd(column) / sqrt(length(column))}) |> unlist() }

avg_models_CD_SRL_4states |> split(~tip_state) |> lapply(function(df) { colMeans(df[, c("rates", "alpha", "sigma.sq", "theta")]) }) |> as.data.frame()
avg_models_CID_SRL_4states |> split(~tip_state) |> lapply(function(df) { colMeans(df[, c("rates", "alpha", "sigma.sq", "theta")]) }) |> as.data.frame()


avg_models_CD_SRL_4states |> split(~tip_state) |> lapply(function(df) { stderr_(df[, c("rates", "alpha", "sigma.sq", "theta")]) })
avg_models_CID_SRL_4states |> split(~tip_state) |> lapply(function(df) { stderr_(df[, c("rates", "alpha", "sigma.sq", "theta")]) })
