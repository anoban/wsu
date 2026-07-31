# patched (customized functions from the OUwie library) - because the the library had some implementation issues
# and bugs

library("OUwie")

# patched alternative to OUwie::getModelTable
getModelTable_patched <- function(model.list, type="AICc"){
    # checks
    if(!inherits(model.list, what="list")){
        #if(class(model.list) != "list"){
        stop("Input object must be of class list with each element as a separet fit model to the same dataset.", call. = FALSE)
    }

    if(!all(unlist(lapply(model.list, function(x) class(x))) == "houwie")){
        warning("Not all models are of class houwie. These have been removed.")
        model.list <- model.list[unlist(lapply(model.list, function(x) class(x)) == "houwie")]
    }

    # length check needs to happen before the dim check
    if(length(model.list) < 2){
        stop("Two or models are needed to conduct model averaging.", call. = FALSE)
    }

    if(var(unlist(lapply(model.list, function(x) dim(x$data)[1]))) != 0){
        stop("The number of rows in your data are not the same for all models. Models should not be compared if they are not evaluating the same dataset.", call.=FALSE)
    }

    ParCount <- unlist(lapply(model.list, function(x) x$param.count))
    nTip <- length(model.list[[1]]$phy$tip.label) # number of species
    AIC <- simplify2array(lapply(model.list, "[[", type))
    dAIC <- AIC - min(AIC)
    AICwt <- exp(-0.5 * dAIC)/sum(exp(-0.5 * dAIC))
    LogLik <- simplify2array(lapply(model.list, "[[", "loglik"))
    DiscLik <- simplify2array(lapply(model.list, "[[", "DiscLik"))
    ContLik <- simplify2array(lapply(model.list, "[[", "ContLik"))

    model_table <- data.frame(np = ParCount,
                              lnLik = LogLik,
                              DiscLik = DiscLik,
                              ContLik = ContLik,
                              AIC = AIC,
                              dAIC = dAIC,
                              AICwt = AICwt)
    colnames(model_table) <- gsub("AIC", type, colnames(model_table))

    model_table
}

# patched alternative to OUwie::getModelAvgParams
getModelAvgParams_patched <- function(model.list, BM_alpha_treatment="zero", type="AICc", force=FALSE, threshold_diff=1e10){
    # the hardcoded threshold in hOUwie was 1e5, we made it adjustable for customizations

    is_houwie <- unlist(lapply(model.list, function(x) inherits(x, what="houwie")))
    if(!all(is_houwie)){
        warning("Some of the input models are not of class houwie, these have been removed.")
        model.list <- model.list[is_houwie]
    }

    if(!inherits(model.list, what="list") | length(model.list) < 2){
        #if(class(model.list) != "list" | length(model.list) < 2){
        stop("getModelAvgParams requires multiple houwie model objects to be input as a list.", call. = FALSE)
    }

    # these three lines are not needed at all
    # rate_cats <- simplify2array(lapply(model.list, "[[", "rate.cat"))
    # n_states <- simplify2array(lapply(model.list, function(x) dim(x$index.disc)[1]))
    # n_obs <- unique(n_states/rate_cats)

    # name the models
    if(is.null(names(model.list))){
        mod_names <- paste0("M", 1:length(model.list))
        names(model.list) <- mod_names
    }else{
        mod_names <- names(model.list)
    }

    # pull the weights based on the specified criterion - default is AICc
    mods_table <- getModelTable_patched(model.list, type=type)
    if(diff(range(mods_table[, 5])) > threshold_diff){ # 5th column is the criterion AICc or BICc
        if(!force){
            max_aic <- max(mods_table[, 5]) # the highest AICc
            # if you end up with extreme differences in AICcs where there are models in the list with AICcs
            # way lesser than the highest - could be a convergence failure - most failed models have an extrememly low AICc
            # often times a negative AICc
            # this will filter out the failed models
            model.list <- model.list[abs(mods_table[, 5] - max_aic)  < threshold_diff]

            mods_table <- getModelTable_patched(model.list, type=type)
            mod_names <- names(model.list)
        }else{
            warning("It is possible that one or more of your models failed to converge. The AIC between the best and worst models exceeds 1e10. Set force=FALSE to automatically remove potentially failed runs.")
        }
    }

    AICwts <- mods_table[, 7]
    tip_values_by_model <- lapply(model.list, OUwie:::get_tip_values) # get_tip_values is a "private" function
    for(i in 1:length(tip_values_by_model)){
        tip_values_by_model[[i]] <- tip_values_by_model[[i]] * AICwts[i] # apply the AICc weights to the model estimates
    }

    weighted_tip_values <- Reduce("+", tip_values_by_model)
    observed_tip_states <- model.list[[1]]$hOUwie.dat$PossibleTraits[as.numeric(model.list[[1]]$hOUwie.dat$data.cor[, 2])]
    names(observed_tip_states) <- model.list[[1]]$hOUwie.dat$data.cor[, 1]
    weighted_tip_values <- weighted_tip_values[match(names(observed_tip_states), rownames(weighted_tip_values)), ]
    weighted_tip_values$tip_state <- observed_tip_states

    weighted_tip_values
}
