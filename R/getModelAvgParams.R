library("OUwie")

getModelAveragedParams <- function(model.list, BM_alpha_treatment="zero", type="AICc", force=FALSE){

    if(any(unlist(lapply(model.list, class))!="houwie")){
        warning("Some of the input models are not of class houwie, these have been removed.")
        model.list <- model.list[which(unlist(lapply(model.list, class))=="houwie")]
    }

    if(!inherits(model.list, what="list") | length(model.list) < 2){
        #if(class(model.list) != "list" | length(model.list) < 2){
        stop("getModelAvgParams requires multiple houwie model objects to be input as a list.", call. = FALSE)
    }

    rate_cats <- simplify2array(lapply(model.list, "[[", "rate.cat"))
    n_states <- simplify2array(lapply(model.list, function(x) dim(x$index.disc)[1]))
    n_obs <- unique(n_states/rate_cats)

    # name the models
    if(is.null(names(model.list))){
        mod_names <- paste0("M", 1:length(model.list))
        names(model.list) <- mod_names
    }else{
        mod_names <- names(model.list)
    }

    # pull the aic weights
    print("line 28")
    mods_table <- OUwie::getModelTable(model.list, type=type)
    # print(mods_table)
    if(diff(range(mods_table[,5])) > 1e5){
        if(!force){
            max_aic <- max(mods_table[,5])
            model.list <- model.list[abs(mods_table[,5] - max_aic)  < 1e5]
            print(paste0("The size of model.list after removing the failed models is now ", length(model.list)))
            print("line 34")
            mods_table <- OUwie::getModelTable(model.list, type=type)
            mod_names <- names(model.list)
        }else{
            warning("It is possible that one or more of your models failed to converge. The AIC between the best and worst models exceeds 1e10. Set force=FALSE to automatically remove potentially failed runs.")
        }
    }

    # print(mods_table)
    AICwts <- mods_table[,7]
    # print(AICwts)
    tip_values_by_model <- lapply(model.list, get_tip_values)
    # print("here")
    # print(tip_values_by_model)
    for(i in 1:length(tip_values_by_model)){
        tip_values_by_model[[i]] <- tip_values_by_model[[i]] * AICwts[i]
    }

    weighted_tip_values <- Reduce("+", tip_values_by_model)
    observed_tip_states <- model.list[[1]]$hOUwie.dat$PossibleTraits[as.numeric(model.list[[1]]$hOUwie.dat$data.cor[,2])]
    names(observed_tip_states) <- model.list[[1]]$hOUwie.dat$data.cor[,1]
    weighted_tip_values <- weighted_tip_values[match(names(observed_tip_states), rownames(weighted_tip_values)),]
    weighted_tip_values$tip_state <- observed_tip_states

    return(weighted_tip_values)
}
