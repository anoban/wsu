for (model in list.files(path = "./thirdparty/2020_houwie/empirical_fit")){
    load(file = paste0("./thirdparty/2020_houwie/empirical_fit", "/", model))
    print(paste("model =", model,
                "rate.cat =", fit$rate.cat,
                "discrete_model =", fit$discrete_model,
                "continuous_model =", fit$continuous_model,
                "nSim =", fit$nSim,
                "null.model =", fit$null.model
                ))
}



