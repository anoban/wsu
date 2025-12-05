for (model in list.files(path = "./thirdparty/2020_houwie/empirical_fit")){
    load(file = paste0("./thirdparty/2020_houwie/empirical_fit", "/", model))
    print()
}



load("./2020_houwie/empirical_fit/FitSD=CID+_OUM.Rsave")
load("./2020_houwie/empirical_fit/FitSD=CID_OU1.Rsave")
load("./2020_houwie/empirical_fit/FitSD=CD_OUM.Rsave")
load("./2020_houwie/empirical_fit/FitSD=HYB_OUM.Rsave")

