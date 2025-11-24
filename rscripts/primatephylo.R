library("ape")
library("phytools")
library("nlme")

primate_eyes <- read.csv("./primateEyes.csv", row.names = "Genus_species")
primate_tree <- ape::read.tree("./primateEyes.phy")
species_list <- rownames(primate_eyes) # has underscores
corr <- ape::corBrownian(phy = primate_tree, form~species_list)
ancova <- nlme::gls(log(Orbit_area)~log(Skull_length)+Activity_pattern, data = primate_eyes, correlation = corr)

