library("ape")
library("phytools")
library("nlme")

primate_eyes <- read.csv("./primateEyes.csv", row.names = "Genus_species")
primate_tree <- ape::read.tree("./primateEyes.phy")
species_list <- rownames(primate_eyes) # has underscores
corr <- ape::corBrownian(phy = primate_tree, form~species_list)
ancova <- nlme::gls(log(Orbit_area)~log(Skull_length)+Activity_pattern, data = primate_eyes, correlation = corr)
anova(ancova)

ptcolors <- setNames(c("blue", "red", "green"), nm = sort(unique(primate_eyes$Activity_pattern)))
dummy_x <- seq(min(primate_eyes$Skull_length), max(primate_eyes$Skull_length), length.out = 100)

par(mar=c(5, 5, 1, 1))
plot(y = primate_eyes$Orbit_area, x = primate_eyes$Skull_length, pch = 21, col = ptcolors[primate_eyes$Activity_pattern], log = "xy", ylab = "Orbit area", xlab = "Skull length")
legend("topright", legend = names(ptcolors), pt.bg = ptcolors, pch = 21)
lines(x = dummy_x, y = exp(predict(ancova, newdata = data.frame(Skull_length = dummy_x, Activity_pattern = rep("Nocturnal", 100)))), col = ptcolors["Nocturnal"])
