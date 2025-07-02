# install.packages("devtools")
# devtools::install_github("ropenscilabs/datastorr")
# devtools::install_github("wcornwell/taxonlookup")

library(taxonlookup)

lookup_table(c("Pinus ponderosa","Quercus agrifolia"), by_species=TRUE)
lookup_table(c("Cynodon dactylon", "Azadirachta indica"), by_species=TRUE)
write.csv(data.frame(plant_lookup()), fileEncoding = "latin1",
          file = "./plantlookup.csv", quote = FALSE, row.names = FALSE)

