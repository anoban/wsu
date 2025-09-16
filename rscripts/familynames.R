if (!require("devtools")) install.packages("devtools")
if (!require("datastorr")) devtools::install_github("ropenscilabs/datastorr")
if (!require("taxonlookup")) devtools::install_github("wcornwell/taxonlookup")

library(taxonlookup)

lookup_table(c("Pinus ponderosa", "Quercus agrifolia"), by_species = TRUE)
lookup_table(c("Cynodon dactylon", "Azadirachta indica"), by_species = TRUE)

# this is more elegant than the version downloadable from github :/
write.csv(data.frame(plant_lookup()), fileEncoding = "latin1", file = "../data/chapter2/plantlookup_serialized_from_r.csv",
          quote = FALSE, row.names = FALSE)
