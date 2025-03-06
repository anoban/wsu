# have already installed all these craps

# install.packages("devtools")
# devtools::install_github("ropenscilabs/datastorr")
# devtools::install_github("wcornwell/taxonlookup")

library(taxonlookup)

# hate to admit but this works lol :(
lookup_table(c("Pinus ponderosa","Quercus agrifolia"), by_species=TRUE)
lookup_table(c("Cynodon dactylon", "Azadirachta indica"), by_species=TRUE)

# let's serialize the lookup table so we don't have to use R again :)
write.csv(data.frame(plant_lookup()), fileEncoding = "latin1",
          file = "./plantlookup.csv", quote = FALSE, row.names = FALSE)

# THAT'S ALL I NEEDED :)))