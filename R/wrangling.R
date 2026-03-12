library("ape")

#------------------------------------------------------------------------------------------
# THE NEW 995 SPECIES PHYLOGENY AFTER STATE CHANGES IN THE 1005 SPECIES PHYLOGENY
# 5 FINAL STATES WITH ErM AND OM SPECIES REMOVED
#------------------------------------------------------------------------------------------

megatree <- ape::read.tree("../data/chapter2/uphylomaker/GBOTB.extended.TPL.tre")
taxonomy_data <- read.csv("../data/chapter2/FREDv3subset/collab_fineroots_log_995_species_means_5states.csv")

species_list <- data.frame(species = taxonomy_data$binominal, genus = taxonomy_data$F01286, family = taxonomy_data$F01289, species.relative = NA, genus.relative = NA)
genus_list <- data.frame(genus = taxonomy_data$F01286, family = taxonomy_data$F01289)
genus_list <- genus_list[!duplicated(genus_list), ] # drop the duplicates

phylogeny <- U.PhyloMaker::phylo.maker(sp.list = species_list, tree = megatree, gen.list = genus_list)
ape::write.tree(phylogeny$phylo, file = "../data/chapter2/uphylomaker/collab_fineroots_log_995_species_means_5states.tre")

# name match the dataset
phylogeny <- ape::read.tree("../data/chapter2/uphylomaker/collab_fineroots_log_995_species_means_5states.tre")
taxonomy_data <- taxonomy_data[match(phylogeny$tip.label, taxonomy_data$binominal), ]
stopifnot(taxonomy_data$binominal == phylogeny$tip.label)
write.csv(x = taxonomy_data, file = "../data/chapter2/FREDv3subset/collab_fineroots_log_995_species_means_5states_name_matched_with_phylogeny.csv", row.names = FALSE)

