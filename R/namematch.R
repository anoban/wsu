library("ape")

# name match the finalized root trait data with the phylogeny

tree <- ape::read.tree("./../data/chapter2/uphylomaker/FRED4_1301_species.tre")
tree

# may be we should get rid of the ErM and OrM species before name matching

traits <- read.csv("./../data/chapter2/FRED/subsets/final_name_matched.csv")
traits$binominal <- gsub(traits$binominal, pattern = ' ', replacement = '_')

setdiff(tree$tip.label, traits$binominal) # some species failed to bind during the phylogeny creating and we used their synonyms
# these need to be revised in the trait data dataset, otherwise it will propagate errors during model fits!


idx <- match(tree$tip.label, traits$binominal)
idx
all(traits[idx, ]$binominal == tree$tip.label)
