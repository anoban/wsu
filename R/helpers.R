library("ape")
library("corHMM")

readallRds <- function(dirpath, cd, regstrip, rm_underscores = TRUE) {
    # dirpath = path to the directory containing the .Rds files
    # cd = character dependent or independent models, these are expected to contain "CD" or "CID" in their names
    # regstrip = the regex pattern to strip out of the file names when naming objects, can be a plain string as well
    # rm_underscores - strip all the underscores in the object name

    fnames <- list.files(dirpath) # all the files in the specified dir
    if(cd) fnames <- fnames[grep(pattern = "CD", fnames)]
    else fnames <- fnames[grep(pattern = "CID", fnames)] # cherry pick CD or CID models, assuming their names contain "CD" or "CID"
    # print(fnames)

    paths <- paste0(dirpath, fnames) # relative paths for all the Rds files
    stopifnot(length(fnames)==length(paths))

    mnames <- gsub(x = gsub(pattern = regstrip, replacement = '', x = fnames), pattern = ifelse(rm_underscores, '_', ''), replacement = '') # remove the unnecessary parts of the file names to create the model names
    # also remove all the underscores
    stopifnot(length(fnames)==length(mnames))

    models <- lapply(X = paths, FUN = readRDS) # read the needed Rds files into a list of objects
    stopifnot(length(models)==length(mnames))

    names(models) <- mnames # set their names
    models
}

# changes in discrete and continuous characters along every edge in the phylogeny
paired_dc_changes <- function(phylogeny, rdextant, srlextant, discextant, rdinternodes, srlinternodes, discprobinternodes, discstates) {
    # the phylogeny object that the continuous and discrete trait models were fitted with
    # rdextant - a named vector of RD values for the extant taxa in the phylogeny
    # srlextant - a named vector of SRL values for the extant taxa in the phylogeny
    # discextant - a named vector of discrete trait values for the extant taxa in the phylogeny

    # rdinternodes - ACE of RD of the internal nodes in the phylogeny (numeric vector)
    # srlinternodes - ACE of SRL of the internal nodes in the phylogeny (numeric vector)
    # discprobinternodes - ACE probabilities of the states for the internal nodes in the phylogeny (data.frame)

    # discstates - the names of the discrete states to replace the names in the discprobinternodes with

    # the tip labels of the phylogeny are expected to match with the names of the named trait vectors
    stopifnot(all(phylogeny$tip.label==names(rdextant)))
    stopifnot(all(phylogeny$tip.label==names(srlextant)))
    stopifnot(all(phylogeny$tip.label==names(discextant)))

    stopifnot(length(rdinternodes)==length(srlinternodes))
    stopifnot(nrow(discprobinternodes)==phylogeny$Nnode)
    stopifnot(length(discstates)==ncol(discprobinternodes))

    # all the reconstructed trait values of the internodes, in the order of the internode number
    aces <- data.frame(disc = discstates[apply(discprobinternodes, MARGIN = 1, FUN = which.max)], # corHMM uses stupid encodings for column names instead of the actual states
                       rd = rdinternodes, srl = srlinternodes)
    # in ACEs, the continuous trait reconstructions will have node numbers as names (for the internodes) but the discrete state probability table won't
    # when combined into a dataframe, the row names (internode numbers) will be used as the row names for the combined data frame.

    extants <- data.frame(disc = discextant, rd = rdextant, srl = srlextant) # trait values of the tips (leaf nodes)

    allnodes <- rbind(extants, aces) # all the extant species (tips), followed by the extinct clades (internodes)

    transitions <- list() # figure out all the discrete state transitions included in our reconstruction - this will be common for RD & SRL
    changes_srl <- vector(length = nrow(phylogeny$edge), mode = "double")
    changes_rd <- vector(length = nrow(phylogeny$edge), mode = "double")

    for (i in 1:nrow(phylogeny$edge)) {
        # PHYLOGENY$edge[i, ] returns a tuple of node numbers defining a branch (from, to)
        ancestor <- phylogeny$edge[i, ][1] # ancestor
        descendant <- phylogeny$edge[i, ][2] # descendant

        # corresponding discrete state shift
        transitions[[i]] <- paste0(allnodes[ancestor, "disc"], "_to_", allnodes[descendant, "disc"])

        # corresponding continuous trait change
        changes_srl[i] <- (allnodes[ancestor, "srl"] - allnodes[descendant, "srl"]) # SRL of ancestor - SRL of the descendant
        changes_rd[i] <- (allnodes[ancestor, "rd"] - allnodes[descendant, "rd"]) # RD of ancestor - RD of the descendant

    }

    stopifnot(length(transitions)==nrow(phylogeny$edge))
    # return a dataframe of all shifts and the corresponding changes in continuous traits
    data.frame(shifts = unlist(transitions), delta_rd = changes_rd, delta_srl = changes_srl)

}


failed_convergence <- function(model_table, stripoff='', threshold_diff_aicc=1e5){
  # expected to directly redirect the output of the OUwie::hOUwie() function to this function
  model_names <- row.names(model_table)
  model_names <- gsub(pattern = stripoff, replacement = '', x = model_names) # strip off the specified pattern from the names
  aiccs <- model_table[, "AICc"] # this becomes a hard requirement of the function, that the input dataframe needs to have a column named "AICc"
  # so the criteria is that the absoulte difference AICc of a given model and the model with the highest AICc (worst fit) shouldn't exceed 1e5,
  # else it's considered a convergence failure
  model_names[abs(aiccs - max(aiccs)) >= threshold_diff_aicc]  # with an option provided to pass in a custom difference threshold
}
