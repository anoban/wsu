library("ape")
library("corHMM")

readallRds <- function(dirpath, cd, regstrip, rmunderscores = TRUE) {
    # dirpath = path to the directory containing the .Rds files
    # cd = character dependent or independent models, these are expected to contain "CD" or "CID" in their names
    # regstrip = the regex pattern to strip out of the file names when naming objects, can be a plain string as well
    # rmunderscores - strip all the underscores in the object name
    
    fnames <- list.files(dirpath) # all the files in the specified dir
    if(cd) fnames <- fnames[grep(pattern = "CD", fnames)]
    else fnames <- fnames[grep(pattern = "CID", fnames)] # cherry pick CD or CID models, assuming their names contain "CD" or "CID"
    # print(fnames)
    
    paths <- paste0(dirpath, fnames) # relative paths for all the Rds files
    stopifnot(length(fnames)==length(paths))
    
    mnames <- gsub(x = gsub(pattern = regstrip, replacement = '', x = fnames), pattern = ifelse(rmunderscores, '_', ''), replacement = '') # remove the unnecessary parts of the file names to create the model names
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
        from <- phylogeny$edge[i, ][1] # ancestor
        to <- phylogeny$edge[i, ][2] # descendant

        # corresponding discrete state shift
        transitions[[i]] <- paste0(allnodes[from, "disc"], "_to_", allnodes[to, "disc"])

        # corresponding continuous trait change
        changes_srl[i] <- (allnodes[from, "srl"] - allnodes[to, "srl"]) # SRL of ancestor - SRL of the descendant
        changes_rd[i] <- (allnodes[from, "rd"] - allnodes[to, "rd"]) # RD of ancestor - RD of the descendant

    }

    stopifnot(length(transitions)==nrow(phylogeny$edge))
    # return a dataframe of all shifts and the corresponding changes in continuous traits
    data.frame(shifts = unlist(transitions), delta_rd = changes_rd, delta_srl = changes_srl)

}
