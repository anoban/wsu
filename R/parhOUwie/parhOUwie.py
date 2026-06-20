from os import path


def create_rscript(
    phylogeny: str,
    data: str,
    model_savedir: str,
    continuous_trait: str,
    discrete_model: str,
    continuous_model: str,
    nsims: int,
    null_model: bool,
) -> str:
    """ """

    # do a few sanity checks first
    CONTINUOUS_TRAITS = ("F00679", "F00727", "F00709")
    DISCRETE_MODELS = ("ER", "SYM", "ARD")
    CONTINUOUS_MODELS = ("OUM", "OUMA", "OUMV", "OUMVA")
    EXPRESSION_TEMPLATE = r"library('ape');library('OUwie');phylogeny <- ape::read.tree('{}');data <- read.csv('{}')[, c('binominal', 'state', '{}')];stopifnot(all(phylogeny$tip.label == data$binominal));model <- OUwie::hOUwie(phy = phylogeny, data = data, rate.cat = {}, discrete_model = '{}', continuous_model = '{}', nSim = {}, null.model = '{}');saveRDS(object = model, file = '{}');"

    if continuous_trait not in CONTINUOUS_TRAITS:
        raise ValueError(f"Argument continuous_trait must be one of {CONTINUOUS_TRAITS}, but got {continuous_trait}")

    if discrete_model not in DISCRETE_MODELS:
        raise ValueError(f"Argument discrete_model must be one of {DISCRETE_MODELS}, but got {discrete_model}")

    if continuous_model not in CONTINUOUS_MODELS:
        raise ValueError(f"Argument continuous_model must be one of {CONTINUOUS_MODELS}, but got {continuous_model}")

    if not path.isfile(phylogeny):
        raise ValueError()

    if not path.isfile(data):
        raise ValueError()

    if not path.isdir(model_savedir):
        raise ValueError
