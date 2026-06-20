from os import path


def model_savepath(
    model_savedir: str, continuous_trait: str, discrete_model: str, continuous_model: str, nsims: int, null_model: bool
) -> str:
    """ """
    return path.join(model_savedir, f"{discrete_model}{continuous_model}_{continuous_trait}_{'CID' if null_model else 'CD'}_{nsims}.Rds")


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
    """
    dynamically create an R expression to fit hOUwie models, by invoking the R interpreter with, using the provided parameters
    """

    CONTINUOUS_TRAITS = ("F00679", "F00727", "F00709")
    DISCRETE_MODELS = ("ER", "SYM", "ARD")
    CONTINUOUS_MODELS = ("OUM", "OUMA", "OUMV", "OUMVA")

    # do a few sanity checks first
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

    _savepath = model_savepath(
        model_savedir=model_savedir,
        continuous_trait=continuous_trait,
        discrete_model=discrete_model,
        continuous_model=continuous_model,
        nsims=nsims,
        null_model=null_model,
    )

    return f"library('ape');library('OUwie');phylogeny <- ape::read.tree('{phylogeny}');data <- read.csv('{data}')[, c('binominal', 'state', '{continuous_trait}')];stopifnot(all(phylogeny$tip.label == data$binominal));model <- OUwie::hOUwie(phy = phylogeny, data = data, rate.cat = {2 if null_model else 1}, discrete_model = '{discrete_model}', continuous_model = '{continuous_model}', nSim = {nsims}, null.model = {'TRUE' if null_model else 'FALSE'});saveRDS(object = model, file = '{_savepath}');"


def main() -> None:
    pass


if __name__ == "__main__":
    main()
