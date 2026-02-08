import subprocess

R_FULL_PATH = r"C:/R-4.5.2/bin/R.exe"
RSCRIPT_FULL_PATH = r"C:/R-4.5.2/bin/Rscript.exe"

CONTINUOUS_TRAIT_MODELS = ("OUM", "OUMA", "OUMV", "OUMVA")
DISCRETE_TRAIT_MODELS = ("ER", "SYM", "ARD")


def __name_houwie_for_saving(
    discrete_model: str, continuous_model: str, null_model: bool, continuous_trait: str, discrete_trait: str, extra_suffix: str
) -> str:
    """
    generate an appropriate and informative name to serialzie the hOUwie models to disk

    :param discrete_model: Description
    :type discrete_model: str
    :param continuous_model: Description
    :type continuous_model: str
    :param null_model: Description
    :type null_model: bool
    :param continuous_trait: Description
    :type continuous_trait: str
    :param discrete_trait: Description
    :type discrete_trait: str
    :param extra_suffix: Description
    :type extra_suffix: str
    :return: Description
    :rtype: str
    """

    if discrete_model not in DISCRETE_TRAIT_MODELS:
        raise ValueError(f"Only the following values are are accepted for discrete_model: {DISCRETE_TRAIT_MODELS}")
    if continuous_model not in CONTINUOUS_TRAIT_MODELS:
        raise ValueError(f"Only the following values are are accepted for continuous_model: {CONTINUOUS_TRAIT_MODELS}")

    # e.g. ARD_OUMV_RD_MYCO_CD_395sp.Rds
    return f"{discrete_model}_{continuous_model}_{continuous_trait}_{discrete_trait}_{'CID' if null_model else 'CD'}_{extra_suffix}.Rds"


def generate_houwie_rscript(
    phylogeny: str,
    traitdata: str,
    rate_cat: int,
    discrete_model: str,
    continuous_model: str,
    null_model: bool,
    savedir: str,
    conttrait: str,
    disctrait: str,
    suffix: str,
    nsims: int = 30,
) -> str:
    """
    whips up a (string format) R script on the fly, so it can be passed via the expression (-e) argument to R.exe or Rscript.exe

    :param phylogeny: Description
    :type phylogeny: str
    :param traitdata: Description
    :type traitdata: str
    :param rate_cat: Description
    :type rate_cat: int
    :param discrete_model: Description
    :type discrete_model: str
    :param continuous_model: Description
    :type continuous_model: str
    :param null_model: Description
    :type null_model: bool
    :param savedir: Description
    :type savedir: str
    :param conttrait: Description
    :type conttrait: str
    :param disctrait: Description
    :type disctrait: str
    :param suffix: Description
    :type suffix: str
    :param nsims: Description
    :type nsims: int
    :return: Description
    :rtype: str

    """

    _R_SCRIPT_TEMPLATE = r"""
    suppressPackageStartupMessages({{
        library("ape")
        library("corHMM")
        library("OUwie")
    }})

    phylogeny <- ape::read.tree("{}")
    data <- read.csv("{}")
    stopifnot(all(phylogeny$tip.label == data$binominal))

    model <- OUwie::hOUwie(phy = phylogeny, data = data, rate.cat = {}, discrete_model = "{}", continuous_model = "{}", nSim = {}, null.model = {})
    saveRDS(object = model, file = "{}")
    """

    # 1st placeholder - path to the phylogenetic tree
    # 2nd placeholder - path to the trait data (MUST BE NAME MATCHED TO THE PHYLOGENY)
    # 3rd placeholder - rate category (1, 2)
    # 4th placeholder - one of the discrete model types
    # 5th placeholder - one of the continuous model types
    # 6th placeholder - number of simulations to run
    # 7th placeholder - NULL model (TRUE or FALSE)
    # 8th placeholder - path to serialize the fit model

    # the trait data is expected to have the following three columns (in the specified order) - binominal names, discrete trait and continuous trait
    # the binominal names must be identical to the tip labels of the phylogeny

    return _R_SCRIPT_TEMPLATE.format(
        phylogeny,
        traitdata,
        rate_cat,
        discrete_model,
        continuous_model,
        nsims,
        "TRUE" if null_model else "FALSE",
        savedir
        + __name_houwie_for_saving(
            discrete_model=discrete_model,
            continuous_model=continuous_model,
            null_model=null_model,
            continuous_trait=conttrait,
            discrete_trait=disctrait,
            extra_suffix=suffix,
        ),
    )


if __name__ == r"__main__":
    subprocess.Popen()
