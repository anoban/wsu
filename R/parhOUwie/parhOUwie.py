#!usr/bin/env python3

import subprocess
from datetime import datetime
from os import path
from time import sleep
from typing import NamedTuple, Optional

DISCRETE_MODELS = ("ER", "SYM", "ARD")
CONTINUOUS_MODELS = ("OUM", "OUMA", "OUMV", "OUMVA")


def model_savepath(
    model_savedir: str,
    discrete_model: str,
    continuous_model: str,
    nsims: int,
    null_model: bool,
    continuous_trait: str,
    discrete_trait: Optional[str],
) -> str:
    """
    put together a path to serialize the hOUwie models, given the arguments
    the model name will be in the format: discrete_modelcontinuous_model_continuous_trait_CD/CID_nsims.Rds
    e.g. SYMOUMA_F00679_CD_100.Rds
    """

    return path.join(
        model_savedir,
        f"{discrete_model}{continuous_model}_{discrete_trait if discrete_trait else ''}{continuous_trait}_{'CID' if null_model else 'CD'}_{nsims}.Rds",
    )


def create_rscript(
    phylogeny: str,
    data: str,
    model_savedir: str,
    discrete_model: str,
    continuous_model: str,
    nsims: int,
    null_model: bool,
    discrete_trait: str,
    continuous_trait: str,
    binominal: str,
    include_disc_trait_in_model_names: bool = False,
) -> str:
    """
    dynamically create an R expression to fit hOUwie models, by invoking the R interpreter with, using the provided parameters
    using the template below:

    library('ape');
    library('OUwie');

    phylogeny <- ape::read.tree('{phylogeny}');
    data <- read.csv('{data}')[, c('{binominal}', '{discrete_trait}', '{continuous_trait}')];
    stopifnot(all(phylogeny$tip.label == data$binominal));

    model <- OUwie::hOUwie(phy = phylogeny, data = data, rate.cat = {2 if null_model else 1}, discrete_model = '{discrete_model}', continuous_model = '{continuous_model}', nSim = {nsims}, null.model = {'TRUE' if null_model else 'FALSE'});
    saveRDS(object = model, file = '{savepath}');
    """

    # do a few sanity checks first
    # if continuous_trait not in ("F00679", "F00727", "F00709"):
    #     raise ValueError(f"Argument continuous_trait must be one of F00679, F00727 or F00709, but got {continuous_trait}")

    if discrete_model not in DISCRETE_MODELS:
        raise ValueError(f"Argument discrete_model must be one of {DISCRETE_MODELS}, but got {discrete_model}")

    if continuous_model not in CONTINUOUS_MODELS:
        raise ValueError(f"Argument continuous_model must be one of {CONTINUOUS_MODELS}, but got {continuous_model}")

    if not path.isfile(phylogeny):
        raise RuntimeError(f"{phylogeny} doesn't exist or is not a file")

    if not path.isfile(data):
        raise RuntimeError(f"{data} doesn't exist or is not a file")

    with open(file=data, mode="rt") as fp:
        columns = fp.readline().replace("\n", "").replace('"', "").split(",")  # read the first line of the csv and extract the column names
        if (discrete_trait not in columns) or (continuous_trait not in columns) or (binominal not in columns):
            raise RuntimeError(
                f"File {data} is expected to contain all of the following three columns: {binominal, discrete_trait, continuous_trait} but {columns} were found instead"
            )

    if not model_savedir.endswith(("/")):
        raise ValueError(f"Argument model_savedir is expected to and with a '/', but {model_savedir} doesn't")

    if not path.isdir(model_savedir):
        raise RuntimeError(f"{model_savedir} doesn't exist or is not a directory")

    _savepath = model_savepath(
        model_savedir=model_savedir,
        discrete_model=discrete_model,
        continuous_model=continuous_model,
        nsims=nsims,
        null_model=null_model,
        continuous_trait=continuous_trait,
        discrete_trait=discrete_trait if include_disc_trait_in_model_names else None,
    )

    return f"library('ape');library('OUwie');phylogeny <- ape::read.tree('{phylogeny}');data <- read.csv('{data}')[, c('{binominal}', '{discrete_trait}', '{continuous_trait}')];stopifnot(all(phylogeny$tip.label == data$binominal));model <- OUwie::hOUwie(phy = phylogeny, data = data, rate.cat = {2 if null_model else 1}, discrete_model = '{discrete_model}', continuous_model = '{continuous_model}', nSim = {nsims}, null.model = {'TRUE' if null_model else 'FALSE'});saveRDS(object = model, file = '{_savepath}');"


class houwie_params(NamedTuple):
    """
    a named tuple class to define hOUwie model parameters

    discrete - discrete model type - one of "ER", "SYM" or "ARD"
    continuous - continuous model type - one of "OUM", "OUMA", "OUMV" or "OUMVA"
    null - null model - True or False
    """

    discrete: str
    continuous: str
    null: bool


def logger(directory: str, finished_proc: subprocess.Popen[str], fit: str, start: datetime, stop: datetime) -> None:
    """
    log the stdout and stderr of the list of processes to stdout.log and stderr.log files in the
    specified directory
    """

    with (
        open(file=path.join(directory, "stdout.log"), mode="a+") as fp_out,
        open(file=path.join(directory, "stderr.log"), mode="a+") as fp_err,
    ):
        fp_out.write(
            f"{fit}, started - {start}, finished - {stop}, duration - {(stop - start).total_seconds():,} seconds\n{finished_proc.stdout.read()}\n\n\n"  # pyright: ignore[reportOptionalMemberAccess]
        )
        fp_err.write(
            f"{fit}, started - {start}, finished - {stop}, duration - {(stop - start).total_seconds():,} seconds\n{finished_proc.stderr.read()}\n\n\n"  # pyright: ignore[reportOptionalMemberAccess]
        )


def handle_parallel_waits(logdir: str, launched_fits: dict[str, subprocess.Popen[str]], tick: datetime, wait_seconds: int = 60) -> None:
    """
    handle the processes launched in paralell, using subprocess.Popen()
    constantly check the processess every once in the specified timeframe (wait_seconds) and if any have finished, terminate that process
    read in its captured stdout and stderr and log them

    the keys of the dict are expected to be in the following format: "ARDOUMV_CID"
    """

    # this while block is the once a minute poll loop
    while any([proc.poll() is None for proc in launched_fits.values()]) or len(
        launched_fits
    ):  # while there are subprocesses that have not signalled completion or while the dict is not empty
        _finished_fits: list[str] = []
        for fit, proccess in launched_fits.items():
            if proccess.poll() is not None:  # if the process has signalled finish
                tock = datetime.now()
                proccess.terminate()  # terminate the process
                logger(directory=logdir, finished_proc=proccess, fit=fit, start=tick, stop=tock)  # log the details of the finished process
                _finished_fits.append(fit)  # will use this to remove the finished procs from the dict

        for fit in _finished_fits:
            del launched_fits[fit]  # remove the finished processes from the dict

        sleep(wait_seconds)  # wait for a minute before the next iteration


def main(
    rinterpreter: str,
    phylo: str,
    dataset: str,
    savedir: str,
    nsims: int,
    continuous_trait: str,
    discrete_trait: str = "state",
    binominal: str = "binominal",
) -> None:
    """ """

    HOUWIE_PARAMS = [  # all possible combinations of the three hOUwie model parameters
        houwie_params(discrete=d, continuous=c, null=n) for d in DISCRETE_MODELS for c in CONTINUOUS_MODELS for n in (True, False)
    ]

    tick = datetime.now()  # time at process launch
    procs = {  # launch all the 24 procs in parallel
        f"{params.discrete}{params.continuous}_{'CID' if params.null else 'CD'}": subprocess.Popen(  # subprocess.Popen is non-blocking whereas subprocess.call is blocking
            [
                rinterpreter,
                "-e",
                create_rscript(
                    phylogeny=phylo,
                    data=dataset,
                    model_savedir=savedir,
                    discrete_model=params.discrete,
                    continuous_model=params.continuous,
                    nsims=nsims,
                    null_model=params.null,
                    discrete_trait=discrete_trait,
                    continuous_trait=continuous_trait,
                    binominal=binominal,
                ),
            ],
            shell=False,  # do not show the shell
            stdout=subprocess.PIPE,  # establish a pipe to child process's stdout
            stderr=subprocess.PIPE,  # establish a pipe to child process's stderr
            text=True,  # data sent through the pipes are assumed to be utf8 encoded text
            encoding="utf8",
        )
        for params in HOUWIE_PARAMS
    }

    handle_parallel_waits(logdir=savedir, launched_fits=procs, tick=tick)


if __name__ == "__main__":
    main(
        rinterpreter=r"C:/Program Files/R/R-4.6.0/bin/R.exe",
        phylo=r"../primateEyes.phy",
        dataset=r"../primateEyes.csv",
        savedir="./",
        nsims=30,
        continuous_trait="Skull_length",
        discrete_trait="Activity_pattern_code",
        binominal="Genus_species",
    )
