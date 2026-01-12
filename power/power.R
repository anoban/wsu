library("lme4")
library("lmerTest")

#-------------
# CONSTANTS
#-------------

PAIRS6X6 <- expand.grid(1:6, 1:6) # all possible combinations of soils and seeds
NROWS_PAIRS6X6 <- nrow(PAIRS6X6)
colnames(PAIRS6X6) <- c("seed", "soil") # update the column names

# parameters to specify a realistic random dist for SRL
MEAN_SRL <- 27.4254539108333 # mean(themeda$SRL)
STD_SRL <- 10.6594022650714 # sd(themeda$SRL)

NREPLICATES <- seq(from = 3, to = 12) # a realistic range of replicates we can afford to have in the experiment
EFFECT_SIZES <- seq(from = 0.1, to = 0.9, length.out = 30) # range of effect sizes to test
NITERATIONS <- 1000 # number of iterations to compute the proportion of times we had the defired effect?????

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# compute the power given effect size and number of replicates while handling warnings gracefully
#-----------------------------------------------------------------------------------------------------------------------------------------------------

# I'm yet to find a language with a syntax more horrendous than this POS :/

power <- function(dataset, niters, efsize, ssize, mn, stdev, bmask) {
    # dataset - the data to reference the variables in the formula against, will also be passed to lmerTest::lmer
    # niters - number of times to repeat the sampling & model fit
    # efsize - effect size
    # ssize - sample size
    # mn - mean for the normal dist to draw trait (SRL) values from
    # stdev - standard deviation for the normal dist to draw trait (SRL) values from
    # bmask - boolean mask specifying the sympatric records

    pvalues <- vector(mode = "numeric", length = niters)

    for(i in 1:niters) {
        failed_to_converge <- FALSE
        withCallingHandlers( # look up for reference - https://adv-r.hadley.nz/conditions.html
            warning = function(cndtn) { # run this function when a warning is signalled
                # probable causes for this handler being called -
                # 1) Model failed to converge with 1 negative eigenvalue
                # 2) Model failed to converge with max|grad|
                failed_to_converge <<- TRUE
                # more on invokeRestart at https://docs.tibco.com/pub/enterprise-runtime-for-R/6.1.0/doc/html/Language_Reference/base/conditions.html
                invokeRestart(r="muffleWarning") # has a simple recovery strategy: “Suppress the warning”. it consumes the warning (so it does not “bubble up” to higher function call levels) and resumes the execution.
            },
            message = function(cndtn) { # run this function when a message is signalled
                # probable cause for this handler getting called - "boundary (singular) fit: see help('isSingular')"
                invokeRestart(r="muffleWarning")
            },
            expr = { # the block of code to be executed within the control of the handlers
                repeat {
                    dataset["SRL"] <- rnorm(mean = mn, sd = stdev, n = ssize) # randomly populate the SRL, using the specified params
                    dataset$SRL[bmask] <- dataset$SRL[bmask] * (1 + efsize) # apply the effect for sympatric records
                    mixmod <- lmerTest::lmer(formula = "SRL~gsep+(1|seed)+(1|soil)", data = dataset) # fit the model to the data, if we do use lme4::lmer(), the next line will raise an error because the output of lme4::lmer does not include p values
                    # if a warning is emitted from the above line, the handler will set failed_to_converge to TRUE
                    pvalues[i] <- anova(mixmod)["gsep", "Pr(>F)"] # this line will always be executed, this is where the control returns after a warning is caught
                    if (!failed_to_converge) break # if the model converged succesfully, break out the loop, else continue to resample and model fit.
                }
            }
        )
    }
    mean(pvalues < 0.05) # return fraction of p-values that are less than 0.05 (which is our power????)
}


#---------------------
# simulation loop
#---------------------

tm <- Sys.time()

for (i in seq_along(NREPLICATES)) {
    data <- PAIRS6X6[rep(1:NROWS_PAIRS6X6, NREPLICATES[i]), ] # expand the PAIRS6X6 dataframe such that each row gets repcilated NREPLICATES[i] times
    bmask_sympatric <- (data$soil == data$seed) # a boolean mask for sympatric records in the dataset
    # new column based on the boolean mask => geographical separation, convenient for specifying the model formula
    # without the result of lapply() wrapped inside an unlist(), the new column is annealed as a dataframe instead of a vector, which raises an exception when fitting the model or just use mapply()
    data["gsep"] <- mapply(bmask_sympatric, FUN = function (b) ifelse(b, 'S', 'A'))
    ss <- nrow(data)

    for (j in seq_along(EFFECT_SIZES)) {
        # allopatric vs sympatric becomes our fixed effect => ~gsep and seed and soil origins become our (non nested) random effects
        # https://stats.stackexchange.com/questions/674034/statistical-test-for-significance-of-mean-differences-between-two-groups
        # in the matrix called power, rows are effect sizes and columns are number of reps
        power[j, i] <- power(dataset = data, niters = NITERATIONS, efsize = EFFECT_SIZES[j], ssize = ss, mn = MEAN_SRL, stdev = STD_SRL, bmask = bmask_sympatric)
    }
}

tm <- Sys.time() - tm
