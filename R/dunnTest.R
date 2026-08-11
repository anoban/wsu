library("FSA")

fred4_raw <- read.csv("../data/chapter2/FRED/subsets/continuous_raw.csv")
fred4_raw[, 5:7] <- fred4_raw[, 5:7] |> log() |> scale()

dunnRD <- FSA::dunnTest(F00679~as.factor(binominal), data = fred4_raw, method = "holm");
dunnSRL <- FSA::dunnTest(F00727~as.factor(binominal), data = fred4_raw, method = "holm");
dunnRTD <- FSA::dunnTest(F00709~as.factor(binominal), data = fred4_raw, method = "holm");

save(dunnRD, dunnSRL, dunnRTD, file = "../data/chapter2/rdata/dunnTest.RData")

load("../data/chapter2/rdata/dunnTest.RData")

# all the pairwise species comparisons
dunnRD$res$Comparison

length(dunnRD$res$Comparison) == length(unique(dunnRD$res$Comparison)) # damn
length(dunnRD$res$Comparison)

# species pairs with adjusted P value < 0.05
subset(dunnRD$res, P.adj < 0.05)

subset(dunnRD$res, P.adj < 0.05) |> dim() # 485 species pairs
subset(dunnSRL$res, P.adj < 0.05) |> dim() # 326 species pairs
subset(dunnRTD$res, P.adj < 0.05) |> dim() # 650 species pairs
