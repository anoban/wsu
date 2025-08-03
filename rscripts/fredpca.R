fred_subset <- read.csv("../data/fred_subset.csv")
data <- fred_subset[, c("RD", "SRL", "RN", "RTD")]

# PCA biplot, not grouping the data by root order
png("../plots/fred_4traits_pca_r.png", width = 15, height = 15, units = "in", res = 400)
biplot(prcomp(scale(data)))
dev.off()

# PCA biplots, grouped by root order
mask <- fred_subset$F00056
mask[is.na(mask)] <- 0 # NA are replaced with 0

png("../plots/pca_order_grouped_r.png", width = 15, height = 15, units = "in", res = 400)
par(mfrow = c(2, 2), mar = rep(1.95, 4))
i <- 0
for (df in split(data, mask)) {
    pca <- prcomp(scale(df))
    biplot(pca, main = i)
    i <- i + 1
}
dev.off()
