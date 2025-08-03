fred_subset <- read.csv("../data/fred_subset.csv")
data = fred_subset[, c("RD", "SRL", "RN", "RTD")]

mask <- fred_subset$F00056
mask[is.na(mask)] <- 0 # NA are replaced with 0

png("../plots/pca_order_grouped_r.png", width=10, height=10, units="in", res=300)
par(mfrow=c(2, 2), mar=rep(1, 4))
i <- 0
for (df in split(data, mask)){
   pca <- prcomp(scale(df))
   biplot(pca, main=i)
   i = i+1 
}
dev.off()
