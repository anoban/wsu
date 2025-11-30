results = read.csv("../data-simulated/simulate-results.csv")
results0 = read.csv("../data-simulated/simulate-results-uncorrelated.csv")

par(mfrow=c(2,2), mar=c(3,3,2,2), tcl=-0.2, mgp=c(3,0.4,0))
plot(rowSums(results[,1:3]), rowSums(results[,5:7]), las=1, bty="l", 
    pch=21, bg="white", log='xy', cex.axis=0.75, xlim=c(0.0011, 0.129), ylim=c(0.0011, 0.129))
mtext(expression(paste("True ", lambda[X]+lambda[Y]+lambda[XY])), 1, 1.5, 
    at=0.18, cex=0.8, adj=1)
mtext(expression(paste("Estimated ", lambda[X]+lambda[Y]+lambda[XY])), 2, 1.5, 
    at=0.18, las=2, cex=0.8, adj=0)
legend("bottomright", bty="n", cex=0.8, ncol=1, inset=0.015,
    legend=c("Generating model:\ncorrelated Mk", "",
             "Analysis model:\ncorrelated Mk"))
abline(0,1)


plot(rowSums(results0[,1:2]), rowSums(results0[,5:7]), las=1, pch=21, 
    bty='l', log='xy', bg="white", xlab="", ylab="", cex.axis=0.75,xlim=c(0.0011, 0.129),ylim=c(0.0011, 0.129))
mtext(expression(paste("True ", lambda[X]+lambda[Y])), 1, 1.5, 
    at=0.18, cex=0.8, adj=1)
mtext(expression(paste("Estimated ", lambda[X]+lambda[Y]+lambda[XY])), 2, 1.5, 
    at=0.18, las=2, cex=0.8, adj=0)
legend("bottomright", bty="n", cex=0.8, ncol=1,inset=0.015,
    legend=c("Generating model:\nindependent Mk", "",
             "Analysis model:\ncorrelated Mk"))
abline(0,1)

plot(results$corr, results$corr.hat, las=1, pch=21, bty='l',xlim=c(0,1),
  ylim=c(0,1), bg="white", xlab="", ylab="", cex.axis=0.75)
mtext(expression(paste("True ", rho[XY])), 1, 1.5, at=1, cex=0.8)
mtext(expression(paste("Estimated ", rho[XY])), 2, 1.5, at=1.07, las=2,
    cex=0.8, adj=0)
legend("bottomright", bty="n", cex=0.8, ncol=1,inset=0.015,
    legend=c("Generating model:\ncorrelated Mk", "",
             "Analysis model:\ncorrelated Mk"))
abline(0,1)


plot(rowSums(results[,1:3]), rowSums(results[,9:10]), las=1, bty="l", 
    pch=21, bg="white", log='xy', cex.axis=0.75, xlim=c(0.0011, 0.129),ylim=c(0.0011, 0.2))
mtext(expression(paste("True ", lambda[X]+lambda[Y]+lambda[XY])), 1, 1.5, 
    at=0.18, cex=0.8, adj=1)
mtext(expression(paste("Estimated ", lambda[X]+lambda[Y])), 2, 1.5, 
    at=0.28, las=2, cex=0.8, adj=0)
legend("bottomright", bty="n", cex=0.8, ncol=1,inset=0.015,
    legend=c("Generating model:\ncorrelated Mk", "",
             "Analysis model:\nindependent Mk"))
abline(0,1)

dev.print(pdf, file="simulate-results-fig.pdf")
