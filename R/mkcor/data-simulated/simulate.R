#! /usr/local/bin/Rscript --vanilla

library(mkcor)

sim_mkcor = function(phy, correlated=TRUE) {
    kx = sample(2:5, 1)
    ky = sample(2:5, 1)
    rx = sample(2:20, 1) / sum(phy$brlen)
    ry = sample(2:20, 1) / sum(phy$brlen)
    rxy = sample(2:20, 1) / sum(phy$brlen)
    rho = rxy / sqrt((rx+rxy)*(ry+rxy))
    if (correlated) {
        dat = simulate_mkcor(c(rx, ry, rxy), kx, ky, phy)
    } else {
        rxy = 0
        rho = 0
        dat = simulate_mkcor(c(rx, ry), kx, ky, phy)
    }
    list(data=dat, rx=rx, ry=ry, rxy=rxy, corr=rho, kx=kx, ky=ky,
        nx=tabulate(dat$x$state.id, kx), ny=tabulate(dat$y$state.id, ky))
}

sz.lo = 100L
sz.hi = 600L

NCLADES = 50L
NREPS = 10L
N = NCLADES*NREPS

CORRELATED = as.logical(commandArgs(trailingOnly=TRUE)[1L])

phy = as.ephylo(read.tree("squamates_Title_Science2024_ultrametric_constrained.tre"))

rclade = ephylo_rclade(phy, sz.lo, sz.hi, replace=FALSE)

clades = rclade(NCLADES)
corr.hat = rep(NA_real_, N)
rx.hat = rep(NA_real_, N)
ry.hat = rep(NA_real_, N)
rxy.hat = rep(NA_real_, N)
corr.se = rep(NA_real_, N)
rx.se = rep(NA_real_, N)
ry.se = rep(NA_real_, N)
rxy.se = rep(NA_real_, N)
rx0.hat = rep(NA_real_, N)
ry0.hat = rep(NA_real_, N)
rx0.se = rep(NA_real_, N)
ry0.se = rep(NA_real_, N)
corr = rep(NA_real_, N)
rx = rep(NA_real_, N)
ry = rep(NA_real_, N)
rxy = rep(NA_real_, N)
LR = rep(NA_real_, N)
kx = rep(NA_integer_, N)
ky = rep(NA_integer_, N)
tree.index = rep(NA_integer_, N)
tree.size = rep(NA_integer_, N)
ii = 1
for (i in 1:NCLADES)
{
    simtree = as.ephylo(extract.clade(as.phylo(phy), clades[i]))
    for (j in 1:NREPS)
    {
        repeat {
            sim = sim_mkcor(simtree, CORRELATED)
            if (all(sim$nx > 0) && all(sim$ny > 0)) break
        }
        corr[ii] = sim$corr
        rx[ii] = sim$rx
        ry[ii] = sim$ry
        rxy[ii] = sim$rxy
        kx[ii] = sim$kx
        ky[ii] = sim$ky
        tree.index[ii] = i
        tree.size[ii] = simtree$num.tips

        fit0 = mkcor_fit_em(sim$data$x, sim$data$y, simtree, correlated=FALSE, 
            tol=1e-4, num_fits=3)
        fit = mkcor_fit_em(sim$data$x, sim$data$y, simtree, correlated=TRUE,
            tol=1e-4, num_fits=3)
        if (fit$value < fit0$value)
        {
            fit = mkcor_fit_em(sim$data$x, sim$data$y, simtree, correlated=TRUE,
                tol=1e-5, num_fits=3)
        }

        rx.hat[ii] = fit$par$par[1]
        ry.hat[ii] = fit$par$par[2]
        rxy.hat[ii] = fit$par$par[3]
        corr.hat[ii] = fit$par$par[4]
        rx0.hat[ii] = fit0$par$par[1]
        ry0.hat[ii] = fit0$par$par[2]
        LR[ii] = -2*(fit0$value - fit$value)
        cat(sprintf("%d / %d\n", ii, N))
        ii = ii + 1
    }
}

results = data.frame(
    rx=rx,
    ry=ry,
    rxy=rxy,
    corr=corr,
    rx.hat=rx.hat,
    ry.hat=ry.hat,
    rxy.hat=rxy.hat,
    corr.hat=corr.hat,
    rx0.hat=rx0.hat,
    ry0.hat=ry0.hat,
    LR=LR,
    kx=kx,
    ky=ky,
    tree.index=tree.index,
    tree.size=tree.size
)

if (CORRELATED) {
    write.csv(results, file="simulate-results.csv", row.names=FALSE)
} else {
    write.csv(results, file="simulate-results-uncorrelated.csv", row.names=FALSE)
}

