library("geiger")

# patched version of aov.phylo from geiger v2
# https://github.com/mwpennell/geiger-v2/blob/master/R/traits.R#L1013

phy.anova <- function(resp, group, data, taxa, phy, nsim=1000, test=c("Wilks", "Pillai", "Hotelling-Lawley", "Roy"), ...) {

    yy=merge(xx[[1]], xx[[2]], by=0)
    if(nrow(yy)==0) stop(flag)
    rownames(yy)=yy[,1]
    yy=yy[,-1]

    tmp<-treedata(phy, yy, sort=TRUE)
    phy=tmp$phy
    yy=yy[phy$tip.label,]

    group=structure(yy[,ncol(yy)], names=rownames(yy))
    dat=as.matrix(yy[,-ncol(yy)])
    rownames(dat)=rownames(yy)

    s<-ratematrix(phy, dat)

    multivar=ifelse(ncol(dat)>1, TRUE, FALSE)

    if(multivar){
        test=match.arg(test, c("Wilks", "Pillai", "Hotelling-Lawley", "Roy"))
        m=summary.manova(mod<-manova(dat~group), test=test)
        f.data=m[[4]][1,2]
        FUN=function(xx) summary.manova(manova(as.matrix(xx)~group), test=test)[[4]][1,2]
        sims<-sim.char(phy, s, nsim=nsim)
        f.null<-apply(sims, 3, FUN)
        out=as.data.frame(m[[4]])
        attr(out, "heading")=c("Multivariate Analysis of Variance Table\n","Response: dat")
    } else {
        test=NULL
        m=anova(mod<-lm(dat~group))
        f.data<-m[1,4]
        FUN=function(xx) anova(lm(xx~group))[1,4]
        out=as.data.frame(m)
    }

    colnames(out)=gsub(" ", "-", colnames(out))
    sims<-sim.char(phy, s, nsim=nsim)
    f.null<-apply(sims, 3, FUN)

    if(multivar) {
        if(test=="Wilks") {
            p.phylo = (sum(f.null < f.data) + 1)/(nsim + 1)
        } else {
            p.phylo = (sum(f.null > f.data) + 1)/(nsim + 1)
        }
    } else {
        p.phylo = (sum(f.null > f.data) + 1)/(nsim + 1)
    }

    out$'Pr(>F) given phy'=c(p.phylo, NA)
    class(out) <- c("anova", "data.frame")
    print(out, ...)
    attr(mod, "summary")=out
    return(mod)
}
