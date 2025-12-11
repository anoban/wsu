set.seed(1123581321)
library(mkcor)

outline_clade = function(root, phy, ...)
{
    l = get("last_plot.phylo", env=.PlotPhyloEnv)
    tips = ephylo_tips(phy, root)
    # connect the tips
    tips = ephylo_tips(phy, root)
    r = max(phy$time[tips])
    theta1 = atan2(l$yy[head(tips, 1)], l$xx[head(tips, 1)])
    theta2 = atan2(l$yy[tail(tips, 1)], l$xx[tail(tips, 1)])
    x = r*cos(seq(theta1, theta2,,100))
    y = r*sin(seq(theta1, theta2,,100))
    # do the right outline
    node = tail(tips, 1)
    while (node != root) {
        r = phy$time[phy$parent[node]]
        theta1 = atan2(tail(y, 1), tail(x, 1))
        xx2 = l$xx[phy$parent[node]]
        yy2 = l$yy[phy$parent[node]]
        theta2 = atan2(yy2, xx2)
        xx1 = r*cos(seq(theta1, theta2,,10))
        yy1 = r*sin(seq(theta1, theta2,,10))
        x = c(x, xx1)
        y = c(y, yy1)
        node = phy$parent[node]
    }
    # do the left outline
    while (node != tips[1]) {
        xx2 = l$xx[phy$left.child[node]]
        yy2 = l$yy[phy$left.child[node]]
        theta1 = atan2(tail(y, 1), tail(x, 1))
        theta2 = atan2(yy2, xx2)
        r = phy$time[node]
        xx1 = r*cos(seq(theta1, theta2,,10))
        yy1 = r*sin(seq(theta1, theta2,,10))
        x = c(x, xx1, xx2)
        y = c(y, yy1, yy2)
        node = phy$left.child[node]
    }
    polygon(x, y, ...)
}

clade_theta = function(root, phy)
{
    l = get("last_plot.phylo", env=.PlotPhyloEnv)
    # connect the tips
    tips = ephylo_tips(phy, root)
    r = max(phy$time[tips])
    theta1 = atan2(l$yy[head(tips, 1)], l$xx[head(tips, 1)])
    theta2 = atan2(l$yy[tail(tips, 1)], l$xx[tail(tips, 1)])
    (theta1 + theta2) / 2
}

piepoints = function(x, rad, pie, piecol, piebg, ...)
{
    w = par("pin")[1]/diff(par("usr")[1:2])
    h = par("pin")[2]/diff(par("usr")[3:4])
    asp = w/h

    theta = apply(
        pie
        , 1
        , function(p) {
            ang = cumsum((360 * p / sum(p)) * (pi / 180))
            ang = cbind(c(0, ang[-length(ang)]), c(ang[-length(ang)], 2*pi))
            ang
        }
        , simplify=FALSE
    )

    for (i in seq_along(theta))
    {
        xx = x[i, 1]
        yy = x[i, 2]
        th = theta[[i]]
        for (j in 1:nrow(th))
        {
            if ((th[j,2] - th[j,1]) > 0)
            {
                wedges = seq(th[j, 1], th[j, 2], length.out=30)
                xvec = rad[i] * cos(wedges) + xx
                yvec = rad[i] * asp * sin(wedges) + yy
                if (isTRUE(all.equal(unname(th[j,2] - th[j,1]), 2*pi)))
                {
                    polygon(
                        xvec, yvec
                        , border=piecol[j]
                        , col=piebg[j]
                        , ...
                    )
                }
                else
                {
                    polygon(
                        c(xx, xvec), c(yy, yvec)
                        , border=piecol[j]
                        , col=piebg[j]
                        , ...
                    )
                }
            }
            else
                next
        }
    }
}

ttype1 = function(s) {
    n = nrow(s)
    m = matrix(0L, n, n)
    for (i in 1:n) {
        for (j in 1:n) {
            # coincident change
            if (s[i,1] != s[j,1] && s[i,2] != s[j,2])
                m[i,j] = 1L
        }
    }
    m
}

ttype2 = function(s) {
    n = nrow(s)
    m = matrix(0L, n, n)
    for (i in 1:n) {
        for (j in 1:n) {
            # coincident change
            if (s[i,1] != s[j,1] && s[i,2] != s[j,2]) {
                # increase plant consumption
                A = (s[i,2] %in% c("Carnivorous","Insectivorous","Omnivorous")
                        && s[j,2] == "Herbivorous") ||
                    (s[i,2] %in% c("Carnivorous","Insectivorous")
                        && s[j,2] == "Omnivorous")
                # increase cusp number
                B = as.integer(substr(s[i,1],1,1)) < as.integer(substr(s[j,1],1,1))
                if (A && B)
                    m[i,j] = 1L
            }
        }
    }
    m
}


ttype3 = function(s) {
    n = nrow(s)
    m = matrix(0L, n, n)
    for (i in 1:n) {
        for (j in 1:n) {
            # coincident change
            if (s[i,1] != s[j,1] && s[i,2] != s[j,2]) {
                # decrease plant consumption
                A = (s[i,2] == "Herbivorous" &&
                        s[j,2] %in% c("Carnivorous","Insectivorous","Omnivorous")) ||
                    (s[i,2] == "Omnivorous" &&
                        s[j,2] %in% c("Carnivorous","Insectivorous"))
                # decrease cusp number
                B = as.integer(substr(s[i,1],1,1)) > as.integer(substr(s[j,1],1,1))
                if (A && B)
                    m[i,j] = 1L
            }
        }
    }
    m
}


phy = as.ephylo(read.tree("../data-empirical/41467_2021_26285_MOESM4_ESM.txt"))

dat = read.csv("../data-empirical/41467_2021_26285_MOESM6_ESM.csv", sep=";")

# original
x = data.frame(tip.label=dat[,1],state.id=as.factor(dat[,6]))
y = data.frame(tip.label=dat[,1],state.id=factor(dat[,10],
    levels=c("Insectivorous","Carnivorous","Omnivorous","Herbivorous")))

fit = mkcor_fit_em(x, y, phy, num_fits=5)
fit0 = mkcor_fit_em(x, y, phy, correlated=FALSE, num_fits=5)

colv = c(
    rev(c("#543005",
    "#8c510a",
    "#bf812d",
    "#dfc27d")),

    "#80cdc1",
    "#35978f",
    "#01665e",
    "#003c30",

    rev(c("#40004b",
    "#762a83",
    "#9970ab",
    "#c2a5cf")),

    "#a6dba0",
    "#5aae61",
    "#1b7837",
    "#00441b"
)


idx1 = which(ttype1(fit$states) == 1L)
idx2 = which(ttype2(fit$states) == 1L)
idx3 = which(ttype3(fit$states) == 1L)

e1 = apply(fit$branch.counts, 3, function(m) sum(m[idx1]))
e2 = apply(fit$branch.counts, 3, function(m) sum(m[idx2]))
e3 = apply(fit$branch.counts, 3, function(m) sum(m[idx3]))

gekkota = 558L
dibamidae = 604L
scincoidea = 607L
polyglyphanodontia = 689L # nested in lacertoidea
lacertoidea = 683L
mosasuria = 785L
serpentes = 794L
anguimorpha = 886L
iguania = 924L

layout(matrix(c(1,1,2,2,1,1,3,2),4,2))
par(mar=c(0,0,1,1),xpd=NA)
plot(phy, show.tip.label=FALSE, type='fan', edge.color='dark grey',
    open.angle=180, edge.width=0.5)
l = get("last_plot.phylo", env=.PlotPhyloEnv)
outline_clade(gekkota, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
outline_clade(scincoidea, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
outline_clade(lacertoidea, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
outline_clade(serpentes, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
outline_clade(anguimorpha, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
outline_clade(iguania, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
r = 1.075*max(phy$time)
theta = clade_theta(gekkota, phy)
text(r*cos(theta), r*sin(theta), "Gekkota", cex=0.8)
theta = clade_theta(scincoidea, phy)
text(r*cos(theta), r*sin(theta), "Scincoidea", cex=0.8)
theta = clade_theta(lacertoidea, phy)
text(r*cos(theta), r*sin(theta), "Lacertoidea", cex=0.8)
theta = clade_theta(serpentes, phy)
text(r*cos(theta), r*sin(theta), "Serpentes", cex=0.8)
theta = clade_theta(anguimorpha, phy)
text(r*cos(theta), r*sin(theta), "Anguimorpha", cex=0.8)
theta = clade_theta(iguania, phy)
text(r*cos(theta), r*sin(theta), "Iguania", cex=0.8)
for (node in phy$root:phy$num.nodes) {
    piepoints(
        matrix(c(l$xx[node],l$yy[node]),nrow=1),
        rad=3, piecol=rep(1,16),
        pie=fit$state.prob[node,,drop=FALSE], piebg=colv, lwd=0.5
    )
}
mtext("Ancestral dental-diet state reconstructions",1,cex=0.6,line=-1.5,adj=0.17)
legend(
    "bottom",
    bty="n",
    pch=21,
    pt.bg=colv,
    pt.cex=1.5,
    pt.lwd=0.5,
    legend=paste(
    gsub("[0-9]_", "", fit$states[,1]),
    fit$states[,2],
    sep=" - "),
    ncol=4,
    cex=0.9,
    inset=-0.15
)

#par(mar=c(5,4,13,1))
par(mar=c(2,4,4,1),xpd=NA)
plot.new()
plot.window(ylim=c(0,1), xlim=c(1, 28))
ord = order(e1, decreasing=TRUE)[1:28]
for (i in 1:28) {
    node = ord[i]
    pnode = phy$parent[node]
    segments(i, 0, i, e1[node])
    piepoints(
        matrix(c(i,0),nrow=1),
        rad=0.25,piecol=rep(1,16),
        pie=fit$state.prob[pnode,,drop=FALSE], piebg=colv, lwd=0.3
    )
    piepoints(
        matrix(c(i,e1[node]), nrow=1),
        rad=0.25,piecol=rep(1,16),
        pie=fit$state.prob[node,,drop=FALSE], piebg=colv, lwd=0.3
    )
}
axis(2, las=1, tcl=-0.2, mgp=c(3,0.4,0))
#mtext("Edge index",1,line=1,cex=0.8)
mtext("Expected number of coincident\ndental-diet transitions on edge",2,line=2,
    cex=0.6)
text(2, -0.04, "Ancestral state", cex=0.8, adj=0)
text(2, 0.9, "Derived state", cex=0.8, adj=0)
par(mar=c(0,8,4,0))
plot(phy, show.tip.label=FALSE, type='fan', edge.color='dark grey',
    open.angle=180, edge.width=0.5)
l = get("last_plot.phylo", env=.PlotPhyloEnv)
outline_clade(gekkota, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
outline_clade(scincoidea, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
outline_clade(lacertoidea, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
outline_clade(serpentes, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
outline_clade(anguimorpha, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
outline_clade(iguania, phy, border=1, col=scales::alpha(1, 0.05), lwd=0.25)
edgelabels(edge=match(ord,phy$edge[,2]), cex=e1[ord], pch=19)
edgelabels(edge=match(ord,phy$edge[,2]), cex=e1[ord], pch=21, bg='white', lwd=0.25)
mtext("Distribution of coincident dental-diet transitions",1,cex=0.5,line=-0.5,adj=0.75)

dev.print(pdf, file="squamate-teeth-fig.pdf")
