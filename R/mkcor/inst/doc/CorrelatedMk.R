### R code from vignette source 'CorrelatedMk.Rnw'

###################################################
### code chunk number 1: CorrelatedMk.Rnw:31-32
###################################################
options(width = 80, prompt = "> ")


###################################################
### code chunk number 2: CorrelatedMk.Rnw:260-264
###################################################
set.seed(1123581321)
library(mkcor)
data(squamate_tree)
data(squamate_dental_diet_states)


###################################################
### code chunk number 3: CorrelatedMk.Rnw:270-279
###################################################
x = data.frame(
    tip.label = squamate_dental_diet_states$species,
    state.id = factor(squamate_dental_diet_states$tooth.complexity)
)
y = data.frame(
    tip.label = squamate_dental_diet_states$species,
    state.id = factor(squamate_dental_diet_states$diet,
      levels=c("Insectivorous","Carnivorous","Omnivorous","Herbivorous"))
)


###################################################
### code chunk number 4: CorrelatedMk.Rnw:290-292
###################################################
fit0 = mkcor_fit_em(x, y, squamate_tree, correlated=FALSE, num_fits=5)
fit = mkcor_fit_em(x, y, squamate_tree, correlated=TRUE, num_fits=5)


###################################################
### code chunk number 5: CorrelatedMk.Rnw:304-311
###################################################
plot(fit$logL, las=1, ylab="log likelihood", xlab="EM iteration", 
  ylim=range(union(fit$logL, fit0$logL)), type='s', bty='l')
points(fit$logL, pch=19, cex=0.8)
points(fit0$logL, type='s')
points(fit0$logL, pch=21, cex=0.8, bg='white')
legend('right', legend=c("correlated", "independent"), lty=1, pch=c(19,21),
  pt.bg='white', title='Mk model', inset=0.01)


###################################################
### code chunk number 6: CorrelatedMk.Rnw:321-322
###################################################
pchisq(-2*(fit0$value - fit$value), 1, lower.tail=FALSE)


###################################################
### code chunk number 7: CorrelatedMk.Rnw:327-328
###################################################
fit$par


###################################################
### code chunk number 8: CorrelatedMk.Rnw:343-344
###################################################
c(fit$par$par[4] - 1.96*fit$par$se[4], fit$par$par[4] + 1.96*fit$par$se[4])


###################################################
### code chunk number 9: CorrelatedMk.Rnw:353-354
###################################################
fit$event.counts


###################################################
### code chunk number 10: CorrelatedMk.Rnw:376-377
###################################################
dim(fit$branch.counts)


###################################################
### code chunk number 11: CorrelatedMk.Rnw:387-388
###################################################
fit$states


###################################################
### code chunk number 12: classify_coincident_transitions
###################################################
classify_coincident_transitions = function(expected_counts, state_map)
{
  s = state_map
  n = nrow(s)
  counts = numeric(3)
  for (i in 1:n)
  {
    for (j in 1:n)
    {
      # coincident change
      if (s[i,1] != s[j,1] && s[i,2] != s[j,2])
      {
        counts[3] = counts[3] + expected_counts[i,j]
        # increase plant consumption
        A = (s[i,2] %in% c("Carnivorous","Insectivorous","Omnivorous") 
              && s[j,2] == "Herbivorous") ||
            (s[i,2] %in% c("Carnivorous","Insectivorous") 
              && s[j,2] == "Omnivorous")
        # increase cusp number
        B = as.integer(substr(s[i,1],1,1)) < as.integer(substr(s[j,1],1,1))
        if (A && B)
          counts[1] = counts[1] + expected_counts[i,j]
        # decrease plant consumption
        A = (s[i,2] == "Herbivorous" &&
              s[j,2] %in% c("Carnivorous","Insectivorous","Omnivorous")) ||
            (s[i,2] == "Omnivorous" &&
              s[j,2] %in% c("Carnivorous","Insectivorous"))
        # decrease cusp number
        B = as.integer(substr(s[i,1],1,1)) > as.integer(substr(s[j,1],1,1))
        if (A && B)
          counts[2] = counts[2] + expected_counts[i,j]
      }
    }
  }
  return (counts)
}


###################################################
### code chunk number 13: CorrelatedMk.Rnw:440-444
###################################################
branch_counts = apply(fit$branch.counts, 3, classify_coincident_transitions, 
  fit$states)
counts = rowSums(branch_counts)
counts


###################################################
### code chunk number 14: outline_clade
###################################################
outline_clade = function(root, phy, ...)
{
  l = get("last_plot.phylo", env=.PlotPhyloEnv)
  tips = ephylo_tips(phy, root)
  # connect the tips
  tips = ephylo_tips(phy, root)
  r = max(phy$time[tips])
  theta1 = atan2(l$yy[head(tips, 1)], l$xx[head(tips, 1)])
  theta2 = atan2(l$yy[tail(tips, 1)], l$xx[tail(tips, 1)])
  if (theta1 >= 0 && theta2 < 0)
    theta2 = theta2 + 2*pi
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
    if (theta1 >= 0 && theta2 < 0)
      theta2 = theta2 + 2*pi
    if (theta1 < 0 && theta2 >= 0)
      theta2 = theta2 - 2*pi
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
    if (theta1 >= 0 && theta2 < 0)
      theta2 = theta2 + 2*pi
    if (theta1 < 0 && theta2 >= 0)
      theta2 = theta2 - 2*pi
    r = phy$time[node]
    xx1 = r*cos(seq(theta1, theta2,,10))
    yy1 = r*sin(seq(theta1, theta2,,10))
    x = c(x, xx1, xx2)
    y = c(y, yy1, yy2)
    node = phy$left.child[node]
  }
  polygon(x, y, ...)
}


###################################################
### code chunk number 15: clade_theta
###################################################
clade_theta = function(root, phy)
{
  l = get("last_plot.phylo", env=.PlotPhyloEnv)
  # connect the tips
  tips = ephylo_tips(phy, root)
  r = max(phy$time[tips])
  theta1 = atan2(l$yy[head(tips, 1)], l$xx[head(tips, 1)])
  theta2 = atan2(l$yy[tail(tips, 1)], l$xx[tail(tips, 1)])
  if (theta1 >= 0 && theta2 < 0)
      theta2 = theta2 + 2*pi
  if (theta1 < 0 && theta2 >= 0)
    theta2 = theta2 - 2*pi
  (theta1 + theta2) / 2
}


###################################################
### code chunk number 16: outline_clade_plot_setup
###################################################
# node indices corresponding to major named squamate groups
gekkota = 558L
dibamidae = 604L
scincoidea = 607L
polyglyphanodontia = 689L # nested in lacertoidea
lacertoidea = 683L
mosasuria = 785L
serpentes = 794L
anguimorpha = 886L
iguania = 924L
# limit plotting to the top 3/4 of branches
ord = order(branch_counts[3,], decreasing=TRUE)[1:28]


###################################################
### code chunk number 17: outline_clade_plot
###################################################
par(mar=c(0,0,0,0),xpd=NA)
plot(squamate_tree, show.tip.label=FALSE, type='fan', edge.color='dark grey', 
    open.angle=5, edge.width=0.5)
l = get("last_plot.phylo", env=.PlotPhyloEnv)
outline_clade(gekkota, squamate_tree, border=1, col="#0000000D", 
  lwd=0.5)
outline_clade(scincoidea, squamate_tree, border=1, col="#0000000D", 
  lwd=0.5)
outline_clade(lacertoidea, squamate_tree, border=1, col="#0000000D", 
  lwd=0.5)
outline_clade(serpentes, squamate_tree, border=1, col="#0000000D", 
  lwd=0.5)
outline_clade(anguimorpha, squamate_tree, border=1, col="#0000000D", 
  lwd=0.5)
outline_clade(iguania, squamate_tree, border=1, col="#0000000D", 
  lwd=0.5)
edgelabels(edge=match(ord,squamate_tree$edge[,2]), cex=branch_counts[3, ord], 
  pch=19)
edgelabels(edge=match(ord,squamate_tree$edge[,2]), cex=branch_counts[3, ord], 
  pch=21, bg='white', lwd=0.25)
r = 1.05*max(squamate_tree$time)
theta = clade_theta(gekkota, squamate_tree)
text(r*cos(theta), r*sin(theta), "1", cex=0.8)
theta = clade_theta(scincoidea, squamate_tree)
text(r*cos(theta), r*sin(theta), "2", cex=0.8)
theta = clade_theta(lacertoidea, squamate_tree)
text(r*cos(theta), r*sin(theta), "3", cex=0.8)
theta = clade_theta(serpentes, squamate_tree)
text(r*cos(theta), r*sin(theta), "4", cex=0.8)
theta = clade_theta(anguimorpha, squamate_tree)
text(r*cos(theta), r*sin(theta), "5", cex=0.8)
theta = clade_theta(iguania, squamate_tree)
text(r*cos(theta), r*sin(theta), "6", cex=0.8)
legend(0,0, legend=c("1. Gekkota", "2. Scincoidea", "3. Lacertoidea",
  "4. Serpentes", "5. Anguimorpha", "6. Iguania"), ncol=2, bty='n',
  cex=0.8)


###################################################
### code chunk number 18: CorrelatedMk.Rnw:601-602
###################################################
print(fit$state.probs[squamate_tree$root, ], digits=3)


###################################################
### code chunk number 19: CorrelatedMk.Rnw:609-610
###################################################
fit$states[which.max(fit$state.probs[squamate_tree$root, ]),]


###################################################
### code chunk number 20: piepoints
###################################################
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


###################################################
### code chunk number 21: asr_plot
###################################################
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
layout(matrix(c(1,1,1,1,1,1,2,2),2,4))
par(mar=c(0,0,0,0),xpd=NA)
plot(squamate_tree, show.tip.label=FALSE, type='fan', edge.color='dark grey', 
    open.angle=5, edge.width=0.5)
l = get("last_plot.phylo", env=.PlotPhyloEnv)
for (node in 1:squamate_tree$num.nodes) {
    piepoints(
        matrix(c(l$xx[node],l$yy[node]),nrow=1),
        rad=3, piecol=rep(1,16),
        pie=fit$state.prob[node,,drop=FALSE], piebg=colv, lwd=0.5
    )
}
par(mar=c(0,0,0,0), xpd=NA)
plot.new()
plot.window(xlim=c(0,1),ylim=c(0,1))
legend(
    "center",
    bty="n",
    pch=21, 
    pt.bg=colv, 
    pt.cex=1.5,
    pt.lwd=0.5,
    legend=paste(
    gsub("[0-9]_", "", fit$states[,1]),
    fit$states[,2],
    sep=" - "),
    ncol=1,
    cex=0.9,
    inset=0.1
)


###################################################
### code chunk number 22: CorrelatedMk.Rnw:730-736
###################################################
iguania_plants = sum(y[ephylo_tips(squamate_tree,iguania),2]=="Herbivorous"
  |y[ephylo_tips(squamate_tree,iguania),2]=="Omnivorous")
skink_plants = sum(y[ephylo_tips(squamate_tree,scincoidea),2]=="Herbivorous"
  |y[ephylo_tips(squamate_tree,scincoidea),2]=="Omnivorous")
lacertid_plants = sum(y[ephylo_tips(squamate_tree,lacertoidea),2]=="Herbivorous"
  |y[ephylo_tips(squamate_tree,lacertoidea),2]=="Omnivorous")


###################################################
### code chunk number 23: CorrelatedMk.Rnw:755-756 (eval = FALSE)
###################################################
## classify_coincident_transitions = function(expected_counts, state_map)
## {
##   s = state_map
##   n = nrow(s)
##   counts = numeric(3)
##   for (i in 1:n)
##   {
##     for (j in 1:n)
##     {
##       # coincident change
##       if (s[i,1] != s[j,1] && s[i,2] != s[j,2])
##       {
##         counts[3] = counts[3] + expected_counts[i,j]
##         # increase plant consumption
##         A = (s[i,2] %in% c("Carnivorous","Insectivorous","Omnivorous") 
##               && s[j,2] == "Herbivorous") ||
##             (s[i,2] %in% c("Carnivorous","Insectivorous") 
##               && s[j,2] == "Omnivorous")
##         # increase cusp number
##         B = as.integer(substr(s[i,1],1,1)) < as.integer(substr(s[j,1],1,1))
##         if (A && B)
##           counts[1] = counts[1] + expected_counts[i,j]
##         # decrease plant consumption
##         A = (s[i,2] == "Herbivorous" &&
##               s[j,2] %in% c("Carnivorous","Insectivorous","Omnivorous")) ||
##             (s[i,2] == "Omnivorous" &&
##               s[j,2] %in% c("Carnivorous","Insectivorous"))
##         # decrease cusp number
##         B = as.integer(substr(s[i,1],1,1)) > as.integer(substr(s[j,1],1,1))
##         if (A && B)
##           counts[2] = counts[2] + expected_counts[i,j]
##       }
##     }
##   }
##   return (counts)
## }


###################################################
### code chunk number 24: CorrelatedMk.Rnw:759-760 (eval = FALSE)
###################################################
## outline_clade = function(root, phy, ...)
## {
##   l = get("last_plot.phylo", env=.PlotPhyloEnv)
##   tips = ephylo_tips(phy, root)
##   # connect the tips
##   tips = ephylo_tips(phy, root)
##   r = max(phy$time[tips])
##   theta1 = atan2(l$yy[head(tips, 1)], l$xx[head(tips, 1)])
##   theta2 = atan2(l$yy[tail(tips, 1)], l$xx[tail(tips, 1)])
##   if (theta1 >= 0 && theta2 < 0)
##     theta2 = theta2 + 2*pi
##   x = r*cos(seq(theta1, theta2,,100))
##   y = r*sin(seq(theta1, theta2,,100))
##   # do the right outline
##   node = tail(tips, 1)
##   while (node != root) {
##     r = phy$time[phy$parent[node]]
##     theta1 = atan2(tail(y, 1), tail(x, 1))
##     xx2 = l$xx[phy$parent[node]]
##     yy2 = l$yy[phy$parent[node]]
##     theta2 = atan2(yy2, xx2)
##     if (theta1 >= 0 && theta2 < 0)
##       theta2 = theta2 + 2*pi
##     if (theta1 < 0 && theta2 >= 0)
##       theta2 = theta2 - 2*pi
##     xx1 = r*cos(seq(theta1, theta2,,10))
##     yy1 = r*sin(seq(theta1, theta2,,10))
##     x = c(x, xx1)
##     y = c(y, yy1)
##     node = phy$parent[node]
##   }
##   # do the left outline
##   while (node != tips[1]) {
##     xx2 = l$xx[phy$left.child[node]]
##     yy2 = l$yy[phy$left.child[node]]
##     theta1 = atan2(tail(y, 1), tail(x, 1))
##     theta2 = atan2(yy2, xx2)
##     if (theta1 >= 0 && theta2 < 0)
##       theta2 = theta2 + 2*pi
##     if (theta1 < 0 && theta2 >= 0)
##       theta2 = theta2 - 2*pi
##     r = phy$time[node]
##     xx1 = r*cos(seq(theta1, theta2,,10))
##     yy1 = r*sin(seq(theta1, theta2,,10))
##     x = c(x, xx1, xx2)
##     y = c(y, yy1, yy2)
##     node = phy$left.child[node]
##   }
##   polygon(x, y, ...)
## }


###################################################
### code chunk number 25: CorrelatedMk.Rnw:763-764 (eval = FALSE)
###################################################
## clade_theta = function(root, phy)
## {
##   l = get("last_plot.phylo", env=.PlotPhyloEnv)
##   # connect the tips
##   tips = ephylo_tips(phy, root)
##   r = max(phy$time[tips])
##   theta1 = atan2(l$yy[head(tips, 1)], l$xx[head(tips, 1)])
##   theta2 = atan2(l$yy[tail(tips, 1)], l$xx[tail(tips, 1)])
##   if (theta1 >= 0 && theta2 < 0)
##       theta2 = theta2 + 2*pi
##   if (theta1 < 0 && theta2 >= 0)
##     theta2 = theta2 - 2*pi
##   (theta1 + theta2) / 2
## }


###################################################
### code chunk number 26: CorrelatedMk.Rnw:768-769 (eval = FALSE)
###################################################
## # node indices corresponding to major named squamate groups
## gekkota = 558L
## dibamidae = 604L
## scincoidea = 607L
## polyglyphanodontia = 689L # nested in lacertoidea
## lacertoidea = 683L
## mosasuria = 785L
## serpentes = 794L
## anguimorpha = 886L
## iguania = 924L
## # limit plotting to the top 3/4 of branches
## ord = order(branch_counts[3,], decreasing=TRUE)[1:28]


###################################################
### code chunk number 27: CorrelatedMk.Rnw:773-774 (eval = FALSE)
###################################################
## par(mar=c(0,0,0,0),xpd=NA)
## plot(squamate_tree, show.tip.label=FALSE, type='fan', edge.color='dark grey', 
##     open.angle=5, edge.width=0.5)
## l = get("last_plot.phylo", env=.PlotPhyloEnv)
## outline_clade(gekkota, squamate_tree, border=1, col="#0000000D", 
##   lwd=0.5)
## outline_clade(scincoidea, squamate_tree, border=1, col="#0000000D", 
##   lwd=0.5)
## outline_clade(lacertoidea, squamate_tree, border=1, col="#0000000D", 
##   lwd=0.5)
## outline_clade(serpentes, squamate_tree, border=1, col="#0000000D", 
##   lwd=0.5)
## outline_clade(anguimorpha, squamate_tree, border=1, col="#0000000D", 
##   lwd=0.5)
## outline_clade(iguania, squamate_tree, border=1, col="#0000000D", 
##   lwd=0.5)
## edgelabels(edge=match(ord,squamate_tree$edge[,2]), cex=branch_counts[3, ord], 
##   pch=19)
## edgelabels(edge=match(ord,squamate_tree$edge[,2]), cex=branch_counts[3, ord], 
##   pch=21, bg='white', lwd=0.25)
## r = 1.05*max(squamate_tree$time)
## theta = clade_theta(gekkota, squamate_tree)
## text(r*cos(theta), r*sin(theta), "1", cex=0.8)
## theta = clade_theta(scincoidea, squamate_tree)
## text(r*cos(theta), r*sin(theta), "2", cex=0.8)
## theta = clade_theta(lacertoidea, squamate_tree)
## text(r*cos(theta), r*sin(theta), "3", cex=0.8)
## theta = clade_theta(serpentes, squamate_tree)
## text(r*cos(theta), r*sin(theta), "4", cex=0.8)
## theta = clade_theta(anguimorpha, squamate_tree)
## text(r*cos(theta), r*sin(theta), "5", cex=0.8)
## theta = clade_theta(iguania, squamate_tree)
## text(r*cos(theta), r*sin(theta), "6", cex=0.8)
## legend(0,0, legend=c("1. Gekkota", "2. Scincoidea", "3. Lacertoidea",
##   "4. Serpentes", "5. Anguimorpha", "6. Iguania"), ncol=2, bty='n',
##   cex=0.8)


###################################################
### code chunk number 28: CorrelatedMk.Rnw:778-779 (eval = FALSE)
###################################################
## piepoints = function(x, rad, pie, piecol, piebg, ...)
## {
##   w = par("pin")[1]/diff(par("usr")[1:2])
##   h = par("pin")[2]/diff(par("usr")[3:4])
##   asp = w/h
## 
##   theta = apply(
##     pie
##     , 1
##     , function(p) {
##         ang = cumsum((360 * p / sum(p)) * (pi / 180))
##         ang = cbind(c(0, ang[-length(ang)]), c(ang[-length(ang)], 2*pi))
##         ang
##     }
##     , simplify=FALSE
##   )
## 
##   for (i in seq_along(theta))
##   {
##     xx = x[i, 1]
##     yy = x[i, 2]
##     th = theta[[i]]
##     for (j in 1:nrow(th))
##     {
##       if ((th[j,2] - th[j,1]) > 0)
##       {
##         wedges = seq(th[j, 1], th[j, 2], length.out=30)
##         xvec = rad[i] * cos(wedges) + xx
##         yvec = rad[i] * asp * sin(wedges) + yy
##         if (isTRUE(all.equal(unname(th[j,2] - th[j,1]), 2*pi)))
##         {
##           polygon(
##             xvec, yvec
##             , border=piecol[j]
##             , col=piebg[j]
##             , ...
##           )
##         }
##         else
##         {
##           polygon(
##             c(xx, xvec), c(yy, yvec)
##             , border=piecol[j]
##             , col=piebg[j]
##             , ...
##           )
##         }
##       }
##       else
##         next
##     }
##   }
## }


###################################################
### code chunk number 29: CorrelatedMk.Rnw:783-784 (eval = FALSE)
###################################################
## colv = c(
##   rev(c("#543005",
##   "#8c510a",
##   "#bf812d",
##   "#dfc27d")),
## 
##   "#80cdc1",
##   "#35978f",
##   "#01665e",
##   "#003c30",
## 
##   rev(c("#40004b",
##   "#762a83",
##   "#9970ab",
##   "#c2a5cf")),
## 
##   "#a6dba0",
##   "#5aae61",
##   "#1b7837",
##   "#00441b"
## )
## layout(matrix(c(1,1,1,1,1,1,2,2),2,4))
## par(mar=c(0,0,0,0),xpd=NA)
## plot(squamate_tree, show.tip.label=FALSE, type='fan', edge.color='dark grey', 
##     open.angle=5, edge.width=0.5)
## l = get("last_plot.phylo", env=.PlotPhyloEnv)
## for (node in 1:squamate_tree$num.nodes) {
##     piepoints(
##         matrix(c(l$xx[node],l$yy[node]),nrow=1),
##         rad=3, piecol=rep(1,16),
##         pie=fit$state.prob[node,,drop=FALSE], piebg=colv, lwd=0.5
##     )
## }
## par(mar=c(0,0,0,0), xpd=NA)
## plot.new()
## plot.window(xlim=c(0,1),ylim=c(0,1))
## legend(
##     "center",
##     bty="n",
##     pch=21, 
##     pt.bg=colv, 
##     pt.cex=1.5,
##     pt.lwd=0.5,
##     legend=paste(
##     gsub("[0-9]_", "", fit$states[,1]),
##     fit$states[,2],
##     sep=" - "),
##     ncol=1,
##     cex=0.9,
##     inset=0.1
## )


