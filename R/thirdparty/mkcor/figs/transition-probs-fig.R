lerp = function(a, b, t) {
    (1-t)*a + t*b
}

pij = function(t, rx, ry, rxy, kx, ky, i, j, k, l)
{
    dx = -kx*(rx + rxy)/(kx-1)
    dy = -ky*(ry + rxy)/(ky-1)
    dxy = -(kx*rx/(kx-1)+ky*ry/(ky-1)+((kx-1)*(ky-1)-1)*rxy/((kx-1)*(ky-1)))

    if (i == j && k == l)
    {
        p = (1+(kx-1)*exp(dx*t)+(ky-1)*exp(dy*t)+(kx-1)*(ky-1)*exp(dxy*t))
    }
    else if (i == j && k != l)
    {
        p = (1+(kx-1)*exp(dx*t)-exp(dy*t)-(kx-1)*exp(dxy*t))
    }
    else if (i != j && k == l) 
    {
        p = (1-exp(dx*t)+(ky-1)*exp(dy*t)-(ky-1)*exp(dxy*t))
    }
    else
    {
        p = (1-exp(dx*t)-exp(dy*t)+exp(dxy*t))
    }
    p/(kx*ky)
}


solve_for_rxy = function(rx, ry, corr) {
    r2 = corr*corr
    r4 = r2^2
    (r2*(-(rx+ry)) - sqrt(r4*(rx+ry)^2 - 4*rx*ry*r2*(r2-1))) / (2*(r2 - 1))
}

layout(matrix(c(1,1,2,1),2,2))

par(mar=c(3,5,4,2)+0.1)
curve(pij(x, 2, 2, 0, 2, 2, 0, 0, 0, 0), 0, 1, ylim=c(0, 1),
    ylab="",lty=4,cex.axis=0.8,
    xlab="", las=1, mgp=c(2.25,.5,0), tcl=-0.2, n=512, bty="l",xaxt="n")
axis(1, tcl=-0.2, at=seq(0,1,.2), labels=rep("", 6))

#for (r in seq(.1, .9, .1))
for (r in c(.6, .9))
{
    rxy = solve_for_rxy(2,2,r)
    curve(pij(x, 2, 2, rxy, 2, 2, 0, 0, 0, 0), 0, 1, add=TRUE, col=1, n=512,lty=4)
}
#for (r in seq(0, .9, .1))
for (r in c(0, .6, .9))
{
    rxy = solve_for_rxy(2,2,r)
    curve(pij(x, 2, 2, rxy, 2, 2, 0, 1, 0, 1), 0, 1, add=TRUE, col=1, n=512,lty=1)
}
text(0.2, pij(0.2,2,2,0,2,2,0,1,0,1), 
    labels=expression(paste(rho[XY], " = ", 0)), pos=1, cex=0.75, offset=0.5)
text(0.15, pij(0.15,2,2,solve_for_rxy(2,2,.6),2,2,0,1,0,1), 
    labels=expression(paste(rho[XY], " = ", 0.6)), pos=1, cex=0.75, offset=0.85)
text(0.08, pij(0.1,2,2,solve_for_rxy(2,2,.9),2,2,0,1,0,1), 
    labels=expression(paste(rho[XY], " = ", 0.9)), pos=1, cex=0.75, offset=0.95)

text(0.2, pij(0.2,2,2,0,2,2,0,0,0,0), 
    labels=expression(paste(rho[XY], " = ", 0)), pos=4, cex=0.75, offset=0.25)
text(0.15, pij(0.15,2,2,solve_for_rxy(2,2,.6),2,2,0,0,0,0), 
    labels=expression(paste(rho[XY], " = ", 0.6)), pos=4, cex=0.75, offset=.25)
text(.1, pij(.1,2,2,solve_for_rxy(2,2,.9),2,2,0,0,0,0), 
    labels=expression(paste(rho[XY], " = ", 0.9)), pos=4, cex=0.75, offset=.25)

mtext("Time", 1, line=0.5, at=1, cex=0.8)
mtext("Probability", 2, line=0.5, at=1.05, cex=0.8, las=1)


legend("top", legend=c("Neither character changes", "Both characters change"), 
    lty=c(4, 1), bty="n", cex=0.8, ncol=2)


par(xpd=NA,mar=c(0,0,7,3))
plot.new()
plot.window(xlim=c(0,1), ylim=c(0,1))
text(0, 0.5, expression(paste(italic(i),", ",italic(k))), cex=0.8)
points(0,0.5,cex=6)
text(0.5, 1, expression(paste(italic(j),", ",italic(k))), cex=0.8)
points(0.5,1,cex=6)
text(0.5, 0, expression(paste(italic(i),", ",italic(l))), cex=0.8)
points(0.5,0,cex=6)
text(1, 0.5, expression(paste(italic(j),", ",italic(l))), cex=0.8)
points(1,0.5,cex=6)

arrows(0.1, 0.5, 0.9, 0.5, code=3, length=0.05)
text(0.5, 0.5, expression(frac(lambda[XY],(k[X]-1)(k[Y]-1))), pos=3, cex=0.8)

arrows(
    lerp(0,0.5,0.15), 
    lerp(0.5,1,0.1), 
    lerp(0,0.5,0.85), 
    lerp(0.5,1,0.85), 
    code=3, 
    length=0.05)
arrows(
    lerp(0.5,1,0.15), 
    lerp(0,0.5,0.15), 
    lerp(0.5,1,0.85), 
    lerp(0,0.5,0.85), 
    code=3, 
    length=0.05)
text(
    lerp(0,0.5,0.5), 
    lerp(0.5,1,0.5), 
    expression(frac(lambda[X],k[X]-1)), 
    pos=2, offset=1, cex=0.8)
text(
    lerp(0.5,1,0.5), 
    lerp(0,0.5,0.5), 
    expression(frac(lambda[X],k[X]-1)), 
    pos=4, offset=1.25, cex=0.8)


arrows(
    lerp(0.5,1,0.15), 
    lerp(1,0.5,0.15), 
    lerp(0.5,1,0.85), 
    lerp(1,0.5,0.85), 
    code=3, 
    length=0.05)
arrows(
    lerp(0,0.5,0.15), 
    lerp(0.5,0,0.15), 
    lerp(0,0.5,0.85), 
    lerp(0.5,0,0.85), 
    code=3, 
    length=0.05)
text(
    lerp(0.5,1,0.5), 
    lerp(1,0.5,0.5), 
    expression(frac(lambda[Y],k[Y]-1)), 
    pos=4, offset=1, cex=0.8)
text(
    lerp(0,0.5,0.5), 
    lerp(0.5,0,0.5), 
    expression(frac(lambda[Y],k[Y]-1)), 
    pos=2, offset=1, cex=0.8)

legend("bottom", bty="n",horiz=TRUE,inset=.25,
    legend=c(expression(italic(i) != italic(j)),
        expression(italic(k) != italic(l))),cex=0.8)

dev.print(pdf, file="transition-probs-fig.pdf")
