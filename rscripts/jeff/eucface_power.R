rm(list=ls())

setwd('~/RProjects/eucface_power')

library(lubridate)


##### foliar data #####

# foliar N, P and gas exchange

Ndata<-read.csv('NPdata.csv')
names(Ndata)[c(8, 10, 11, 13)] <- c('TreeHt2009', 'no.leaves', 'percentN', 'percentP')
GEdata<-read.csv('GEdata.csv')
names(GEdata)[c(4:12)] <- c('leaf.no', 'percentN', 'percentP', 'LMA', 'Narea', 'Anet_390', 'Anet_540', 'plotsite', 'height_m')
GEdata <- GEdata[, c(1:12)]

N1<-Ndata[which(Ndata$no.leaves==1),]
N2<-Ndata[which(Ndata$no.leaves>1),]

Nfull<-N1[,c(1:2,5:9,11,13)]
for(i in 1:nrow(N2)){
  tree<-N2$Tree[i]
  Nsub<-GEdata[which(GEdata$Tree==tree),'percentN']
  for(j in 1:length(Nsub)) {
    N2sub<-N2[i,c(1:2,5:9,11,13)]
    N2sub[1,'percentN']<-Nsub[j]
    Nfull<-rbind(Nfull,N2sub)
  }
}

rm(tree,Nsub,N2sub,i,j,N1,N2)


Nfull$Ring<-factor(Nfull$Ring)
Nfull$Tree<-factor(Nfull$Tree)
Nfull <- droplevels(Nfull)

GEdata$Ring<-0
GEdata$Ring[which(GEdata$Tree>=100)]<-1
GEdata$Ring[which(GEdata$Tree>=200)]<-2
GEdata$Ring[which(GEdata$Tree>=300)]<-3
GEdata$Ring[which(GEdata$Tree>=400)]<-4
GEdata$Ring[which(GEdata$Tree>=500)]<-5
GEdata$Ring[which(GEdata$Tree>=600)]<-6
GEdata$Ring<-factor(GEdata$Ring)
GEdata$Tree<-factor(GEdata$Tree)
GEdata <- droplevels(GEdata)


##### soil nutrient data #####

# extractables

extracts <- read.csv('extracts.csv', stringsAsFactors=F)
names(extracts)[2:4] <- c('Date', 'Ring', 'Plot')
extracts$Date <- as.Date(extracts$Date, format='%m/%d/%Y')

extracts$month <- month(extracts$Date)
extracts$month.chr <- as.character(extracts$month); extracts$month.chr[extracts$month < 10] <- paste('0', extracts$month.chr[extracts$month < 10], sep='')
extracts$year <- year(extracts$Date)
extracts$yearMonth <- factor(with(extracts, paste(year, month.chr, sep='-')))

extracts <- extracts[extracts$Date <= '2012-09-03', ]; extracts <- droplevels(extracts)
extracts$Date <- ordered(extracts$Date)
extracts$Ring <- factor(extracts$Ring)
extracts$Plot <- factor(extracts$Plot)

extracts$no <- log10(extracts$no)
extracts$nh <- log10(extracts$nh)
extracts$po <- log10(extracts$po)

# ion exchange resins

iems <- read.csv('iems.csv', stringsAsFactors=F)
names(iems)[4:6] <- c('Date', 'Ring', 'Plot')
iems$Date <- as.Date(iems$Date, format='%m/%d/%Y')

iems$month <- month(iems$Date)
iems$month.chr <- as.character(iems$month); iems$month.chr <- paste('0', iems$month.chr, sep='')
iems$year <- year(iems$Date)
iems$yearMonth <- factor(with(iems, paste(year, month.chr, sep='-')))

iems <- iems[!is.na(iems$Date), ]; iems <- droplevels(iems)
iems$Date <- ordered(iems$Date)
iems$Ring <- factor(iems$Ring)
iems$Plot <- factor(iems$Plot)

iems$Nitrate <- log10(iems$Nitrate)
iems$Ammonium <- log10(iems$Ammonium)
iems$Phosphate <- log10(iems$Phosphate)


#save.image('eucface_power.rdata')


# spatial signal?



n.ring<-6
n.tree<-3
n.leaf<-5

trt<-factor(rep(c('elevated','ambient'),each=n.ring/2*n.tree*n.leaf))
ring<-factor(rep(1:n.ring,each=n.tree*n.leaf))
tree<-rep(rep(1:n.tree,each=n.leaf),n.ring)
tree<-tree+rep(1:6*100,each=n.tree*n.leaf)
tree<-factor(tree)

effect.size<-0.15
effect<-c(rep(1+effect.size,n.ring/2*n.tree*n.leaf),rep(1,n.ring/2*n.tree*n.leaf))

rand<-rnorm(n.ring*n.tree*n.leaf,mean=0,sd=0.2)
ranef.tree<-rep(rnorm(n.ring*n.tree,mean=0,sd=0.2),each=n.leaf)

response<-ranef.tree+effect+rand

mod.alt<-lmer(response~trt+(1|tree),REML=F)
mod.nul<-lmer(response~1+(1|tree),REML=F)

tree.sd<-attr(VarCorr(mod.alt)[['tree']],'stddev')[['(Intercept)']]
resid.sd<-attr(VarCorr(mod.alt),'sc')
int.est<-attr(summary(mod.alt),'coefs')['(Intercept)','Estimate']
trt.est<-attr(summary(mod.alt),'coefs')['trtelevated','Estimate']
p.val<-anova(mod.nul,mod.alt)[['Pr(>Chisq)']][2]

out<-cbind(effect.size,n.ring,n.tree,n.leaf,p.val,int.est,trt.est,ranef.tree.sd,tree.sd,rand.sd,resid.sd)





cbind(trt,ring,tree,ranef.tree,rand,response)
plot(response~trt)
#bwplot(response~tree|trt)






## test run ##

source('eucfacePower_v1.R')
for(i in 1:1000) eucface.power('test.txt',0,5,3,0.173,0.160)

test<-read.table('test.txt',sep='\t')
names(test)<-c('effect.size','n.ring','n.tree','n.leaf','p.val','int.est','trt.est','ranef.tree.sd','tree.sd','rand.sd','resid.sd')

mean(test$p.val<0.05)

summary(test)


## real run -- foliar N ##

library(lme4)
library(AICcmodavg)

mod1.ringtree<-lmer(percentN~1+(1|Ring/Tree),data=Nfull)
mod1.ring<-lmer(percentN~1+(1|Ring),data=Nfull)
mod1.tree<-lmer(percentN~1+(1|Tree),data=Nfull)
anova(mod1.ringtree,mod1.ring,mod1.tree)
AICc(mod1.ringtree); AICc(mod1.tree); AICc(mod1.ring); AICc(lm(percentN~1,data=Nfull))
qqnorm(resid(mod1.tree)); qqline(resid(mod1.tree))

# > anova(mod1.ringtree,mod1.ring,mod1.tree)
# refitting model(s) with ML (instead of REML)
# Data: Nfull
# Models:
# mod1.ring: percentN ~ 1 + (1 | Ring)
# mod1.tree: percentN ~ 1 + (1 | Tree)
# mod1.ringtree: percentN ~ 1 + (1 | Ring/Tree)
#               Df     AIC     BIC  logLik deviance  Chisq Chi Df Pr(>Chisq)
# mod1.ring      3  7.0149 14.1983 -0.5075   1.0149
# mod1.tree      3 -1.2549  5.9285  3.6274  -7.2549 8.2698      0     <2e-16 ***
# mod1.ringtree  4  0.6210 10.1988  3.6895  -7.3790 0.1241      1     0.7246
# ---
# Signif. codes:  0 b
