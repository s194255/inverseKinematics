setwd("C:/Users/malth/Documents/Fagpakkeprojekt")
data <- read.csv('ExperimentDataProcessed.csv',sep=";")


#Defining data
data.experiment1 <- data[1:36]
data.experiment2 <- data[37:72]

#Means/Proportions for experiment 1

NFwalking <- mean(colMeans(data.experiment1[1:4]))
NFrunning <- mean(colMeans(data.experiment1[5:8]))
NFboxing <- mean(colMeans(data.experiment1[9:12]))
Gausswalking <- mean(colMeans(data.experiment1[13:16]))
Gaussrunning <- mean(colMeans(data.experiment1[17:20]))
Gaussboxing <- mean(colMeans(data.experiment1[21:24]))
Truewalking <- mean(colMeans(data.experiment1[25:28]))
Truerunning <- mean(colMeans(data.experiment1[29:32]))
Trueboxing <- mean(colMeans(data.experiment1[33:36]))
NF <- mean(colMeans(data.experiment1[1:12]))
Gauss <- mean(colMeans(data.experiment1[13:24]))
True <- mean(colMeans(data.experiment1[25:36]))

Means <- c(NFwalking,NFrunning,NFboxing,Gausswalking,Gaussrunning,Gaussboxing,Truewalking,Truerunning,Trueboxing, NF, Gauss, True)

#Experiment 1
#Vectors
#NF
NFwalkingvec <- c(data.experiment1$ï..NFwalkingrhand1,data.experiment1$NFwalkingrhand2,data.experiment1$NFwalkinglfoot1,data.experiment1$NFwalkinglfoot2)
NFrunningvec <- c(data.experiment1$NFrunningrhand1,data.experiment1$NFrunningrhand2,data.experiment1$NFrunninglfoot1,data.experiment1$NFrunninglfoot2)
NFboxingvec <- c(data.experiment1$NFboxingrhand1, data.experiment1$NFboxingrhand2, data.experiment1$NFboxinglfoot1,data.experiment1$NFboxinglfoot2)
NFall <- c(NFwalkingvec,NFrunningvec,NFboxingvec)
#Gauss
Gausswalkingvec <- c(data.experiment1$Gausswalkingrhand1,data.experiment1$Gausswalkingrhand2,data.experiment1$Gausswalkinglfoot1,data.experiment1$Gausswalkinglfoot2)
Gaussrunningvec <- c(data.experiment1$Gaussrunningrhand1,data.experiment1$Gaussrunningrhand2,data.experiment1$Gaussrunninglfoot1,data.experiment1$Gaussrunninglfoot2)
Gaussboxingvec <- c(data.experiment1$Gaussboxingrhand1,data.experiment1$Gaussboxingrhand2,data.experiment1$Gaussboxinglfoot1,data.experiment1$Gaussboxinglfoot2)
Gaussall <- c(Gausswalkingvec,Gaussrunningvec,Gaussboxingvec)
#True
Truewalkingvec <- c(data.experiment1$Truewalkingrhand1,data.experiment1$Truewalkingrhand2,data.experiment1$Truewalkinglfoot1,data.experiment1$Truewalkinglfoot2)
Truerunningvec <- c(data.experiment1$Truerunningrhand1,data.experiment1$Truerunningrhand2, data.experiment1$Truerunninglfoot1,data.experiment1$Truerunningrlfoot2)
Trueboxingvec <- c(data.experiment1$Trueboxingrhand1,data.experiment1$Trueboxingrhand2,data.experiment1$Trueboxinglfoot1,data.experiment1$Trueboxinglfoot2)
Trueall <- c(Truewalkingvec,Truerunningvec,Trueboxingvec)

#Z-test
#NFGauss
pwalkNFGauss <- prop.test(c(sum(NFwalkingvec),sum(Gausswalkingvec)),n=c(64,64),alternative="two.sided",correct=FALSE)
prunNFGauss <- prop.test(c(sum(NFrunningvec),sum(Gaussrunningvec)),n=c(64,64),alternative="two.sided",correct=FALSE)
pboxNFGauss <- prop.test(c(sum(NFboxingvec),sum(Gaussboxingvec)),n=c(64,64),alternative="two.sided",correct=FALSE)
#NFTrue
pwalkNFTrue <- prop.test(c(sum(NFwalkingvec),sum(Truewalkingvec)),n=c(64,64),alternative="two.sided",correct=FALSE)
prunNFTrue <- prop.test(c(sum(NFrunningvec),sum(Truerunningvec)),n=c(64,64),alternative="two.sided",correct=FALSE)
pboxNFTrue <- prop.test(c(sum(NFboxingvec),sum(Trueboxingvec)),n=c(64,64),alternative="two.sided",correct=FALSE)
#GaussTrue
pwalkGaussTrue <- prop.test(c(sum(Gausswalkingvec),sum(Truewalkingvec)),n=c(64,64),alternative="two.sided",correct=FALSE)
prunGaussTrue <- prop.test(c(sum(Gaussrunningvec),sum(Truerunningvec)),n=c(64,64),alternative="two.sided",correct=FALSE)
pboxGaussTrue <- prop.test(c(sum(Gaussboxingvec),sum(Trueboxingvec)),n=c(64,64),alternative="two.sided",correct=FALSE)
#All
pallNFGauss <- prop.test(c(sum(NFall),sum(Gaussall)),n=c(3*64,3*64),alternative="two.sided",correct=FALSE)
pallNFTrue <- prop.test(c(sum(NFall),sum(Trueall)),n=c(3*64,3*64),alternative="two.sided",correct=FALSE)
pallGaussTrue <- prop.test(c(sum(Gaussall),sum(Trueall)),n=c(3*64,3*64),alternative="two.sided",correct=FALSE)

pvalues1 <- c(pwalkNFGauss$p.value,prunNFGauss$p.value,pboxNFGauss$p.value,pwalkNFTrue$p.value,prunNFTrue$p.value,pboxNFTrue$p.value,pwalkGaussTrue$p.value,prunGaussTrue$p.value,pboxGaussTrue$p.value,pallNFGauss$p.value,pallNFTrue$p.value,pallGaussTrue$p.value)

#Experiment 2

#Means/Proportions for experiment 2

NFwalking2 <- mean(colMeans(data.experiment2[1:4]))
NFrunning2 <- mean(colMeans(data.experiment2[5:8]))
NFboxing2 <- mean(colMeans(data.experiment2[9:12]))
Gausswalking2 <- mean(colMeans(data.experiment2[13:16]))
Gaussrunning2 <- mean(colMeans(data.experiment2[17:20]))
Gaussboxing2 <- mean(colMeans(data.experiment2[21:24]))
Truewalking2 <- mean(colMeans(data.experiment2[25:28]))
Truerunning2 <- mean(colMeans(data.experiment2[29:32]))
Trueboxing2 <- mean(colMeans(data.experiment2[33:36]))
NF2 <- mean(colMeans(data.experiment2[1:12]))
Gauss2 <- mean(colMeans(data.experiment2[13:24]))
True2 <- mean(colMeans(data.experiment2[25:36]))

Means2 <- c(NFwalking2,NFrunning2,NFboxing2,Gausswalking2,Gaussrunning2,Gaussboxing2,Truewalking2,Truerunning2,Trueboxing2, NF2, Gauss2, True2)


#Vectors
#NF
NFwalkingvec2 <- c(data.experiment2$NFwalkingrhand25,data.experiment2$NFwalkingrhand65,data.experiment2$NFwalkinglfoot45,data.experiment2$NFwalkinglfoot95)
NFrunningvec2 <- c(data.experiment2$NFrunningrhand25,data.experiment2$NFrunningrhand65,data.experiment2$NFrunninglfoot55,data.experiment2$NFrunninglfoot75)
NFboxingvec2 <- c(data.experiment2$NFboxingrhand70,data.experiment2$NFboxingrhand145,data.experiment2$NFboxinglfoot110,data.experiment2$NFboxinglfoot300)
NFall2 <- c(NFwalkingvec2,NFrunningvec2,NFboxingvec2)
#Gauss
Gausswalkingvec2 <- c(data.experiment2$Gausswalkingrhand25,data.experiment2$Gausswalkingrhand65,data.experiment2$Gausswalkinglfoot45,data.experiment2$Gausswalkinglfoot95)
Gaussrunningvec2 <- c(data.experiment2$Gaussrunningrhand25,data.experiment2$Gaussrunningrhand65,data.experiment2$Gaussrunninglfoot55,data.experiment2$Gaussrunninglfoot75)
Gaussboxingvec2 <- c(data.experiment2$Gaussboxingrhand70,data.experiment2$Gaussboxingrhand145,data.experiment2$Gaussboxinglfoot110,data.experiment2$Gaussboxinglfoot300)
Gaussall2 <- c(Gausswalkingvec2,Gaussrunningvec2,Gaussboxingvec2)
#True
Truewalkingvec2 <- c(data.experiment2$Truewalkingrhand25,data.experiment2$Truewalkingrhand65,data.experiment2$Truewalkinglfoot45,data.experiment2$Truewalkinglfoot95)
Truerunningvec2 <- c(data.experiment2$Truerunningrhand25,data.experiment2$Truerunningrhand65,data.experiment2$Truerunninglfoot55,data.experiment2$Truerunninglfoot75)
Trueboxingvec2 <- c(data.experiment2$Trueboxingrhand70,data.experiment2$Trueboxingrhand145,data.experiment2$Trueboxinglfoot110,data.experiment2$Trueboxinglfoot300)
Trueall2 <- c(Truewalkingvec2,Truerunningvec2,Trueboxingvec2)

#Z-tests
#NFGauss
pwalkNFGauss2 <- prop.test(c(sum(NFwalkingvec2),sum(Gausswalkingvec2)),n=c(64,64),alternative="two.sided",correct=FALSE)
prunNFGauss2 <- prop.test(c(sum(NFrunningvec2),sum(Gaussrunningvec2)),n=c(64,64),alternative="two.sided",correct=FALSE)
pboxNFGauss2 <- prop.test(c(sum(NFboxingvec2),sum(Gaussboxingvec2)),n=c(64,64),alternative="two.sided",correct=FALSE)
#NFTrue
pwalkNFTrue2 <- prop.test(c(sum(NFwalkingvec2),sum(Truewalkingvec2)),n=c(64,64),alternative="two.sided",correct=FALSE)
prunNFTrue2 <- prop.test(c(sum(NFrunningvec2),sum(Truerunningvec2)),n=c(64,64),alternative="two.sided",correct=FALSE)
pboxNFTrue2 <- prop.test(c(sum(NFboxingvec2),sum(Trueboxingvec2)),n=c(64,64),alternative="two.sided",correct=FALSE)
#GaussTrue
pwalkGaussTrue2 <- prop.test(c(sum(Gausswalkingvec2),sum(Truewalkingvec2)),n=c(64,64),alternative="two.sided",correct=FALSE)
prunGaussTrue2 <- prop.test(c(sum(Gaussrunningvec2),sum(Truerunningvec2)),n=c(64,64),alternative="two.sided",correct=FALSE)
pboxGaussTrue2 <- prop.test(c(sum(Gaussrunningvec2),sum(Truerunningvec2)),n=c(64,64),alternative="two.sided",correct=FALSE)
#all
pallNFGauss2 <- prop.test(c(sum(NFall2),sum(Gaussall2)),n=c(3*64,3*64),alternative="two.sided",correct=FALSE)
pallNFTrue2 <- prop.test(c(sum(NFall2),sum(Trueall2)),n=c(3*64,3*64),alternative="two.sided",correct=FALSE)
pallGaussTrue2 <- prop.test(c(sum(Gaussall2),sum(Trueall2)),n=c(3*64,3*64),alternative="two.sided",correct=FALSE)

pvalues2 <- c(pwalkNFGauss2$p.value,prunNFGauss2$p.value,pboxNFGauss2$p.value,pwalkNFTrue2$p.value,prunNFTrue2$p.value,pboxNFTrue2$p.value,pwalkGaussTrue2$p.value,prunGaussTrue2$p.value,pboxGaussTrue2$p.value,pallNFGauss2$p.value,pallNFTrue2$p.value,pallGaussTrue2$p.value)

pvaluesboth <- c(pvalues1,pvalues2)

#P value correction

p.adjust(pvaluesboth,method="BH")


