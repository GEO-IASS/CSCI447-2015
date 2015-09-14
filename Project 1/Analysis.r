data = read.csv("results.csv")

par(mfrow=c(1,1))
plot(data$Data_Set, data$Algorithm, main="Data Sets vs Algorithms")
plot(data$Algorithm, data$Square_Loss, main="Algorithm vs Squared Loss")
plot(data$Algorithm, data$Hinge_Loss, main="Algorithm vs Hinge Loss")
plot(data$Algorithm, data$Log_Loss, main="Algorithm vs Log Loss")


DaterList <- list("ArtificialCharacters", "Car", "Cmc", "Connect-4", "Ecoli", "Letter-Recognition", "Tic-Tac-Toe", "Wine", "Yeast", "Zoo")
#AlgList <- list("ADABOOSTM1", "IB1", "IBK", "J48", "JRIP", "LOGISTIC", "MULTILAYERPERCEPTRON", "NAIVEBAYES", "RBFNETWORK", "SMO")
AlgList <- list("Decision Tree", "Ensemble", "Feedforward Neural Networks", "K-Nearest Neighbor", "Kernel Neural Network", "Logistic", "Naive Bayes", "Ripper", "Simple Nearest Neighbor", "Support Vector Machine")
plotF<-function(opt){
  data.temp<-data[data$Algorithm==opt,]
  plot(data.temp$Data_Set, data.temp$Square_Loss, main=paste("Lattice with Algorithm", toString(opt)))
  plot(data.temp$Data_Set, data.temp$Hinge_Loss, main=paste("Lattice with Algorithm", toString(opt)))
  plot(data.temp$Data_Set, data.temp$Los_Loss, main=paste("Lattice with Algorithm", toString(opt)))
}

par(mar=c(1,1,1,1))
par(mfrow=c(10,3))
for (i in AlgList){
  plotF(i)
}







data = read.csv("results.csv")
data$Percent = data$Number_Correct/data$Number_Total*100
DaterList <- list("ArtificialCharacters", "Car", "Cmc", "Connect-4", "Ecoli", "Letter-Recognition", "Tic-Tac-Toe", "Wine", "Yeast", "Zoo")
#AlgList <- list("ADABOOSTM1", "IB1", "IBK", "J48", "JRIP", "LOGISTIC", "MULTILAYERPERCEPTRON", "NAIVEBAYES", "RBFNETWORK", "SMO")
AlgList <- list("Decision Tree", "Ensemble", "Feedforward Neural Networks", "K-Nearest Neighbor", "Kernel Neural Network", "Logistic", "Naive Bayes", "Ripper", "Simple Nearest Neighbor", "Support Vector Machine")
#AlgListMini <- list("AB", "N1", "NK", "J4", "JR", "LT", "MP", "NB", "RB", "SM")
AlgListMini <- list("J4", "AB", "MP", "NK", "RB", "LT", "NB", "JR", "N1", "SM")
DaterListMini <- list("AC", "CR", "CM", "C4", "EC", "LR", "T3", "WI", "YS", "ZO")

pdf("Plots/square.pdf")
plotF<-function(opt){
  data.temp<-data[data$Algorithm==opt,]
  plot(data.temp$Data_Set, data.temp$Square_Loss, xaxt = "n", main=paste(toString(opt)))
  axis(1, at=1:10, labels=DaterListMini)
}
par(mar=c(2.5,2,1.5,1))
par(mfrow=c(5,2),oma=c(0,0,4,0))
for (i in AlgList){
  plotF(i)
}
title("Square Loss for Various Classification Algorithms", outer=TRUE, cex.main=2)
dev.off()

pdf("Plots/hinge.pdf")
plotF<-function(opt){
  data.temp<-data[data$Algorithm==opt,]
  plot(data.temp$Data_Set, data.temp$Hinge_Loss,  xaxt = "n", main=paste(toString(opt)))
  axis(1, at=1:10, labels=DaterListMini)
}

par(mar=c(2.5,2,1.5,1))
par(mfrow=c(5,2),oma=c(0,0,4,0))
for (i in AlgList){
  plotF(i)
}
title("Hinge Loss for Various Classification Algorithms", outer=TRUE, cex.main=2)
dev.off()

pdf("Plots/log.pdf")
plotF<-function(opt){
  data.temp<-data[data$Algorithm==opt,]
  plot(data.temp$Data_Set, data.temp$Log_Loss/data.temp$Number_Total, xaxt = "n", main=paste(toString(opt)))
  axis(1, at=1:10, labels=DaterListMini)
}
par(mar=c(2.5,2,1.5,1))
par(mfrow=c(5,2),oma=c(0,0,4,0))
for (i in AlgList){
  plotF(i)
}
title("Log Loss for Various Classification Algorithms", outer=TRUE, cex.main=2)
dev.off()
















pdf("Plots/percentage.pdf")
plotF<-function(opt){
  data.temp<-data[data$Algorithm==opt,]
  plot(data.temp$Data_Set, data.temp$Percent, main=paste(toString(opt)))
}
par(mar=c(2.5,2,1.5,1))
par(mfrow=c(5,2),oma=c(0,0,4,0))
for (i in AlgList){
  plotF(i)
}
title("Square Loss", outer=TRUE)
dev.off()

DaterList <- list("ArtificialCharacters", "Car", "Cmc", "Connect-4", "Ecoli", "Letter-Recognition", "Tic-Tac-Toe", "Wine", "Yeast", "Zoo")
AlgList <- list("ADABOOSTM1", "IB1", "IBK", "J48", "JRIP", "LOGISTIC", "MULTILAYERPERCEPTRON", "NAIVEBAYES", "RBFNETWORK", "SMO")


# Keep this. We like it. -----
pdf("plot.pdf", width=7.5, height=10)
data = read.csv("results.csv")
DaterList <- list("ArtificialCharacters", "Car", "Cmc", "Connect-4", "Ecoli", "Letter-Recognition", "Tic-Tac-Toe", "Wine", "Yeast", "Zoo")
AlgList <- list("ADABOOSTM1", "IB1", "IBK", "J48", "JRIP", "LOGISTIC", "MULTILAYERPERCEPTRON", "NAIVEBAYES", "RBFNETWORK", "SMO")
AlgListMini <- list("ADB", "IB1", "IBK", "J48", "RIP", "LOG", "MP", "NB", "RBFN", "SMO")
plotF<-function(opt){
  data.temp<-data[data$Data_Set==opt,]
  plot(data.temp$Algorithm, data.temp$Square_Loss, xaxt = "n", cex.lab=.2, main=paste(toString(opt)))
  axis(1, at=1:10, labels=AlgListMini)
  plot(data.temp$Algorithm, data.temp$Hinge_Loss, xaxt = "n", cex.lab=.2, main=paste(toString(opt)))
  axis(1, at=1:10, labels=AlgListMini)
  plot(data.temp$Algorithm, data.temp$Los_Loss, xaxt = "n", cex.lab=.2, main=paste(toString(opt)))
  axis(1, at=1:10, labels=AlgListMini)
}
par(mar=c(2.1,2,1.1,1))
par(mfrow=c(10,3))
for (i in DaterList){
  plotF(i)
}
dev.off()
# ----------------------------


data = read.csv("results.csv")
DaterList <- list("ArtificialCharacters", "Car", "Cmc", "Connect-4", "Ecoli", "Letter-Recognition", "Tic-Tac-Toe", "Wine", "Yeast", "Zoo")
DaterListMini <- list("AC", "CR", "CM", "C4", "EC", "LR", "T3", "WI", "YS", "ZO")
AlgList <- list("ADABOOSTM1", "IB1", "IBK", "J48", "JRIP", "LOGISTIC", "MULTILAYERPERCEPTRON", "NAIVEBAYES", "RBFNETWORK", "SMO")
AlgListMini <- list("AB", "N1", "NK", "J4", "JR", "LT", "MP", "NB", "RB", "SM")
plotF<-function(opt){
  data.temp<-data[data$Algorithm==opt,]
  plot(data.temp$Data_Set, data.temp$Square_Loss, xaxt = "n", cex.lab=.2, main=paste(toString(opt)))
  axis(1, at=1:10, labels=DaterListMini)
  plot(data.temp$Data_Set, data.temp$Hinge_Loss, xaxt = "n", cex.lab=.2, main=paste(toString(opt)))
  axis(1, at=1:10, labels=DaterListMini)
  plot(data.temp$Data_Set, data.temp$Los_Loss, xaxt = "n", cex.lab=.2, main=paste(toString(opt)))
  axis(1, at=1:10, labels=DaterListMini)
}
par(mar=c(2.1,2,1.1,1))
par(mfrow=c(10,3))
for (i in AlgList){
  plotF(i)
}

data.log=data[data$Algorithm=='LOGISTIC',]
par(mfrow=c(1,1))
plot(data.log$Data_Set, data.log$Square_Loss)

