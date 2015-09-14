data = read.csv("results.csv")

par(mfrow=c(1,1))
plot(data$Data_Set, data$Algorithm, main="Data Sets vs Algorithms")
plot(data$Algorithm, data$Square_Loss, main="Algorithm vs Squared Loss")
plot(data$Algorithm, data$Hinge_Loss, main="Algorithm vs Hinge Loss")
plot(data$Algorithm, data$Log_Loss, main="Algorithm vs Log Loss")


DaterList <- list("ArtificialCharacters", "Car", "Cmc", "Connect-4", "Ecoli", "Letter-Recognition", "Tic-Tac-Toe", "Wine", "Yeast", "Zoo")
AlgList <- list("ADABOOSTM1", "IB1", "IBK", "J48", "JRIP", "LOGISTIC", "MULTILAYERPERCEPTRON", "NAIVEBAYES", "RBFNETWORK", "SMO")
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

AlgList <- list("ADABOOSTM1", "IB1", "IBK", "J48", "JRIP", "LOGISTIC", "MULTILAYERPERCEPTRON", "NAIVEBAYES", "RBFNETWORK", "SMO")
plotF<-function(opt){
  data.temp<-data[data$Algorithm==opt,]
  plot(data.temp$Data_Set, data.temp$Square_Loss, main=paste("Lattice with Algorithm", toString(opt)))
}

par(mar=c(1,1,1,1))
par(mfrow=c(5,2))
for (i in AlgList){
  plotF(i)
}

DaterList <- list("ArtificialCharacters", "Car", "Cmc", "Connect-4", "Ecoli", "Letter-Recognition", "Tic-Tac-Toe", "Wine", "Yeast", "Zoo")
plotF<-function(opt){
  data.temp<-data[data$Data_Set==opt,]
  plot(data.temp$Algorithm, data.temp$Number_Correct/data.temp$Number_Total, main=paste("Lattice with Algorithm", toString(opt)))
}

par(mar=c(1,1,1,1))
par(mfrow=c(5,2))
for (i in DaterList){
  plotF(i)
}


# Keep this. We like it. -----
DaterList <- list("ArtificialCharacters", "Car", "Cmc", "Connect-4", "Ecoli", "Letter-Recognition", "Tic-Tac-Toe", "Wine", "Yeast", "Zoo")
AlgList <- list("ADABOOSTM1", "IB1", "IBK", "J48", "JRIP", "LOGISTIC", "MULTILAYERPERCEPTRON", "NAIVEBAYES", "RBFNETWORK", "SMO")
plotF<-function(opt){
  data.temp<-data[data$Data_Set==opt,]
  plot(data.temp$Algorithm, data.temp$Square_Loss, main=paste("Lattice with Algorithm", toString(opt)))
  plot(data.temp$Algorithm, data.temp$Hinge_Loss, main=paste("Lattice with Algorithm", toString(opt)))
  plot(data.temp$Algorithm, data.temp$Los_Loss, main=paste("Lattice with Algorithm", toString(opt)))
}

par(mar=c(1,1,1,1))
par(mfrow=c(10,3))
for (i in DaterList){
  plotF(i)
}
# ----------------------------

