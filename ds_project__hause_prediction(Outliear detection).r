
train <- read.csv('.../train-mobiiina199.csv')

dim(train)

library(mice)
library(VIM)

summary(train)

mf<-lm(SalePrice~.,data=train)
plot(mf$fitted.values, mf$residuals,xlab="Fitted Values",ylab="Residuals")
abline(h=mean(mf$residuals),col="red")

library(faraway)

halfnorm(influence(mf)$hat, nlab = 1, ylab="Leverages")

resid <- residuals(mf)
sigma <- summary(mf)$sigma
hi <- influence(mf)$hat

stud.res <- resid/(sigma * sqrt(1-hi))
plot(stud.res, fitted.values(mf),ylab="Fitted Values",xlab="StudentizedResiduals")

#high leverage points
HLV=which(hi>0.8)


#  High StudentizedResiduals points
HSR=which(stud.res>(3))

HSR

train<-train[-c(HSR),]

dim(train)

write.csv(train,'... /train-mobina_final.csv')


