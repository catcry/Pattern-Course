import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import losses
from keras import optimizers
from funz import num_of
from funz import rnd
from loadfunc import load

train=load("KDDTrain+.csv")
test=load("KDDTest+.csv")
test21= load ("KDDTest-21.csv")

#===== Seperate Validation Set from KDDTrain+ Set
from sklearn.model_selection import train_test_split
Xtrain,Xvalid,Ytrain,Yvalid = train_test_split (train[0],train[1], stratify=train[1],test_size=0.1)


Xtrain = Xtrain.reshape((len(Xtrain),1,41))
Xvalid = Xvalid.reshape((len(Xvalid),1,41))

train0=train[0]
test0=test[0]
test210=test21[0]

#============Reshape=====
train0=train0.reshape((len(train[0]),1,41))
test0=test0.reshape((len(test[0]),1,41))
test210=test210.reshape((len(test21[0]),1,41))


#====RNN Model Making========
model= Sequential()
model.add(LSTM(80,input_shape = (Xtrain.shape[1], Xtrain.shape[2])))
model.add(Dense(1, activation = 'tanh'))
sgd = optimizers.SGD(lr=0.008, decay = 1e-6, momentum = 0.9, nesterov=True)
model.compile (loss='mae', optimizer=sgd)
loss_Spec = "model Loss = mae , Optimizer = sgd"
#=====RNN Training and Validation======
history = model.fit (Xtrain,Ytrain, epochs = 50, batch_size=10000, validation_data=(Xvalid,Yvalid), verbose=2, shuffle=False)
hist = "model.fit (Xtrain,Ytrain, epochs = 100, batch_size=10000, validation_data=(Xvalid,Yvalid), verbose=2, shuffle=False)"

yh_Xtrain = model.predict(Xtrain)
yh_Xtrain = rnd(yh_Xtrain)
yh_Xtrain = yh_Xtrain.reshape ((len(yh_Xtrain)))
Xtrain_error = Ytrain - yh_Xtrain
Xtrain_fn = num_of (1,Xtrain_error)
Xtrain_fp = num_of(-1,Xtrain_error)
Xtrain_acc = 1-((Xtrain_fn+Xtrain_fp)/len(Xtrain))
print ("======================Results of XTrain Set (90% of KDDTrain+ Set): ===================================")
print()
print("The Specification of Classifier is:",hist)
print(loss_Spec)
print ("Number of un-Detected Attacks (FN) in Training Set is: FN = ",Xtrain_fn," out of",len(Xtrain))
print ("Number of False Attack Alarms (FP) in Training Set is: FP =",Xtrain_fp," out of",len(Xtrain))
print ("Accuracy on Trainin Set: Accuracy = ",Xtrain_acc,"%")
print()

yh_train = model.predict(train0)
yh_train = rnd(yh_train)
yh_train = yh_train.reshape ((len(yh_train)))
train_error = train[1] - yh_train
train_fn = num_of (1,train_error)
train_fp = num_of(-1,train_error)
train_acc = 1-((train_fn+train_fp)/len(train0))

print ("======================Results of KDDTrain+ Set: ===================================")
print()
print("The Specification of Classifier is:",hist)
print ("Number of un-Detected Attacks (FN) in Training Set is: FN = ",train_fn," out of",len(train0))
print ("Number of False Attack Alarms (FP) in Training Set is: FP =",train_fp," out of",len(train0))
print ("Accuracy on Trainin Set: Accuracy = ",train_acc,"%")
print()

yh_test = model.predict(test0)
yh_test = rnd (yh_test)
yh_test = yh_test.reshape((len(yh_test)))
test_error = test[1] - yh_test
test_fn = num_of (1,test_error)
test_fp = num_of(-1,test_error)
test_acc = 1-((test_fn+test_fp)/len(test0))
print ("======================Results of KDDTest+ Set: ===================================")
print()
print("The Specification of Classifier is:",history)
print ("Number of un-Detected Attacks (FN) in Training Set is: FN = ",test_fn," out of",len(test0))
print ("Number of False Attack Alarms (FP) in Training Set is: FP =",test_fp," out of",len(test0))
print ("Accuracy on Trainin Set: Accuracy = ",test_acc,"%")
print()
                        
yh_test21 = model.predict(test210)
yh_test21 = rnd (yh_test21)
yh_test21 = yh_test21.reshape((len(yh_test21)))
test21_error = test21[1] - yh_test21
test21_fn = num_of (1,test21_error)
test21_fp = num_of(-1,test21_error)
test21_acc = 1-((test21_fn+test21_fp)/len(test210))
print ("======================Results of KDDTest-21 Set: ===================================")
print()
print("The Specification of Classifier is:",history)
print ("Number of un-Detected Attacks (FN) in Training Set is: FN = ",test21_fn," out of",len(test210))
print ("Number of False Attack Alarms (FP) in Training Set is: FP =",test21_fp," out of",len(test210))
print ("Accuracy on Trainin Set: Accuracy = ",test21_acc,"%")
print()
