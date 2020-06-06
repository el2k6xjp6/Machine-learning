import numpy as np
import NaiveBayes
import others

(train_image,train_label,test_image,test_label)=others.ReadTrainingData()
for i in range(2):
    NB = NaiveBayes.NaiveBayes(train_image,train_label,test_image,test_label,32)
    NB.run(i)
    input("")
