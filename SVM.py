from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import neighbors
# from sklearn import linear_model
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score

# import data
originalData = np.loadtxt("DATA_val_truq.csv", delimiter=",") 
noRep = 1
f1MacroAvg = 0
confusionMatrixAvg = 0

for cont in range(0,noRep):

	data = np.random.permutation(originalData) 

	d = int(len(data)*.75)
	datatrain = data[0:d,:]
	datavalidation = data[d:,:]
	x = datatrain[0:,0:1]
	y = datatrain[0:,-1]
	# print (x)
	# print (y)

	# Run classifierb
	# classifier = svm.SVC(kernel='linear')  #0
	# classifier = svm.SVC(kernel='rbf')  #40 
	# classifier = svm.SVC(degree=3, gamma='auto', kernel='poly', C=1) #//tarda mucho
	# classifier = svm.LinearSVC(C=1)
	# classifier = MLPClassifier(hidden_layer_sizes=(10,))

	# SVM
	# classifier = svm.SVC(kernel='rbf') 

	# Neural Network
	# classifier = MLPClassifier(hidden_layer_sizes=(100,), solver ='sgd', learning_rate='invscaling')

	# Decision Tree Classifier
	# classifier = tree.DecisionTreeClassifier()

	# Gradient Boosting Classifier
	# classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)

	# Gaussian Process Filter (NO)
	# kernel = 1.0 * RBF([1.0])
	# classifier = GaussianProcessClassifier(kernel=kernel)

	# NN
	n_neighbors = 15
	classifier = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

	# Passive Aggressive Classifier
	# classifier = PassiveAggressiveClassifier()


	# 
	classifier.fit(x,y)
	resultado = classifier.predict(datavalidation[0:,0:1])
	#print resultado

	# f1_score
	# f1_Score = f1_score(datavalidation[0:,1], resultado, average='macro')  
	f1_Score = f1_score(datavalidation[0:,1], resultado, average=None)  
	f1MacroAvg += f1_Score
	# print ("f1_score macro   ", f1_score(datavalidation[0:,1], resultado, average='macro')*100, "%")
	# print ("f1_score micro   ", f1_score(datavalidation[0:,1], resultado, average='micro')*100, "%")
	# print ("f1_score weighted", f1_score(datavalidation[0:,1], resultado, average='weighted')*100, "%")
	# print ("f1_score None    ", f1_score(datavalidation[0:,1], resultado, average=None)*100, "%")
	# print ("sldjfls ", int(len(datavalidation)) )



	# Compute confusion matrix
	cM = confusion_matrix(datavalidation[0:,-1], resultado)

	# print(cM)

	acierto = 0
	error = 0
	for i in range(0,len(datavalidation)):
	    if datavalidation[i,-1]==resultado[i]:
	        acierto=acierto+1
	    elif datavalidation[i,-1]!=resultado[i]:
	        error=error+1

	promedio = float(acierto) / (acierto + error)
	confusionMatrixAvg += promedio
	# print ("acierto ", acierto, " , error ", error)
	# print ("Promedio", promedio*100, "%")
	# print ("f1_score", f1_Score*100, "%")



	# Grafica de Matriz de Confusion
	# plt.matshow(cm)
	# plt.title('Matriz de Confusion')
	# plt.colorbar()
	# plt.ylabel('Valor Real')
	# plt.xlabel('Valor Predicho')
	# plt.show()
f1MacroAvg = f1MacroAvg/noRep
confusionMatrixAvg = confusionMatrixAvg/noRep

print ("Promedio en ", noRep, " repeticiones: ")
print ("f1_score macro: ", f1MacroAvg)
print ("matriz de confusion: ", confusionMatrixAvg)
