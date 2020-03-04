import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.svm as svm
import sklearn

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
x_80, x_final, y_80, y_final = train_test_split(data, target, train_size=0.8)
x_train, x_test, y_train, y_test = train_test_split(x_80, y_80, train_size=0.5)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_final = scaler.transform(x_final)

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

proyeccion_train = np.dot(x_train,vectores)
proyeccion_test = np.dot(x_test,vectores)
proyeccion_final = np.dot(x_final,vectores)

c = np.logspace(-2,1,100)
f1 = []
for i in c:
    svc = svm.SVC(i,kernel = 'linear')
    svc.fit(proyeccion_train[:,:10], y_train.T)
    prediccion = svc.predict(proyeccion_test[:,:10])
    f1s = sklearn.metrics.f1_score(y_test,prediccion,average = 'macro')
    f1.append(f1s)
    
c_max = c[np.where(f1 == np.max(f1))]

svc = sklearn.svm.SVC(c_max,kernel = 'linear')
svc.fit(proyeccion_train[:,:10], y_train.T)
sklearn.metrics.plot_confusion_matrix(svc,proyeccion_final[:,:10],y_final,normalize = 'true')
plt.title('Matriz de confusión para c = {:.3f}'.format(c_max[0]))
plt.savefig('Conf_matr.png')

