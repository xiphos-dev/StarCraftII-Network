import os
import pickle
import numpy as np
import bz2

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

with open("encoder","rb") as arc:
    encoder = pickle.load(arc)

def producirDatosMatriz(predicciones, y_test):
    
    y_pred = np.argmax(predicciones, axis=-1)

    y_true=np.argmax(y_test, axis=-1)

    cm = confusion_matrix(y_true, y_pred)
    
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    return precision, recall, f1, accuracy, cm

from tensorflow import keras
ruta_modelo = "./Modelo88f_no_dropout_sinpadding_t"
modelo =  keras.models.load_model(ruta_modelo)

numero_secciones_data = 1

for numero in range(1,numero_secciones_data+1):
    numero_str = str(numero)
    with open("y_test_"+numero_str,"rb") as arc:
        y_test = pickle.load(arc)

    ifile = bz2.BZ2File("X_test_"+numero_str,"rb")
    X_test = pickle.load(ifile)
    ifile.close()

    predicciones = modelo.predict(X_test)

    precision, recall, f1, acc, cm = producirDatosMatriz(predicciones, y_test)

    estado = {
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
        "fscore": f1,
        "cm": cm,
        "clases": encoder.classes_,
        "y_pred": predicciones,
        "y_test": y_test
    }

    with open('./Validaciones/metricas_'+numero_str, 'wb') as file_pi:
        pickle.dump(estado, file_pi)
