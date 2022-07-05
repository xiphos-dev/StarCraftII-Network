#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np
import pickle


ruta = "./"
archivo = "midgame_con_puntero_corregido_y_tiempo.csv"
file = ruta+archivo


df = pd.read_csv(file, sep=',', dtype={"Label": str})
#df.head(20)


# ##  Preparando dataset

# In[5]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# In[6]:


valores = ["1 base no tech", "2 base no tech", "3 base no tech"]
estructuras = ["Pylon",
               "Gateway",
               "WarpGate",
               "Battery",
               "Assimilator",
               "Nexus",
               "CyberneticsCore",
               "RoboticsFacility",
               "RoboticsBay",
               "TemplarArchives",
               "DarkShrine",
               "Forge",
               "TwilightCouncil",
               "Stargate",
               "FleetBeacon",
               "PhotonCannon"]

estructuras_tiempo = ["Pylon_tiempo",
               "Gateway_tiempo",
               "Battery_tiempo",
               "Assimilator_tiempo",
               "Nexus_tiempo",
               "CyberneticsCore_tiempo",
               "RoboticsFacility_tiempo",
               "RoboticsBay_tiempo",
               "TemplarArchives_tiempo",
               "DarkShrine_tiempo",
               "Forge_tiempo",
               "TwilightCouncil_tiempo",
               "Stargate_tiempo",
               "FleetBeacon_tiempo",
               "PhotonCannon_tiempo"]

unidades = ["Zealot",
           "Probe",
           "Adept",
           "Stalker",
           "Sentry",
           "HighTemplar",
           "DarkTemplar",
           "VoidRay",
           "Phoenix",
           "Oracle",
           "Carrier",
           "Tempest",
           "Archon",
           "WarpPrism",
           "Colossus",
           "Disruptor",
           "Immortal",
           "Observer"]

unidades_tiempo = ["Zealot_tiempo",
           "Probe_tiempo",
           "Adept_tiempo",
           "Stalker_tiempo",
           "Sentry_tiempo",
           "HighTemplar_tiempo",
           "DarkTemplar_tiempo",
           "VoidRay_tiempo",
           "Phoenix_tiempo",
           "Oracle_tiempo",
           "Carrier_tiempo",
           "Tempes_tiempot",
           "Archon_tiempo",
           "WarpPrism_tiempo",
           "Colossus_tiempo",
           "Disruptor_tiempo",
           "Immortal_tiempo",
           "Observer_tiempo"]

mejoras = [ 
            2208,  # ResearchFluxVanes
            7648,  # Charge
            # 5824, # GroundWeapons
            5760,  # NewGroundWeapons
            5761,  # NewGroundWeapons2
            5762,  # NewGroundWeapons3
            7552,  # AirWeapons1
            7553,  # AirWeapons2
            7554,  # AirWeapons3
            7555,  # ResearchAirArmor 1
            7556,  # ResearchAirArmor 2
            7557,  # ResearchAirArmor 3
            # 7622, # ResearchWarpGate
            7558,  # NuevoResearchWarpGate
            5892,  # ResearchStorm
            7585,  # ResearchBlink
            7586,  # ResearchGlaives se llama 'AdeptPiercingAttack'
            5794,  # ResearchGraviticDrive
            5797,  # ResearchExtendedThermalLance
          ]

estructuras_reducido = [
               "Pylon",
               "Gateway",
               "Assimilator",
               "Nexus",
               "RoboticsFacility",
               "RoboticsBay",
               "DarkShrine",
               "TwilightCouncil",
               "Stargate",
               "FleetBeacon"]

builds_objetivo = [
    
    "2 base DT",
    "2 base Glaives",
    "2 base Robo",
    "2 base Twilight",
    "2 base Stargate",
    "3 base DT",
    "3 base Fleet Beacon",
    "3 base Glaives",
    "3 base RoboBay",
    "3 base Stargate",
    "3 base Twilight"
]

mapeo_numero_estructura = {
    
    "Pylon": 0,
    "Gateway": 1,
    "Assimilator": 2,
    #"Nexus": 3,
    #"RoboticsFacility": 4,
    #"RoboticsBay": 5,
    #"DarkShrine": 6,
    #"TwilightCouncil": 7,
    #"Stargate": 8,
    #"FleetBeacon": 9,


}
    


# In[8]:


df = df[df["Label"].isin(builds_objetivo)]
X = df.drop(["Label"], axis=1).drop(estructuras_tiempo, axis=1)
y = df["Label"]


# In[9]:


def biyeccionCoordenadas(x,y):
    x_biyeccion = 2*x
    y_biyeccion = 2*y
    return x_biyeccion, y_biyeccion


# In[10]:


import math

def ubicarEstructura(dataset, nfila, fila, estructura):
    parche =  ["Pylon","Gateway","Assimilator"]
   # parche =  ["Assimilator"]
    if estructura not in parche:#medida de parche para solo tratar con 3 estructuras por ahora
        return dataset
    canal = mapeo_numero_estructura[estructura]
    #print(fila.axes[0][5])
    columnas_elegidas_x = [col for col in fila.axes[0] if estructura+"_x" in col]
    columnas_elegidas_y = [col for col in fila.axes[0] if estructura+"_y" in col]
    
    #print(columnas_elegidas_x)
    #print(columnas_elegidas_y)
    
    i = 0
    for campo in columnas_elegidas_x:
        campo_y = columnas_elegidas_y[i]
        i+=1
        
        x = float(fila[campo])
        y = float(fila[campo_y])
    
        x,y = biyeccionCoordenadas(x,y)
        x = int(x)
        y = int(y)
        #print("Fila:"+str(nfila))
        #print(x)
        #print(y)
        #print(campo)
        if x != 0 and y != 0:
            if dataset[nfila][canal][y][x] != 0: #la celda se encuentra ocupada por otra estructura
                print("Not possible")
                print("Estructura:"+estructura)
                print("Replay:"+str(fila["Replay"]))
                print(x)
                print(y)
                print("*"*10)
            dataset[nfila][canal][y][x] = int(canal)
        #print("-"*20 + campo)
    return dataset
            

def transformarFilaEnInputConvolucional(df):
    canales = len(mapeo_numero_estructura.keys())
    ancho = 340 #170*2
    alto = 340 #150*2 - ahora con padding para que sea 340*340
    total = df.shape[0]
    dataset = np.zeros((total,canales,alto,ancho))
    df = df.reset_index()
    for index, row in df.iterrows():
        #print(index)
        for estructura in estructuras_reducido:
            dataset = ubicarEstructura(dataset,index,row,estructura)
    return dataset


def fixAssimilators(df):
    df = df.reset_index()
    estructura="Assimilator"
    for index, row in df.iterrows():
        print(index)
        tablero = np.zeros((300,340))
        columnas_elegidas_x = [col for col in row.axes[0] if estructura+"_x" in col]
        columnas_elegidas_y = [col for col in row.axes[0] if estructura+"_y" in col]
        
        i = 0    
        for campo in columnas_elegidas_x:
            campo_y = columnas_elegidas_y[i]
            i+=1

            x = float(row[campo])
            y = float(row[campo_y])
            x,y = biyeccionCoordenadas(x,y)
            x = int(x)
            y = int(y)
            
            if x != 0 and y != 0:
                if tablero[y][x] != 0: #la celda se encuentra ocupada por otra estructura
                    print("Fixing")
                    #print(x)
                    #print(y)
                    print("*"*10)
                    df.iloc[index][campo] = 0
                    df.iloc[index][campo_y] = 0
                else:
                    tablero[y][x] = 2
    return df


dataset = transformarFilaEnInputConvolucional(X)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoder_mapeo = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


categorias = dummy_y.shape[1]


params = encoder.get_params(y)


def divisionTestTrain(dataset, y,porcion=0.7):
    y = y.reset_index()
    posiciones = np.arange(len(dataset))
    rng = np.random.default_rng()
    orden_final = rng.permuted(posiciones) #tengo que guardar este arreglo en memoria para recordar el orden del shuffle
    #lo anterior solo es cierto si es que no hay una manera de garantizar el mismo resultado al mezclar el orden de la lista de enteros
    limite_train = len(dataset)*porcion
    
    #si la cota de ejemplos de entrenamiento es un decimal, hay que transformarlo en un entero para que cumpla su funcion como slicer
    if not limite_train.is_integer():
        limite_train = math.floor(limite_train)
    limite_train = int(limite_train) 
    
    total, canales, alto, ancho = dataset.shape
    train_set = np.zeros((limite_train, canales, alto, ancho))
    y_train = []
    test_set = np.zeros((total-limite_train, canales, alto, ancho))
    y_test = []
    contador_n_test = 0
    contador_n_total = 0
    flag_test = False
    for posicion in orden_final:
        if flag_test:
            test_set[contador_n_test] = dataset[posicion]
            y_test.append(y.loc[posicion]["Label"])
            contador_n_test+=1
        else:
            train_set[contador_n_total] = dataset[posicion]
            y_train.append(y.loc[posicion]["Label"])
            if contador_n_total >= limite_train-1:
                flag_test = True
        contador_n_total+=1
        
    return train_set,np.array(y_train),test_set,np.array(y_test)
        
#y = y.reset_index()
#dataset = dataset.reset_index()


X_train, X_test, y_train, y_test = train_test_split(dataset,dummy_y,test_size=0.20, random_state=100)

import pickle
import os

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


ruta = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"X_train")

with open(ruta,"wb") as arc:
    pickle.dump(X_train,arc)

ruta = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"X_test")

with open(ruta,"wb") as arc:
    pickle.dump(X_test,arc)

ruta = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"y_train")
with open(ruta,"wb") as arc:
    pickle.dump(y_train,arc)

ruta = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"y_test")
with open(ruta,"wb") as arc:
    pickle.dump(y_test,arc)
