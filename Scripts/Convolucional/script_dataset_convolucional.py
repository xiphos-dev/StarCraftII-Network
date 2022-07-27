#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np


# In[2]:


ruta = "../../"
archivo = "interloper_protoss.csv"
file = ruta+archivo



df = pd.read_csv(file, sep=',', dtype={"Label": str})
#df.head(20)



ruta = "../../"
archivo = "extension_dataset_p_interloper.csv"
file = ruta+archivo


df2 = pd.read_csv(file, sep=',', dtype={"Label": str})

df = df.append(df2, ignore_index = True)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# In[6]:


valores = ["1 base no tech", "2 base no tech", "3 base no tech"]

estructuras = ["Pylon",
               "Gateway",
               "Assimilator",
               "Nexus",
               "CyberneticsCore",
               "RoboticsFacility",
               "RoboticsBay",
               "TemplarArchive",
               "DarkShrine",
               "Forge",
               "TwilightCouncil",
               "Stargate",
               "FleetBeacon",
               "PhotonCannon"]

estructuras_tiempo = ["Pylon_tiempo",
               "Gateway_tiempo",
               "Assimilator_tiempo",
               "Nexus_tiempo",
               "CyberneticsCore_tiempo",
               "RoboticsFacility_tiempo",
               "RoboticsBay_tiempo",
               "TemplarArchive_tiempo",
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

mapeo_unidades_tiempo ={unidad:numero for numero,unidad in enumerate(unidades,100)}
#print(mapeo_unidades_tiempo.items())

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
           "Tempest_tiempo",
           "Archon_tiempo",
           "WarpPrism_tiempo",
           "Colossus_tiempo",
           "Disruptor_tiempo",
           "Immortal_tiempo",
           "Observer_tiempo"]

mejoras = [ 
            "ResearchFluxVanes",
            "ResearchCharge",
            "ResearchPsiStormTech",
            "ResearchBlink",
            "ResearchAdeptPiercingAttack",
            "ResearchGraviticDrive",
            "ResearchExtendedThermalLance",
            "ResearchWarpGate",
            "ResearchAnionPulseCrystals",
            "ResearchGraviticBoosters",
            "UpgradeGroundWeapons1",
            "UpgradeGroundWeapons2",
            "UpgradeGroundWeapons3",
            "UpgradeGroundArmor1",
            "UpgradeGroundArmor2",
            "UpgradeGroundArmor3",
            "UpgradeAirWeapons1",
            "UpgradeAirWeapons2",
            "UpgradeAirWeapons3",
            "UpgradeAirArmor1",
            "UpgradeAirArmor2",
            "UpgradeAirArmor3",
            "UpgradeShields1",
            "UpgradeShields2",
            'UpgradesShields3',
            'UpgradeToMothership',
            'CancelUpgradeToMothership',
            'ResearchDarkTemplarBlinkUpgrade'
          ]

mapeo_mejora_numero = {mejora: numero for numero,mejora in enumerate(mejoras,50)}


# In[19]:


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

builds_poco_presentes = [
    "1 base Blink",
    "1 base Glaives",
    "1 base Blink 7+Gates",
    "1 base Fleet Beacon",
    "1 base Glaives 7+Gates",
    "1 base RoboBay",
    "1 base Twilight 7+Gates",
    "2 base Blink 7+Gates",
    "3 base",
    "3 base +1 ground",
]

mapeo_numero_estructura = {
    
    "Pylon": 1,
    "Assimilator": 2,
    "Gateway": 3,
    "Forge": 4,
    "Nexus": 5,
    "RoboticsFacility": 6,
    "RoboticsBay": 7,
    "DarkShrine": 8,
    "TwilightCouncil": 9,
    "Stargate": 10,
    "TemplarArchives": 11,
    "CyberneticsCore": 12,
    "PhotonCannon": 13,
    "FleetBeacon":14,
}

mapeo_estructura_numero = {valor: llave for llave,valor in mapeo_numero_estructura.items()}
mapeo_estructura_canal = {estructura:0 for estructura in estructuras}

tier_1 = ["Stargate",'TwilightCouncil','RoboticsFacility','DarkShrine','TemplarArchives']

estructuras_permutables = [
    "Gateway",
    "CyberneticsCore",
    "RoboticsFacility",
    "RoboticsBay",
    "TemplarArchive",
    "DarkShrine",
    "Forge",
    "TwilightCouncil",
    "Stargate",
    "FleetBeacon",
    "PhotonCannon"
]

mapeo_numero_estructuras_permutables = {estructura:numero for numero,estructura in enumerate(estructuras_permutables)}
mapeo_estructuras_permutables_numero = {numero:estructura for estructura,numero in mapeo_numero_estructuras_permutables.items()}


# In[8]:

valores = df["Label"].value_counts()

for llave, valor in valores.items():
    if valor < 500:
        del valores[llave]
        
#del valores["1 Hatch "] 
#del valores["2 Hatch "] 
#del valores["3 Hatch "] 



builds = df[df["Label"].isin(valores.keys())]

builds_objetivo = builds.Label.unique().tolist()

for build in builds_objetivo: 
    print(build)
print(len(builds_objetivo))


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
    #parche =  ["Pylon","Gateway","Assimilator"]
   # parche =  ["Assimilator"]
    #if estructura not in parche:#medida de parche para solo tratar con 3 estructuras por ahora
     #   return dataset
    canal = mapeo_estructura_canal[estructura]
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
            dataset[nfila][canal][y][x] = mapeo_estructura_numero[estructura]
        #print("-"*20 + campo)
    return dataset
            


def aReefMainONatural(x,y):
    
    if y <= 29:
        #x_1,y_1 = 144,23
        #x_2,y_2 = 148,29
        m_1 = (29-23)/(148-144)*1.0
        c_1 = 29 - m_1*148

        y_en_recta = m_1*x+c_1

        if y <= y_en_recta: #dentro de main
            return True,False
        elif x >= 118:
            return False,True

    elif y <= 36:
    #x_1,y_1 = 148,29
    #x_2,y_2 = 155,36  
        m_1 = (36-29)/(155-148)*1.0
        c_1 = 29 - m_1*148

        y_en_recta = m_1*x+c_1

        if y <= y_en_recta: #dentro de main
            return True,False 
        else:
            #x_1,y_1 = 118,29
            #x_2,y_2 = 140,43  
            m_1 = (43-29)/(140-118)*1.0
            c_1 = 43 - m_1*140

            y_en_recta = m_1*x+c_1

            if y <= y_en_recta: #dentro de la natural
                return False,True 
    elif y <= 38:
            if x >= 158 and x <= 162:
                return True, False

    return False, False

def interloperMainONatural(x,y):
    
    if x >= 104:
        if y <= 31:#dentro de main
                return True, False
        elif y <= 39:
            if x <= 110:
                #x_1,y_1 = 104,31
                #x_2,y_2 = 110,39  
                m_1 = (39-31)/(110-104)*1.0
                c_1 = 39 - m_1*110

                y_en_recta = m_1*x+c_1

                if y <= y_en_recta: #dentro de la main
                    return True, False 
            else:
                return True, False
            
        elif x <= 113 and x >= 108:
            #x_1,y_1 = 108,43
            #x_2,y_2 = 113,48  
            m_1 = (48-43)/(113-108)*1.0
            c_1 = 48 - m_1*113

            y_en_recta = m_1*x+c_1

            if y <= y_en_recta: #dentro de la main
                return True, False 
                
        elif x <= 118 and x >= 113:
            #x_1,y_1 = 113,48
            #x_2,y_2 = 118,43  
            m_1 = (43-48)/(118-113)*1.0
            c_1 = 48 - m_1*113

            y_en_recta = m_1*x+c_1

            if y <= y_en_recta: #dentro de la main
                return True, False 
                
        elif x <= 125 and x >= 118:
            #x_1,y_1 = 118,43
            #x_2,y_2 = 125,44  
            m_1 = (44-43)/(125-118)*1.0
            c_1 = 44 - m_1*125

            y_en_recta = m_1*x+c_1

            if y <= y_en_recta: #dentro de la main
                return True, False 
            
        elif x <= 125 and x >= 131:
            if y <= 44:
                return True, False
        elif x >= 131 :
            #x_1,y_1 = 131,44
            #x_2,y_2 = 135,40  
            m_1 = (40-44)/(135-131)*1.0
            c_1 = 44 - m_1*131

            y_en_recta = m_1*x+c_1

            if y <= y_en_recta: #dentro de la main
                return True, False 
    else:
        return False, False


def transformarFilaEnInputConvolucional(df):
    canales = len(mapeo_numero_estructura.keys())
    #ancho = 340 #170*2
    #alto = 340 #150*2 - ahora con padding para que sea 340*340
    ancho = 80
    alto = 137
    total = df.shape[0]
    dataset = np.zeros((total,1,alto,ancho))
    df = df.reset_index()
    for index, row in df.iterrows():
        #print(index)
        for estructura in estructuras_reducido:
            dataset = ubicarEstructura(dataset,index,row,estructura)
    return dataset


# In[15]:


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


# In[18]:


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoder_mapeo = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
print(encoder_mapeo)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

categorias = dummy_y.shape[1]


params = encoder.get_params(y)



X_train, X_test, y_train, y_test = train_test_split(dataset,dummy_y,test_size=0.20, random_state=106)


from shutil import make_archive

make_archive("X_train","zip",X_train)

make_archive("X_test","zip",X_test)

make_archive("y_train","zip",y_train)

make_archive("y_test","zip",y_test)




