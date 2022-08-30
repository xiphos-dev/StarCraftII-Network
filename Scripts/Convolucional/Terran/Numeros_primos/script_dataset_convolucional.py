#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:



import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np
import pickle


with open("encoder","rb") as arc:
    encoder = pickle.load(arc)

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

'''
df_encoder = pd.read_csv(file, sep=',', dtype={"Label": str}, usecols=["Label"])
y_completo = df_encoder["Label"]

encoder = LabelEncoder()
encoder.fit(y_completo)
encoder_mapeo = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
print(encoder_mapeo)
encoded_Y = encoder.transform(y_completo)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

categorias = dummy_y.shape[1]


params = encoder.get_params(y_completo) 

del df_encoder
del y_completo

'''


def gen_primes(listado):
    """ Generate an infinite sequence of prime numbers.
    """
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}
    
    # The running integer that's checked for primeness
    q = 2
    maximo = len(listado)
    item_actual = 0
    while q <= maximo :
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            # 
            yield q,listado[item_actual]
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next 
            # multiples of its witnesses to prepare for larger
            # numbers
            # 
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        
        q += 1
        item_actual += 1

numero = "1"

ruta = "../../../../"
archivo = "interloper_terran.csv"
file = ruta+archivo

df = pd.read_csv(file, sep=',', dtype={"Label": str})
#df.head(20)


ruta = "../"
archivo = "extension_dataset_t_interloper.csv"
file = ruta+archivo

df2 = pd.read_csv(file, sep=',', dtype={"Label": str})

df = pd.concat([df,df2], ignore_index=True)

df = df.loc[:200000]

del df2

cota_inferior = 200000
cota_superior = 600000

'''

df = pd.read_csv(file, sep=',', dtype={"Label": str}, skiprows=range(cota_inferior,cota_superior), nrows=400000, engine="python")
'''
#df.head()

print("Leido")

estructuras = [            
            "CommandCenter",
            "PlanetaryFortress",
            "OrbitalCommand",#este lleva un valor arbitrario de llave pues no es construido directamente, asi que no tiene llave propia
            "SupplyDepot",
            "Refinery",
            "Barracks",
            "Factory",
            "Starport",
            "EngineeringBay",
            "Armory",
            "MissileTurret",
            "SensorTower",
            "Bunker",
            "FusionCore",
            "GhostAcademy"]

estructuras_addons = [
    
                "StarportReactor",
                "StarportTechLab",
                "FactoryReactor",
                "FactoryTechLab",
                "BarracksReactor",
                "BarracksTechLab",
]

estructuras_tiempo = [valor+"_tiempo" for valor in estructuras]

estructuras_addons_tiempo = [valor+"_tiempo" for valor in estructuras_addons]

unidades = ["SCV",
            "Reaper",
            "Marine",
            "WidowMine",
            "Medivac",
            "Viking",
            "Marauder",
            "Raven",
            "SiegeTank",
            "Liberator",
            "BattleCruiser",
            "Cyclone",
            "Hellbat",
            "Hellion",
            "Thor",
            "Banshee",
            "Ghost"]

mapeo_unidades_tiempo ={unidad:numero for numero,unidad in enumerate(unidades, 1)}
#print(mapeo_unidades_tiempo.items())

unidades_tiempo = [unidad+"_tiempo" for unidad in unidades]

mejoras = [
                "UpgradeTerranInfantryWeapons1",
                "UpgradeTerranInfantryWeapons2",
                "UpgradeTerranInfantryWeapons3",
                "UpgradeTerranInfantryArmor1",
                "UpgradeTerranInfantryArmor2",
                "UpgradeTerranInfantryArmor3",
                "UpgradeStructureArmor",

                "ResearchStimpack",
                "ResearchCombatShield",
                "ResearchConcussiveShells",
                "ResearchMedivacIncreaseSpeedBoost",
                "ResearchInfernalPreIgniter",
                "ResearchCloakingField",
                "ResearchWeaponRefit",
                "ResearchNeosteelFrame",
                "ResearchHiSecAutoTracking",
                "ResearchDrillingClaws",
                "ResearchBansheeSpeed",
                "ResearchPersonalCloaking",
                "ResearchRavenRecalibratedExplosives",
                'ResearchCorvidReactor',

                "ResearchTerranVehicleAndShipArmorsLevel1",
                "ResearchTerranVehicleAndShipArmorsLevel2",
                "ResearchTerranVehicleAndShipArmorsLevel3",

                "UpgradeVehicleWeapons1",
                "UpgradeVehicleWeapons2",
                "UpgradeVehicleWeapons3",

                "UpgradeShipWeapons1",
                "UpgradeShipWeapons2",
                "UpgradeShipWeapons3",
                "ResearchCycloneLockOnDamageUpgrade",

                "ResearchLiberatorAGRangeUpgrade"
        ]



estructuras_permutables = [
            "SupplyDepot",
            "Barracks",
            "Factory",
            "Starport",
            "EngineeringBay",
            "Armory",
            "MissileTurret",
            "SensorTower",
            "Bunker",
            "FusionCore",
            "GhostAcademy"
]


#mapeo_numero_estructura = {elemento:numero for numero,elemento in enumerate(estructuras_permutables)}


mapeo_estructura_canal = {estructura:0 for estructura in estructuras }
#mapeo_estructura_numero = {valor: llave for llave,valor in mapeo_numero_estructura.items()}

#mapeo_estructura_addons_canal = {estructura: 1 for estructura in estructuras_addons}
#mapeo_estructura_canal.update(mapeo_estructura_addons_canal)

#mapeo_estructura_numero = {valor: llave for llave,valor in mapeo_numero_estructura.items()}




mapeo_numero_estructura = {estructura:numero for numero,estructura in gen_primes(estructuras)}
mapeo_estructuras_numero = {numero:estructura for estructura,numero in mapeo_numero_estructura.items()}

#tier_1 = ["Factory","FactoryReactor","FactoryTechLab",'Staport','StarportReactor','StaportTechlab']
tier_1 = ["ResearchConcussiveShells","ResearchStimpack","ResearchInfernalPreIgniter","ResearchCombatShield","ResearchCloakingField","ResearchLiberatorAGRangeUpgrade"]
valores = df["Label"].value_counts()

for llave, valor in valores.items():
    if valor < 500:
        del valores[llave]
        
#del valores["1 Hatch "] 
#del valores["2 Hatch "] 
#del valores["3 Hatch "] 



builds = df[df["Label"].isin(valores.keys())]

builds_objetivo = builds.Label.unique().tolist()

df = df[df["Label"].isin(builds_objetivo)]
X = df.drop(["Label"], axis=1).drop(estructuras_tiempo, axis=1)
y = df["Label"]



# In[9]:

def mascaraCoordenadas(x,y,mapa): #esta funcion "centra" las coordenadas en un nuevo origen
    #el objetivo es crear un sistema de coordenadas que comience en cero para las estructuras dentro de la main y natural
    if mapa == "Interloper":
        x -= 104
        return x,y

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

        x,y = mascaraCoordenadas(x,y,"Interloper")
    
        if x < 0 or y > 68: # x menor a cero implica que la estructura esta a la izquierda de la main
              continue              # y mayor a 68 implica que la estructura esta sobre el limite superior de la natural

        x,y = biyeccionCoordenadas(x,y)
        x = int(x)
        y = int(y)
        #print("Fila:"+str(nfila))
        #print(x)
        #print(y)
        #print(campo)
        if x != 0 and y != 0:
            if dataset[nfila][canal][y][x] != 0: #la celda se encuentra ocupada por otra estructura
                dataset[nfila][canal][y][x] *= mapeo_numero_estructura[estructura]
            else:
                dataset[nfila][canal][y][x] = mapeo_numero_estructura[estructura]
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
    canales = 1
    #ancho = 340 #170*2
    #alto = 340 #150*2 - ahora con padding para que sea 340*340
    ancho = 80
    alto = 137
    total = df.shape[0]
    dataset = np.zeros((total,1,alto,ancho))
    df = df.reset_index()
    for index, row in df.iterrows():
        #print(index)
        for estructura in estructuras_permutables:
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

'''
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoder_mapeo = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
print(encoder_mapeo)
'''
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

categorias = dummy_y.shape[1]


params = encoder.get_params(y)



X_train, X_test, y_train, y_test = train_test_split(dataset[cota_inferior:][:][:][:],dummy_y[cota_inferior:][:],test_size=0.20, random_state=106)


from shutil import make_archive
import bz2
import pickle


print(X_train.shape)
ofile = bz2.BZ2File("X_train_"+numero,"wb")
pickle.dump(X_train,ofile)
ofile.close()

ofile = bz2.BZ2File("X_test_"+numero,"wb")
pickle.dump(X_test,ofile)
ofile.close()

with open("y_train_"+numero,"wb") as arc:
    pickle.dump(y_train,arc)

with open("y_test_"+numero,"wb") as arc:
    pickle.dump(y_test,arc)

#with open("encoder","wb") as arc:
 #   pickle.dump(encoder,arc)

