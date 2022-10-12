#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np


# In[2]:


ruta = "../../../"
archivo = "zerg.csv"
file = ruta+archivo


# In[3]:


df = pd.read_csv(file, sep=',', dtype={"Label": str})
#df.head(20)


# ##  Preparando dataset

# In[4]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# In[11]:


estructuras = [
            "Hatchery",
            "Extractor",
            "SpawningPool",
            "RoachWarren",
            "BanelingNest",
            "SporeCrawler",
            "EvolutionChamber",
            "HydraliskDen",
            "InfestationPit",
            "SpineCrawler",
            "Spire",
            "UltraliskCavern"
              ]
mapeo_numero_estructura = {estructura:numero for numero,estructura in enumerate(estructuras,50)}
mapeo_estructura_numero = {valor: llave for llave,valor in mapeo_numero_estructura.items()}

estructuras_tiempo = [valor+"_tiempo" for valor in estructuras]

unidades = [
            "Overlord",
            "Overseer",
            "Drone",
            "Zergling",
            "Queen",
            "Roach",
            "Hydralisk",
            "Baneling",
            "Mutalisk",
            "Corruptor",
            "Infestor",
            "Broodlord",
            "Ravager",
            "Swarmhost"
]

mapeo_unidades_tiempo ={unidad:numero for numero,unidad in enumerate(unidades,1)}
#print(mapeo_unidades_tiempo.items())

unidades_tiempo = {llave+"_tiempo" for llave in unidades }

mejoras = [ 
            "EvolveChitinousPlating",
            "EvolveAdrenalGlands",
            "EvolvePneumatizedCarapace",
            "EvolveMetabolicBoost",
            "EvolveCentrifugalHooks",
            "EvolvePathogenGlands",
            "EvolveBurrow",
            "EvolveTunnelingClaws",
            "EvolveGlialReconstitution",
            "EvolveFlyerAttacks1",
            "EvolveFlyerAttacks2",
            "EvolveFlyerAttacks3",
            "EvolveFlyerCarapace1",
            "EvolveFlyerCarapace2",
            "EvolveFlyerCarapace3",
            "EvolveNeuralParasite",

            "ResearchEvolveMuscularAugments",

            "ResearchZergMeleeWeaponsLevel1",
            "ResearchZergMeleeWeaponsLevel2",
            "ResearchZergMeleeWeaponsLevel3",

            "ResearchZergGroundArmorsLevel1",
            "ResearchZergGroundArmorsLevel2",
            "ResearchZergGroundArmorsLevel3",

            "ResearchZergMissileWeaponsLevel1",
            "ResearchZergMissileWeaponsLevel2",
            "ResearchZergMissileWeaponsLevel3",
    
            "UpgradeToLair",
            "CancelUpgradeToLair",
            "UpgradeToLurkerDenMP",
            "CancelUpgradeToLurkerDenMP",
            "UpgradeToHive",
            "CancelUpgradeToHive",
            "UpgradeToGreaterSpire",
            "CancelUpgradeToGreaterSpire",
          ]

mapeo_mejora_numero = {mejora: numero for numero,mejora in enumerate(mejoras,100)}

#aqui se extraen las builds que tienen suficientes ejemplos para el entrenamiento

valores = df["Label"].value_counts()

for llave, valor in valores.items():
    if valor < 500:
        del valores[llave]
        
#del valores["1 Hatch "] 
#del valores["2 Hatch "] 
#del valores["3 Hatch "] 



builds = df[df["Label"].isin(valores.keys())]

builds_objetivo = builds.Label.unique().tolist()

bases = [build for build in builds_objetivo if "1 Hatch" in build]

print(bases)

bases = [build for build in builds_objetivo if "2 Hatch" in build]

print(bases)
bases = [build for build in builds_objetivo if "3 Hatch" in build]

print(bases)
bases = [build for build in builds_objetivo if "4 Hatch" in build]

print(bases)

bases = [build for build in builds_objetivo if "5 Hatch" in build]

print(bases)

bases = [build for build in builds_objetivo if "6 Hatch" in build]

print(bases)

bases = [build for build in builds_objetivo if "7 Hatch" in build]

print(bases)
tier_1 = ["ResearchZergMeleeWeaponsLevel1","ResearchZergMissileWeaponsLevel1","EvolveBurrow","EvolveGlialReconstitution","EvolveCentrifugalHooks"]
#tier_1 = ["Spire",'HydraliskDen','InfestationPit',"UpgradeToLurkerDenMP"]


# In[7]:


def primeraTech(fila, cota_tiempo=336):
    #1:24 minutos in-replay time toma para que transcurra 1 minuto in game
    #1 minuto in game = 1:24 = 84 segundos reales
    #2 minutos in game = 2:48 = 168 segundos reales
    #3 minutos in game = 4:12 = 252 segundos reales
    #4 minutos in game = 5:36 = 336 segundos reales
    tech = tier_1
    mejoras = "UpgradeToLurkerDenMP"
    minimo_tiempo = 999999 # variable que recuerda el tiempo de la ultima ocurrencia, se inicializa con un valor equivalente (en terminos practicos) a infinito
    tech_elegida="No tech"
    flag_wall = False #algunas estructuras de tech son construidas muy rapido, normalmente como parte del muro
    #eg: un stargate en la muralla al 2:40
    #este flag busca si existe otra estructura construida poco despues antes de la cota de tiempo
    if fila[mejoras] != 0:
        tech_elegida = "LurkerDen"
    else:    
        for i in range(len(tech)):
            if fila[tech[i]+"_t_1"] < minimo_tiempo and fila[tech[i]+"_t_1"] != 0 and fila[tech[i]+"_t_1"] <= cota_tiempo:
                tech_elegida = tech[i]
                minimo_tiempo = fila[tech[i]+"_t_1"]
    return tech_elegida


# In[8]:


def primeraTechMejoras(fila, cota_tiempo=336):
    #1:24 minutos in-replay time toma para que transcurra 1 minuto in game
    #1 minuto in game = 1:24 = 84 segundos reales
    #2 minutos in game = 2:48 = 168 segundos reales
    #3 minutos in game = 4:12 = 252 segundos reales
    #4 minutos in game = 5:36 = 336 segundos reales
    tech = tier_1
    minimo_tiempo = 999999
    tech_elegida="No tech"
    for i in range(len(tech)):
        if fila[tech[i]] < minimo_tiempo and fila[tech[i]] != 0 and fila[tech[i]] <= cota_tiempo:
            tech_elegida = tech[i]
            minimo_tiempo = fila[tech[i]]
    return tech_elegida


# In[9]:


from sklearn.preprocessing import StandardScaler

def escalar(listado_valores,scaler=None):
    listado_valores = np.array(listado_valores)
    if len(listado_valores) != 0:
        if scaler == None:
            scaler = StandardScaler()
            #print("Largo antes:"+str(len(tiempos_primeros)))
            #print("Largo antes:"+str(tiempos_primeros))
            listado_valores = np.reshape(scaler.fit_transform(listado_valores[:, np.newaxis]),(1,-1))
            #print("Largo despues:"+str(tiempos_primeros.shape))
            #print("Largo despues:"+str(tiempos_primeros))
        else:
            listado_valores = np.reshape(scaler.transform(listado_valores[:, np.newaxis]),(1,-1))

    return listado_valores,scaler

def recomponerListado(largo,listado_posiciones,listado_valores,output):
    pos=0
    for i in range(largo):
        if pos<len(listado_posiciones):
            if listado_posiciones[pos] == i:
                output.append(listado_valores[0,pos])
                #output.append(listado_valores[pos])
                pos+=1
            else:
                output.append(0)
        else:
            output.append(0)
    return output


# In[10]:



def inputDenso(df, tiempo_limite):
    output = []
    for index, row in df.iterrows():
        for unidad in unidades:
            cantidad = row[unidad]
            tiempos = row[unidad+"_tiempo"]
            if isinstance(tiempos,str):
                tiempos = row[unidad+"_tiempo"].strip("[").strip("]").split(",")
                tiempo_primero = int(tiempos[0])
                tiempo_ultimo = int(tiempos[0])
                for tiempo in tiempos:
                    if int(tiempo) <= tiempo_limite:
                        if tiempo_ultimo < int(tiempo):
                            tiempo_ultimo = int(tiempo)
                    else:
                        break
            else:
                tiempo_primero = 0
                tiempo_ultimo = 0
            output.append(cantidad)
            output.append(tiempo_primero)
            output.append(tiempo_ultimo)
            
            
        posiciones=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_primeros = []
        for estructura in estructuras:
            tiempo_actual = row[estructura+"_t_1"]
            if tiempo_actual <= tiempo_limite and tiempo_actual != 0:
                tiempos_primeros.append(tiempo_actual)
                posiciones.append(k)
            k+=1

       # tiempos_primeros,scaler = escalar(tiempos_primeros)
        output = recomponerListado(len(estructuras),posiciones,tiempos_primeros,output)
        
        '''
        posiciones=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_mejora = []
        for mejora in mejoras:
            tiempo_mejora = row[mejora]
            if tiempo_mejora <= tiempo_limite and tiempo_mejora != 0:
                tiempos_mejora.append(tiempo_mejora)
                posiciones.append(k)
            k+=1
            
        #tiempos_mejora,_ = escalar(tiempos_mejora,scaler)
        output = recomponerListado(len(mejoras),posiciones,tiempos_mejora,output)
        '''

    output = np.reshape(output,(df.shape[0],len(unidades)*3+len(estructuras)))
    return output


# In[11]:



def inputDensoModificado(df, tiempo_limite):
    output = []
    for index, row in df.iterrows():
            
        posiciones=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_primeros = []
        for estructura in estructuras:
            if estructura not in (tier_1):
                tiempo_actual = row[estructura+"_t_1"]
                if tiempo_actual <= tiempo_limite and tiempo_actual != 0:
                   # print("Added:"+estructura)
                    tiempos_primeros.append(tiempo_actual)
                    posiciones.append(k)
            k+=1

        tiempos_primeros,scaler = escalar(tiempos_primeros)
        output = recomponerListado(len(estructuras)-len(tier_1),posiciones,tiempos_primeros,output)
        

        posiciones=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_mejora = []
        for mejora in mejoras:
            tiempo_mejora = row[mejora]
            if tiempo_mejora <= tiempo_limite and tiempo_mejora != 0:
                tiempos_mejora.append(tiempo_mejora)
                posiciones.append(k)
            k+=1
            
        tiempos_mejora,_ = escalar(tiempos_mejora,scaler)
        output = recomponerListado(len(mejoras),posiciones,tiempos_mejora,output)

        for unidad in unidades:
            cantidad = row[unidad]
            tiempos = row[unidad+"_tiempo"]
            if isinstance(tiempos,str):
                tiempos = row[unidad+"_tiempo"].strip("[").strip("]").split(",")
                tiempo_primero = int(tiempos[0])
                tiempo_ultimo = int(tiempos[0])
                for tiempo in tiempos:
                    if int(tiempo) <= tiempo_limite:
                        if tiempo_ultimo < int(tiempo):
                            tiempo_ultimo = int(tiempo)
                    else:
                        break
            else:
                tiempo_primero = 0
                tiempo_ultimo = 0
            output.append(cantidad)
            tiempo_primero,_ = escalar([tiempo_primero],scaler)
            output.append(tiempo_primero[0,0])
            tiempo_ultimo,_ = escalar([tiempo_ultimo],scaler)
            output.append(tiempo_ultimo[0,0])

    output = np.reshape(output,(df.shape[0],len(unidades)*3+len(estructuras)-len(tier_1)+len(mejoras)))
    return output


# In[12]:


def inputDensoRandomSelection(df, tiempo_limite, probabilidad_seleccion=0.8, filtrar_estructuras=True,filtrar_mejoras=False,filtrar_unidades=False):#esta funcion descarta aleatoriamente algunos datos para simular las observaciones incompletas dentro de una partida real
    n_unidades = 300
    output = []

    for index, row in df.iterrows():
            
        posiciones=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_primeros = []
        for estructura in estructuras:
            if estructura not in (tier_1):
                tiempo_actual = row[estructura+"_t_1"]
                if tiempo_actual <= tiempo_limite and tiempo_actual != 0:
                   # print("Added:"+estructura)
                    if filtrar_estructuras:
                        umbral_seleccion = np.random.random_sample()
                        if umbral_seleccion <= probabilidad_seleccion:
                            tiempos_primeros.append(tiempo_actual)
                            posiciones.append(k)
                    else:
                        tiempos_primeros.append(tiempo_actual)
                        posiciones.append(k)
            k+=1

        tiempos_primeros,scaler = escalar(tiempos_primeros)
        output = recomponerListado(len(estructuras)-len(tier_1),posiciones,tiempos_primeros,output)
        

        posiciones=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_mejora = []
        for mejora in mejoras:
            if mejora not in (tier_1):
                tiempo_mejora = row[mejora]
                if tiempo_mejora <= tiempo_limite and tiempo_mejora != 0:
                    if filtrar_mejoras:
                        umbral_seleccion = np.random.random_sample()
                        if umbral_seleccion <= probabilidad_seleccion:
                            tiempos_mejora.append(tiempo_mejora)
                            posiciones.append(k)
                    else:
                        tiempos_mejora.append(tiempo_mejora)
                        posiciones.append(k)
            k+=1
            
        tiempos_mejora,_ = escalar(tiempos_mejora,scaler)
        output = recomponerListado(len(mejoras),posiciones,tiempos_mejora,output)
        
        orden_unidades=[0] * n_unidades # lista que contiene la secuencia de unidades creadas
        pos = 0
        
        for unidad in unidades:
            cantidad = row[unidad]
            tiempos = row[unidad+"_tiempo"]
            if isinstance(tiempos,str):
                tiempos = row[unidad+"_tiempo"].strip("[").strip("]").split(",")
                tiempo_primero = int(tiempos[0])
                tiempo_ultimo = int(tiempos[0])
                for tiempo in tiempos:
                    if int(tiempo) <= tiempo_limite:
                        if filtrar_unidades:
                            umbral_seleccion = np.random.random_sample()
                            if umbral_seleccion <= probabilidad_seleccion:
                                orden_unidades[pos] = mapeo_unidades_tiempo[unidad]
                                pos+=1
                                if tiempo_ultimo < int(tiempo):
                                    tiempo_ultimo = int(tiempo)
                        else:
                            orden_unidades[pos] = mapeo_unidades_tiempo[unidad]
                            pos+=1
                            if tiempo_ultimo < int(tiempo):
                                tiempo_ultimo = int(tiempo)
                    else:
                        break
            else:
                tiempo_primero = 0
                tiempo_ultimo = 0
            
            '''
            output.append(cantidad)
            tiempo_primero,_ = escalar([tiempo_primero],scaler)
            output.append(tiempo_primero[0,0])
            tiempo_ultimo,_ = escalar([tiempo_ultimo],scaler)
            output.append(tiempo_ultimo[0,0])
            '''

        for unidad in orden_unidades:
            output.append(unidad)

    output = np.reshape(output,(df.shape[0],len(orden_unidades)+len(estructuras)-len(tier_1)+len(mejoras)))
    #output = np.reshape(output,(df.shape[0],len(orden_unidades)+len(unidades)*3+len(estructuras)-len(tier_1)+len(mejoras)))
    return output


# In[13]:


def inputDensoRandomSelectionConDatoClave(df, tiempo_limite, probabilidad_seleccion=0.8, filtrar_estructuras=True,filtrar_mejoras=False,filtrar_unidades=False):#esta funcion descarta aleatoriamente algunos datos para simular las observaciones incompletas dentro de una partida real
    n_unidades = 300
    output = []

    for index, row in df.iterrows():
            
        posiciones=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_primeros = []
        for estructura in estructuras:
            if estructura not in (tier_1):
                tiempo_actual = row[estructura+"_t_1"]
                if tiempo_actual <= tiempo_limite and tiempo_actual != 0:
                   # print("Added:"+estructura)
                    if filtrar_estructuras:
                        umbral_seleccion = np.random.random_sample()
                        if umbral_seleccion <= probabilidad_seleccion:
                            tiempos_primeros.append(tiempo_actual)
                            posiciones.append(k)
                    else:
                        tiempos_primeros.append(tiempo_actual)
                        posiciones.append(k)
            k+=1

        tiempos_primeros,scaler = escalar(tiempos_primeros)
        output = recomponerListado(len(estructuras)-len(tier_1),posiciones,tiempos_primeros,output)
        

        posiciones=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_mejora = []
        for mejora in mejoras:
            tiempo_mejora = row[mejora]
            if tiempo_mejora <= tiempo_limite and tiempo_mejora != 0:
                if filtrar_mejoras:
                    umbral_seleccion = np.random.random_sample()
                    if umbral_seleccion <= probabilidad_seleccion:
                        tiempos_mejora.append(tiempo_mejora)
                        posiciones.append(k)
                else:
                    tiempos_mejora.append(tiempo_mejora)
                    posiciones.append(k)
            k+=1
            
        tiempos_mejora,_ = escalar(tiempos_mejora,scaler)
        output = recomponerListado(len(mejoras),posiciones,tiempos_mejora,output)
        
        orden_unidades=[0] * n_unidades # lista que contiene la secuencia de unidades creadas
        pos = 0
        
        for unidad in unidades:
            cantidad = row[unidad]
            tiempos = row[unidad+"_tiempo"]
            if isinstance(tiempos,str):
                tiempos = row[unidad+"_tiempo"].strip("[").strip("]").split(",")
                tiempo_primero = int(tiempos[0])
                tiempo_ultimo = int(tiempos[0])
                for tiempo in tiempos:
                    if int(tiempo) <= tiempo_limite:
                        if filtrar_unidades:
                            umbral_seleccion = np.random.random_sample()
                            if umbral_seleccion <= probabilidad_seleccion:
                                orden_unidades[pos] = mapeo_unidades_tiempo[unidad]
                                pos+=1
                                if tiempo_ultimo < int(tiempo):
                                    tiempo_ultimo = int(tiempo)
                        else:
                            orden_unidades[pos] = mapeo_unidades_tiempo[unidad]
                            pos+=1
                            if tiempo_ultimo < int(tiempo):
                                tiempo_ultimo = int(tiempo)
                    else:
                        break
            else:
                tiempo_primero = 0
                tiempo_ultimo = 0
            

            output.append(cantidad)
            tiempo_primero,_ = escalar([tiempo_primero],scaler)
            output.append(tiempo_primero[0,0])
            tiempo_ultimo,_ = escalar([tiempo_ultimo],scaler)
            output.append(tiempo_ultimo[0,0])


        for unidad in orden_unidades:
            output.append(unidad)

    #output = np.reshape(output,(df.shape[0],len(orden_unidades)+len(estructuras)-len(tier_1)+len(mejoras)))
    output = np.reshape(output,(df.shape[0],len(orden_unidades)+len(unidades)*3+len(estructuras)-len(tier_1)+len(mejoras)))
    return output


# In[14]:


columnas_coordenadas = [col for col in df.columns if "_x" in col or "_y" in col]
columnas_tiempo = [col for col in df.columns if "_t" in col]


# In[15]:


df = df[df["Label"].isin(builds_objetivo)]
df["Primer_tech"] = df.apply(lambda fila: primeraTechMejoras(fila,700), axis=1)
X = df.drop(columnas_coordenadas, axis=1).drop(["Replay","Label","Primer_tech"],axis=1)
y = df["Primer_tech"]
#y = df["Label"]
X.head()


# In[16]:


y.value_counts()


# In[17]:


X["Drone"] = X.apply(lambda fila: 12 if fila["Drone"] < 12 else fila["Drone"], axis=1)
X.head()


# In[18]:


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[23]:


input_random = inputDensoRandomSelection(X,336,0.8,True,True,True)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)


# In[ ]:


print(dummy_y.shape)
categorias = dummy_y.shape[1]
print(dummy_y[0])


# In[18]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[23]:


input_shape = X_train.shape[1]
print(input_shape)
print(X_train)


import pickle
import bz2
from shutil import make_archive

ofile = bz2.BZ2File("X_train","wb")
pickle.dump(X_train,ofile)
ofile.close()

ofile = bz2.BZ2File("X_test","wb")
pickle.dump(X_test,ofile)
ofile.close()

with open("y_train","wb") as arc:
    pickle.dump(y_train,arc)

with open("y_test","wb") as arc:
    pickle.dump(y_test,arc)

with open("encoder","wb") as arc:
    pickle.dump(encoder,arc)


