#!/usr/bin/env python
# coding: utf-8

# 

# In[19]:

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np


# In[20]:


ruta = "../../../"
archivo = "zerg.csv"
file = ruta+archivo


# In[21]:


df = pd.read_csv(file, sep=',', dtype={"Label": str})
#df.head(20)


# In[22]:


#df = df[df["Probe"] != 0]

df.shape


# ##  Preparando dataset

# In[23]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# In[24]:



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

numero = len(mapeo_unidades_tiempo.keys()) + 1
 
mejoras = [ 
            "UpgradeToLair",
            "CancelUpgradeToLair",
            "UpgradeToLurkerDenMP",
            "CancelUpgradeToLurkerDenMP",
            "UpgradeToHive",
            "CancelUpgradeToHive",
            "UpgradeToGreaterSpire",
            "CancelUpgradeToGreaterSpire",

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
          ]

mapeo_mejora_numero = {mejora: numero for numero,mejora in enumerate(mejoras,numero)}
numero+= len(mapeo_mejora_numero.keys()) 
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
mapeo_numero_estructura = {estructura:numero for numero,estructura in enumerate(estructuras,numero)}
mapeo_estructura_numero = {valor: llave for llave,valor in mapeo_numero_estructura.items()}

estructuras_tiempo = [valor+"_tiempo" for valor in estructuras]

#print(mapeo_unidades_tiempo.items())

unidades_tiempo = {llave+"_tiempo" for llave in unidades }



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

for build in builds_objetivo: 
    print(build)
print(len(builds_objetivo))

tier_1 = ["ResearchZergMeleeWeaponsLevel1","ResearchZergMissileWeaponsLevel1","EvolveBurrow","EvolveGlialReconstitution","EvolveCentrifugalHooks"]
#tier_1 = ["Spire",'HydraliskDen','InfestationPit',"UpgradeToLurkerDenMP"]


def primeraTech(fila, cota_tiempo=336):
    #1:24 minutos in-replay time toma para que transcurra 1 minuto in game
    #1 minuto in game = 1:24 = 84 segundos reales
    #2 minutos in game = 2:48 = 168 segundos reales
    #3 minutos in game = 4:12 = 252 segundos reales
    #4 minutos in game = 5:36 = 336 segundos reales
    #5 minutos in game = 7:00 = 420 segundos reales
    #6 minutos in game = 8:24 = 504 segundos reales
    tech = tier_1
    minimo_tiempo = 999999
    tech_elegida="No tech"
    flag_wall = False #algunas estructuras de tech son construidas muy rapido, normalmente como parte del muro
    #eg: un stargate en la muralla al 2:40
    #este flag busca si existe otra estructura construida poco despues antes de la cota de tiempo
    for i in range(len(tech)):
        if fila[tech[i]+"_t_1"] < minimo_tiempo and fila[tech[i]+"_t_1"] != 0 and fila[tech[i]+"_t_1"] <= cota_tiempo:
            tech_elegida = tech[i]
            minimo_tiempo = fila[tech[i]+"_t_1"]
    return tech_elegida
        
    


# In[27]:


columnas_coordenadas = [col for col in df.columns if "_x" in col or "_y" in col]
columnas_tiempo = [col for col in df.columns if "_t" in col]


df = df[df["Label"].isin(builds_objetivo)]

X = df.drop(["Label"], axis=1).drop("Replay", axis=1).drop(columnas_coordenadas, axis=1).drop(estructuras_tiempo, axis=1)
X["Drone"] = X.apply(lambda fila: 12 if fila["Drone"] < 12 else fila["Drone"], axis=1)
X.head()

y2 = df["Label"]


# In[28]:


columnas_tiempo


# In[29]:


def inputRecurrenteMixtoUno(df,y_target,main,natural,tiempo_limite):
    #esta funcion retorna en una misma lista la informacion de unidades/estructuras
    #columnas = [col for col in df.columns if "_x_" in col or "_y_" in col]
    tiempos = [col for col in df.columns if "_t" in col and col != "Primer_tech"]
    upgrades = [col for col in df.columns if col in mejoras]
    #print(len(tiempos))
    output_rnn = []
    output_denso = []
    vocabulario = []
    #largo = 30 #336 tiempo
    largo = 24
    largo_unidades = 15
    vacio_mapa=[]
    for index, row in df.iterrows():
        #fila=np.full((15,2), ["999",999],dtype=dtype)
        #fila=np.full((15,2), 999, dtype=object)
        #print("Going:"+str(index))
        vacio = True
        vacio_unidades = True
        fila_estructuras=[[0, 999] for _,_ in enumerate(range(largo))]
        fila_final=[]
        fila_denso=[]
        #fila = sorted(fila, key=lambda x: x[1], reverse=True)
       # print(fila)
        pos=0
        i=0
        while i < len(tiempos):
            #x = row[columnas[i]]
            #y = row[columnas[i+1]
            frases = tiempos[i].split("_")
            estructura = frases[0]
            string_end = frases[1]
            if estructura in tier_1: continue
            tiempo = row[tiempos[i]]
            #print(tiempos[i]+":"+str(tiempo))
            if string_end == "tiempo":
                if estructura == "Probe":
                    #print("Probe detectada")
                    i+=1
                    continue #no contamos probes por el problema de inconsistencia al comienzo: algunos seleccionan las 12 probes, otros no
            elif int(tiempo) == 0 or int(tiempo) > tiempo_limite:
                i+=1
                continue
            
            #frases = tiempos[t].split("_")
            #estructura = frases[0]
            #string_end = frases[1]
            
            #procesar listado de unidades
            if string_end == "tiempo":
                unidad = mapeo_unidades_tiempo[estructura]
                if (isinstance(tiempo,str)):
                    tiempo = tiempo.strip("[").strip("]")
                    tiempo = tiempo.split(",")
                    for elemento in tiempo:
                        #print(str(unidad)+":"+elemento)
                        if int(elemento) > tiempo_limite:
                            break
                        if int(elemento) != 0:
                            fila_estructuras[pos_unidades] = [unidad,int(elemento)]
                            pos+=1
                            vacio_unidades = False
                else:
                    if tiempo <= tiempo_limite and tiempo != 0:
                        fila_estructuras[pos_unidades] = [unidad,int(tiempo)]
                        poss+=1
                        vacio_unidades = False

            else:
                estructura = mapeo_numero_estructura[estructura]
                fila_estructuras[pos] = [estructura, int(tiempo)]
                pos+=1
                vacio = False
                    #print(estructura+"*")
                    
                    #print(tiempos[t])
            i+=1
            
        
        #print(fila)
        if not(vacio) or not(vacio_unidades):
            fila_estructuras = sorted(fila_estructuras, key=lambda x: x[1])

            fila_mejoras = []
            for mejora in mejoras:
                fila_mejoras.append(row[mejora])
                
            for posicion in range(len(fila_estructuras)):
                if fila_estructuras[posicion][1] == 999:
                    fila_estructuras[posicion][1] = 0
                fila_denso.append(fila_estructuras[posicion][0])
                fila_denso.append(fila_estructuras[posicion][1])
                
                fila_final.append(fila_estructuras[posicion][0])
                #fila_final.append(fila[posicion][1])
                vocabulario.append(fila_estructuras[posicion][0])
                #vocabulario.append(fila[posicion][1])
                
            for k in range(len(fila_mejoras)):
                fila_denso.append(fila_mejoras[k])
            #print(len(fila_denso))
            #output.append([x[0] for x in fila])
            output_rnn.append(fila_final)
            output_denso.append(fila_denso)
        else:
            vacio_mapa.append(index)
            y_target=y_target.drop(index)
    vocabulario = convertirATupla(vocabulario)
    vocabulario = set(vocabulario)

    x_batch_rnn = np.reshape(output_rnn,[df.shape[0]-len(vacio_mapa),largo-len(tier1)])
    x_batch_denso = np.reshape(output_denso,[df.shape[0]-len(vacio_mapa),(largo*2)-len(tier1)+len(mejoras)])
    
    return x_batch_rnn, x_batch_denso, y_target, vacio_mapa, vocabulario


# In[30]:


def convertirATupla(lista):
    return tuple(i for i in lista)

def inputRecurrenteMixto(df,y_target,main,natural,tiempo_limite):
    #columnas = [col for col in df.columns if "_x_" in col or "_y_" in col]
    tiempos = [col for col in df.columns if "_t" in col and col != "Primer_tech"]
    upgrades = [col for col in df.columns if col in mejoras]
    #print(len(tiempos))
    output_rnn = []
    output_rnn_unidades=[]
    output_rnn_mejoras = []
    vocabulario = []
    vocabulario_unidades = []
    vocabulario_mejoras = []
    #largo = 30 #336 tiempo
    largo = 60
    largo_unidades = 130
    largo_mejoras = len(mejoras)
    vacio_mapa=[]
    for index, row in df.iterrows():
        #fila=np.full((15,2), ["999",999],dtype=dtype)
        #fila=np.full((15,2), 999, dtype=object)
        #print("Going:"+str(index))
        vacio = True
        vacio_unidades = True
        vacio_mejoras = True
        fila_estructuras=[[0, 999] for _,_ in enumerate(range(largo))]
        fila_unidades=[[0, 999] for _,_ in enumerate(range(largo_unidades))]
        fila_mejoras=[[0, 999] for _,_ in enumerate(range(largo_mejoras))]
        fila_final_mejoras=[]
        fila_final_unidades=[]
        #fila = sorted(fila, key=lambda x: x[1], reverse=True)
       # print(fila)
        pos=0
        pos_unidades=0
        i=0
        while i < len(tiempos):
            fila_final=[]
            #x = row[columnas[i]]
            #y = row[columnas[i+1]
            frases = tiempos[i].split("_")
            estructura = frases[0]
            string_end = frases[1]
            if estructura in tier_1: 
                i+=1
                continue
            tiempo = row[tiempos[i]]
            #print(tiempos[i]+":"+str(tiempo))
            '''
            if string_end == "tiempo":
                if estructura == "Probe":
                    #print("Probe detectada")
                    i+=1
                    continue #no contamos probes por el problema de inconsistencia al comienzo: algunos seleccionan las 12 probes, otros no
            '''
            #frases = tiempos[t].split("_")
            #estructura = frases[0]
            #string_end = frases[1]
            
            #procesar listado de unidades
            
            if string_end == "tiempo":
                unidad = mapeo_unidades_tiempo[estructura]
                if (isinstance(tiempo,str)):
                    tiempo = tiempo.strip("[").strip("]")
                    tiempo = tiempo.split(",")
                    for elemento in tiempo:
                        #print(str(unidad)+":"+elemento)
                        if int(elemento) > tiempo_limite:
                            break

                        fila_unidades[pos_unidades] = [unidad,int(elemento)]
                        pos_unidades+=1
                        vacio_unidades = False
                else:
                    if tiempo <= tiempo_limite:
                        fila_unidades[pos_unidades] = [unidad,int(tiempo)]
                        pos_unidades+=1
                        vacio_unidades = False
            elif int(tiempo) > tiempo_limite:
                i+=1
                continue
            else:
                estructura = mapeo_numero_estructura[estructura]
                fila_estructuras[pos] = [estructura, int(tiempo)]
                pos+=1
                vacio = False
                    #print(estructura+"*")
                    
                    #print(tiempos[t])
            i+=1
        pos_mejora=0
        for k in range(len(mejoras)):
            mejora = mejoras[k]
            mejora = mapeo_mejora_numero[mejora]
            tiempo = row[mejora]
            if tiempo <= tiempo_limite:
                fila_mejoras[pos_mejora][0] = int(mejora)
                fila_mejoras[pos_mejora][1] = tiempo
                pos_mejora+=1
                vacio_mejoras=False
        #print(fila)
        if not(vacio) or not(vacio_unidades) or not(vacio_mejoras):
            fila_estructuras = sorted(fila_estructuras, key=lambda x: x[1])
                
            for posicion in range(len(fila_estructuras)):
                if fila_estructuras[posicion][1] == 999:
                    fila_estructuras[posicion][1] = 0
                    
                #fila_denso.append(fila_estructuras[posicion][0])
                #fila_denso.append(fila_estructuras[posicion][1])
                fila_final.append(fila_estructuras[posicion][0])
                #fila_final.append(fila[posicion][1])
                vocabulario.append(fila_estructuras[posicion][0])
                #vocabulario.append(fila[posicion][1])
                
            fila_mejoras = sorted(fila_mejoras, key=lambda x: x[1])
            
            for posicion in range(len(fila_mejoras)):
                if fila_mejoras[posicion][1] == 999:
                    fila_mejoras[posicion][1] = 0
                fila_final.append(fila_mejoras[posicion][0])
                vocabulario.append(fila_mejoras[posicion][0])    
            
            
            fila_unidades = sorted(fila_unidades, key=lambda x: x[1])
            
            for posicion in range(len(fila_unidades)):
                if fila_unidades[posicion][1] == 999:
                    fila_unidades[posicion][1] = 0
                fila_final.append(fila_unidades[posicion][0])
                vocabulario.append(fila_unidades[posicion][0])
            
            #print(len(fila_final_mejoras))
            #output.append([x[0] for x in fila])
            output_rnn.append(fila_final)
        else:
            vacio_mapa.append(index)
            y_target=y_target.drop(index)
            df = df.drop(index)
    vocabulario = convertirATupla(vocabulario)
    vocabulario = set(vocabulario)
    '''
    x_batch_rnn = np.reshape(output_rnn,[df.shape[0],largo])
    x_batch_rnn_unidades = np.reshape(output_rnn_unidades,[df.shape[0],largo_unidades])
    x_batch_rnn_mejoras = np.reshape(output_rnn_mejoras,[df.shape[0],largo_mejoras])
    '''
    
    return df,np.asarray(output_rnn), y_target, vacio_mapa, vocabulario
# In[31]:


def inputRecurrenteConIntervalo(df,y_target,main,natural,tiempo_inicio,tiempo_limite): #esta funcion es identica a la que esta arriba, pero extrae eventos dentro de un intervalo de tiempo a hasta b; la funcion de arriba extra eventos dentro de un intervalo 0 hasta b
    #columnas = [col for col in df.columns if "_x_" in col or "_y_" in col]
    tiempos = [col for col in df.columns if "_t" in col and col != "Primer_tech"]
    upgrades = [col for col in df.columns if col in mejoras]
    #print(len(tiempos))
    output_rnn = []
    output_rnn_unidades=[]
    output_rnn_mejoras = []
    
    output_rnn_mid = []
    output_rnn_unidades_mid =[]
    output_rnn_mejoras_mid  = []
    
    vocabulario = []
    vocabulario_unidades = []
    vocabulario_mejoras = []
    #largo = 30 #336 tiempo
    largo = 300
    largo_unidades = 200
    largo_mejoras = len(mejoras)
    vacio_mapa=[]
    for index, row in df.iterrows():
        #fila=np.full((15,2), ["999",999],dtype=dtype)
        #fila=np.full((15,2), 999, dtype=object)
        #print("Going:"+str(index))
        vacio = True
        vacio_unidades = True
        vacio_mejoras = True
        
        vacio_mid = True
        vacio_unidades_mid = True
        vacio_mejoras_mid = True
        
        fila_estructuras=[[0, 999] for _,_ in enumerate(range(largo))]
        fila_unidades=[[0, 999] for _,_ in enumerate(range(largo_unidades))]
        fila_mejoras=[[0, 999] for _,_ in enumerate(range(largo_mejoras))]
        fila_final=[]
        fila_final_mejoras=[]
        fila_final_unidades=[]
        
        fila_estructuras_mid=[[0, 999] for _,_ in enumerate(range(largo))]
        fila_unidades_mid=[[0, 999] for _,_ in enumerate(range(largo_unidades))]
        fila_mejoras_mid=[[0, 999] for _,_ in enumerate(range(largo_mejoras))]
        fila_final_mid=[]
        fila_final_mejoras_mid=[]
        fila_final_unidades_mid=[]
        
        
        
        #fila = sorted(fila, key=lambda x: x[1], reverse=True)
       # print(fila)
        pos=0
        pos_unidades=0
        
        pos_mid=0
        pos_unidades_mid=0
        i=0
        while i < len(tiempos):
            #x = row[columnas[i]]
            #y = row[columnas[i+1]
            frases = tiempos[i].split("_")
            estructura = frases[0]
            string_end = frases[1]
            if estructura in tier_1: 
                i+=1
                continue
            tiempo = row[tiempos[i]]
            #print(tiempos[i]+":"+str(tiempo))
            '''
            if string_end == "tiempo":
                if estructura == "Probe":
                    #print("Probe detectada")
                    i+=1
                    continue #no contamos probes por el problema de inconsistencia al comienzo: algunos seleccionan las 12 probes, otros no
            '''
            #frases = tiempos[t].split("_")
            #estructura = frases[0]
            #string_end = frases[1]
            
            #procesar listado de unidades
            
            if string_end == "tiempo":
                unidad = mapeo_unidades_tiempo[estructura]
                if (isinstance(tiempo,str)):
                    tiempo = tiempo.strip("[").strip("]")
                    tiempo = tiempo.split(",")
                    for elemento in tiempo:
                        #print(str(unidad)+":"+elemento)
                        if int(elemento) < tiempo_inicio and int(elemento) != 0:#early game
                            fila_unidades[pos_unidades] = [unidad,int(elemento)]
                            pos_unidades+=1
                            vacio_unidades = False
                        
                        elif int(elemento) <= tiempo_limite:#mid game
                            fila_unidades_mid[pos_unidades] = [unidad,int(elemento)]
                            pos_unidades_mid+=1
                            vacio_unidades_mid = False
                        
                        else: #nos pasamos del mid game
                            break
                else:
                    if tiempo < tiempo_inicio and tiempo != 0: #early game
                        fila_unidades[pos_unidades] = [unidad,int(tiempo)]
                        pos_unidades+=1
                        vacio_unidades = False
                        
                    elif tiempo <= tiempo_limite and tiempo_inicio <= tiempo: #mid game
                        fila_unidades_mid[pos_unidades_mid] = [unidad,int(tiempo)]
                        pos_unidades_mid+=1
                        vacio_unidades_mid = False
                        
            elif int(tiempo) == 0 or int(tiempo) > tiempo_limite:
                i+=1
                continue
            elif int(tiempo) < tiempo_inicio: #early game
                estructura = mapeo_numero_estructura[estructura]
                fila_estructuras[pos] = [estructura, int(tiempo)]
                pos+=1
                vacio = False
            elif int(tiempo) <= tiempo_limite: #mid game
                estructura = mapeo_numero_estructura[estructura]
                fila_estructuras_mid[pos_mid] = [estructura, int(tiempo)]
                pos_mid+=1
                vacio_mid = False
                    #print(estructura+"*")
                    
                    #print(tiempos[t])
            i+=1
        pos_mejora=0
        pos_mejora_mid=0
        
        for k in range(len(mejoras)):
            mejora = mejoras[k]
            tiempo = row[mejora]
            if tiempo != 0 and tiempo < tiempo_inicio:
                fila_mejoras[pos_mejora][0] = int(mejora)
                fila_mejoras[pos_mejora][1] = tiempo
                pos_mejora+=1
                vacio_mejoras=False
            elif tiempo != 0 and tiempo <= tiempo_limite:
                fila_mejoras_mid[pos_mejora_mid][0] = mapeo_mejora_numero[mejora]
                fila_mejoras_mid[pos_mejora_mid][1] = tiempo
                pos_mejora_mid+=1
                vacio_mejoras_mid=False
        #print(fila)
        if (not(vacio) or not(vacio_unidades) or not(vacio_mejoras)) and        (not(vacio_mid) or not(vacio_unidades_mid) or not(vacio_mejoras_mid)):
            
            fila_estructuras = sorted(fila_estructuras, key=lambda x: x[1])   
            for posicion in range(len(fila_estructuras)):
                if fila_estructuras[posicion][1] == 999:
                    fila_estructuras[posicion][1] = 0
                    
                #fila_denso.append(fila_estructuras[posicion][0])
                #fila_denso.append(fila_estructuras[posicion][1])
                
                fila_final.append(fila_estructuras[posicion][0])
                
                #fila_final.append(fila[posicion][1])
                vocabulario.append(fila_estructuras[posicion][0])
                #vocabulario.append(fila[posicion][1])
                
            fila_estructuras_mid = sorted(fila_estructuras_mid, key=lambda x: x[1])
            for posicion in range(len(fila_estructuras_mid)):
                if fila_estructuras_mid[posicion][1] == 999:
                    fila_estructuras_mid[posicion][1] = 0
                    
                #fila_denso.append(fila_estructuras[posicion][0])
                #fila_denso.append(fila_estructuras[posicion][1])
                
                fila_final_mid.append(fila_estructuras_mid[posicion][0])
                
                #fila_final.append(fila[posicion][1])
                vocabulario.append(fila_estructuras_mid[posicion][0])
                #vocabulario.append(fila[posicion][1])
                
                
            fila_unidades = sorted(fila_unidades, key=lambda x: x[1])
            
            for posicion in range(len(fila_unidades)):
                if fila_unidades[posicion][1] == 999:
                    fila_unidades[posicion][1] = 0
                fila_final_unidades.append(fila_unidades[posicion][0])
                vocabulario_unidades.append(fila_unidades[posicion][0])
            
            fila_unidades_mid = sorted(fila_unidades_mid, key=lambda x: x[1])
            
            for posicion in range(len(fila_unidades_mid)):
                if fila_unidades_mid[posicion][1] == 999:
                    fila_unidades_mid[posicion][1] = 0
                fila_final_unidades_mid.append(fila_unidades_mid[posicion][0])
                vocabulario_unidades.append(fila_unidades_mid[posicion][0])
            
            
            fila_mejoras = sorted(fila_mejoras, key=lambda x: x[1])
            
            for posicion in range(len(fila_mejoras)):
                if fila_mejoras[posicion][1] == 999:
                    fila_mejoras[posicion][1] = 0
                fila_final_mejoras.append(fila_mejoras[posicion][0])
                vocabulario_mejoras.append(fila_mejoras[posicion][0])
                
            fila_mejoras_mid = sorted(fila_mejoras_mid, key=lambda x: x[1])
            
            for posicion in range(len(fila_mejoras_mid)):
                if fila_mejoras_mid[posicion][1] == 999:
                    fila_mejoras_mid[posicion][1] = 0
                fila_final_mejoras_mid.append(fila_mejoras_mid[posicion][0])
                vocabulario_mejoras.append(fila_mejoras_mid[posicion][0])
            #print(len(fila_final_mejoras))
            #output.append([x[0] for x in fila])
            output_rnn.append(fila_final)
            output_rnn_mejoras.append(fila_final_mejoras)
            output_rnn_unidades.append(fila_final_unidades)
            
            output_rnn_mid.append(fila_final_mid)
            output_rnn_unidades_mid.append(fila_final_unidades)
            output_rnn_mejoras_mid.append(fila_final_mejoras)
        else:
            vacio_mapa.append(index)
            y_target=y_target.drop(index)
            df = df.drop(index)
    vocabulario = convertirATupla(vocabulario)
    vocabulario = set(vocabulario)
    
    vocabulario_unidades = convertirATupla(vocabulario_unidades)
    vocabulario_unidades = set(vocabulario_unidades)
    
    vocabulario_mejoras = convertirATupla(vocabulario_mejoras)
    vocabulario_mejoras = set(vocabulario_mejoras)
    
    x_batch_rnn = np.reshape(output_rnn,[df.shape[0],largo])
    x_batch_rnn_unidades = np.reshape(output_rnn_unidades,[df.shape[0],largo_unidades])
    x_batch_rnn_mejoras = np.reshape(output_rnn_mejoras,[df.shape[0],largo_mejoras])
    
    x_batch_rnn_mid = np.reshape(output_rnn_mid,[df.shape[0],largo])
    x_batch_rnn_unidades_mid = np.reshape(output_rnn_unidades_mid,[df.shape[0],largo_unidades])
    x_batch_rnn_mejoras_mid = np.reshape(output_rnn_mejoras_mid,[df.shape[0],largo_mejoras])
    
    return x_batch_rnn, x_batch_rnn_unidades, x_batch_rnn_mejoras,x_batch_rnn_mid, x_batch_rnn_unidades_mid, x_batch_rnn_mejoras_mid, y_target, vacio_mapa, vocabulario, vocabulario_unidades, vocabulario_mejoras


def inputRecurrenteUnico(df,y_target,main,natural,tiempo_limite):#esta funcion retorna un unico arreglo con toda la informacion del input, en lugar de 3 como la funcion de arriba
    #columnas = [col for col in df.columns if "_x_" in col or "_y_" in col]
    tiempos = [col for col in df.columns if "_t" in col and col != "Primer_tech"]
    upgrades = [col for col in df.columns if col in mejoras]
    #print(len(tiempos))
    output = []
    #output_rnn_unidades=[]
    #output_rnn_mejoras = []
    vocabulario = []
    #vocabulario_unidades = []
    #vocabulario_mejoras = []
    #largo = 30 #336 tiempo
    largo_unidades = 200
    largo_mejoras = len(mejoras)
    largo = 300 
    vacio_mapa=[]
    for index, row in df.iterrows():
        #fila=np.full((15,2), ["999",999],dtype=dtype)
        #fila=np.full((15,2), 999, dtype=object)
        #print("Going:"+str(index))
        vacio = True
        vacio_unidades = True
        vacio_mejoras = True
        fila_estructuras=[[0, 999] for _,_ in enumerate(range(largo))]
        fila_unidades=[[0, 999] for _,_ in enumerate(range(largo_unidades))]
        fila_mejoras=[[0, 999] for _,_ in enumerate(range(largo_mejoras))]
        fila_final=[]
        fila_final_mejoras=[]
        fila_final_unidades=[]
        #fila = sorted(fila, key=lambda x: x[1], reverse=True)
       # print(fila)
        pos=0
        pos_unidades=0
        i=0
        while i < len(tiempos):
            #x = row[columnas[i]]
            #y = row[columnas[i+1]
            frases = tiempos[i].split("_")
            estructura = frases[0]
            string_end = frases[1]
            if estructura in tier_1: 
                i+=1
                continue
            tiempo = row[tiempos[i]]
            #print(tiempos[i]+":"+str(tiempo))
            '''
            if string_end == "tiempo":
                if estructura == "Probe":
                    #print("Probe detectada")
                    i+=1
                    continue #no contamos probes por el problema de inconsistencia al comienzo: algunos seleccionan las 12 probes, otros no
            '''
            #frases = tiempos[t].split("_")
            #estructura = frases[0]
            #string_end = frases[1]
            
            #procesar listado de unidades
            
            if string_end == "tiempo":
                unidad = mapeo_unidades_tiempo[estructura]
                if (isinstance(tiempo,str)):
                    tiempo = tiempo.strip("[").strip("]")
                    tiempo = tiempo.split(",")
                    for elemento in tiempo:
                        #print(str(unidad)+":"+elemento)
                        if int(elemento) > tiempo_limite:
                            break
                        if int(elemento) != 0:
                            fila_unidades[pos_unidades] = [unidad,int(elemento)]
                            pos_unidades+=1
                            vacio_unidades = False
                else:
                    if tiempo <= tiempo_limite and tiempo != 0:
                        fila_unidades[pos_unidades] = [unidad,int(tiempo)]
                        pos_unidades+=1
                        vacio_unidades = False
            elif int(tiempo) == 0 or int(tiempo) > tiempo_limite:
                i+=1
                continue
            else:
                estructura = mapeo_numero_estructura[estructura]
                fila_estructuras[pos] = [estructura, int(tiempo)]
                pos+=1
                vacio = False
                    #print(estructura+"*")
                    
                    #print(tiempos[t])
            i+=1
        pos_mejora=0
        for k in range(len(mejoras)):
            mejora = mejoras[k]
            tiempo = row[mejora]
            if tiempo != 0 and tiempo <= tiempo_limite:
                fila_mejoras[pos_mejora][0] = mapeo_mejora_numero[mejora]
                fila_mejoras[pos_mejora][1] = tiempo
                pos_mejora+=1
                vacio_mejoras=False
        #print(fila)
        if not(vacio) or not(vacio_unidades) or not(vacio_mejoras):
            fila_estructuras = sorted(fila_estructuras, key=lambda x: x[1])
                
            for posicion in range(len(fila_estructuras)):
                if fila_estructuras[posicion][1] == 999:
                    fila_estructuras[posicion][1] = 0
                    
                #fila_denso.append(fila_estructuras[posicion][0])
                #fila_denso.append(fila_estructuras[posicion][1])
                
                output.append(fila_estructuras[posicion][0])
                
                #fila_final.append(fila[posicion][1])
                vocabulario.append(fila_estructuras[posicion][0])
                #vocabulario.append(fila[posicion][1])
                
            fila_unidades = sorted(fila_unidades, key=lambda x: x[1])
            
            for posicion in range(len(fila_unidades)):
                if fila_unidades[posicion][1] == 999:
                    fila_unidades[posicion][1] = 0
                output.append(fila_unidades[posicion][0])
                vocabulario.append(fila_unidades[posicion][0])
            
            fila_mejoras = sorted(fila_mejoras, key=lambda x: x[1])

            
            for posicion in range(len(fila_mejoras)):
                if fila_mejoras[posicion][1] == 999:
                    fila_mejoras[posicion][1] = 0
                output.append(fila_mejoras[posicion][0])
                vocabulario.append(fila_mejoras[posicion][0])
            #print(len(fila_final_mejoras))
            #output.append([x[0] for x in fila])
        else:
            vacio_mapa.append(index)
            y_target=y_target.drop(index)
            df = df.drop(index)
    vocabulario = convertirATupla(vocabulario)
    vocabulario = set(vocabulario)

    
    x_batch_rnn = np.reshape(output,[df.shape[0],largo+largo_mejoras+largo_unidades])
    
    return df,x_batch_rnn, y_target, vacio_mapa, vocabulario

def inputRecurrenteUnicoOrdenado(df,y_target,main,natural,tiempo_limite):#ordena el input final secuencialmente de manera completa. La funcion de arriba solo ordena elementos de una misma categoria e ignora la participacion del resto
    #columnas = [col for col in df.columns if "_x_" in col or "_y_" in col]
    tiempos = [col for col in df.columns if "_t" in col and col != "Primer_tech"]
    upgrades = [col for col in df.columns if col in mejoras]
    #print(len(tiempos))
    output = []
    #output_rnn_unidades=[]
    #output_rnn_mejoras = []
    vocabulario = []
    #vocabulario_unidades = []
    #vocabulario_mejoras = []
    #largo = 30 #336 tiempo
    largo_mejoras = len(mejoras)
    largo = 300 
    largo_unidades = 200 + largo + largo_mejoras
    vacio_mapa=[]
    for index, row in df.iterrows():
        #fila=np.full((15,2), ["999",999],dtype=dtype)
        #fila=np.full((15,2), 999, dtype=object)
        #print("Going:"+str(index))
        vacio = True
        vacio_unidades = True
        vacio_mejoras = True
        #fila_estructuras=[[0, 999] for _,_ in enumerate(range(largo))]
        fila_unidades=[[0, 999] for _,_ in enumerate(range(largo_unidades))]
        #fila_mejoras=[[0, 999] for _,_ in enumerate(range(largo_mejoras))]
        fila_final=[]
        fila_final_mejoras=[]
        fila_final_unidades=[]
        #fila = sorted(fila, key=lambda x: x[1], reverse=True)
       # print(fila)
        pos=0
        pos_unidades=0
        i=0
        while i < len(tiempos):
            #x = row[columnas[i]]
            #y = row[columnas[i+1]
            frases = tiempos[i].split("_")
            estructura = frases[0]
            string_end = frases[1]
            if estructura in tier_1: 
                i+=1
                continue
            tiempo = row[tiempos[i]]
            #print(tiempos[i]+":"+str(tiempo))
            '''
            if string_end == "tiempo":
                if estructura == "Probe":
                    #print("Probe detectada")
                    i+=1
                    continue #no contamos probes por el problema de inconsistencia al comienzo: algunos seleccionan las 12 probes, otros no
            '''
            #frases = tiempos[t].split("_")
            #estructura = frases[0]
            #string_end = frases[1]
            
            #procesar listado de unidades
            
            if string_end == "tiempo":
                unidad = mapeo_unidades_tiempo[estructura]
                if (isinstance(tiempo,str)):
                    tiempo = tiempo.strip("[").strip("]")
                    tiempo = tiempo.split(",")
                    for elemento in tiempo:
                        #print(str(unidad)+":"+elemento)
                        if int(elemento) > tiempo_limite:
                            break
                        if int(elemento) != 0:
                            fila_unidades[pos] = [unidad,int(elemento)]
                            pos+=1
                            vacio_unidades = False
                else:
                    if tiempo <= tiempo_limite and tiempo != 0:
                        fila_unidades[pos] = [unidad,int(tiempo)]
                        pos+=1
                        vacio_unidades = False
            elif int(tiempo) == 0 or int(tiempo) > tiempo_limite:
                i+=1
                continue
            else:
                estructura = mapeo_numero_estructura[estructura]
                fila_unidades[pos] = [estructura, int(tiempo)]
                pos+=1
                vacio = False
                    #print(estructura+"*")
                    
                    #print(tiempos[t])
            i+=1
        pos_mejora=0
        for k in range(len(mejoras)):
            mejora = mejoras[k]
            tiempo = row[mejora]
            if tiempo != 0 and tiempo <= tiempo_limite:
                fila_unidades[pos][0] = mapeo_mejora_numero[mejora]
                fila_unidades[pos][1] = tiempo
                pos+=1
                vacio_mejoras=False
        #print(fila)
        if not(vacio) or not(vacio_unidades) or not(vacio_mejoras):
            salida = []
            
            '''
            fila_estructuras = sorted(fila_estructuras, key=lambda x: x[1])
                
            for posicion in range(len(fila_estructuras)):
                if fila_estructuras[posicion][1] == 999:
                    fila_estructuras[posicion][1] = 0
                    
                #fila_denso.append(fila_estructuras[posicion][0])
                #fila_denso.append(fila_estructuras[posicion][1])
                
                #output.append(fila_estructuras[posicion][0])
                #fila_final.append(fila[posicion][1])
                vocabulario.append(fila_estructuras[posicion][0])
                #vocabulario.append(fila[posicion][1])
            '''        
            fila_unidades = sorted(fila_unidades, key=lambda x: x[1])
            
            for posicion in range(len(fila_unidades)):
                if fila_unidades[posicion][1] == 999:
                    fila_unidades[posicion][1] = 0
                output.append(fila_unidades[posicion][0])
                vocabulario.append(fila_unidades[posicion][0])
            '''      
            fila_mejoras = sorted(fila_mejoras, key=lambda x: x[1])

            
            for posicion in range(len(fila_mejoras)):
                if fila_mejoras[posicion][1] == 999:
                    fila_mejoras[posicion][1] = 0
                output.append(fila_mejoras[posicion][0])
                vocabulario.append(fila_mejoras[posicion][0])
            #print(len(fila_final_mejoras))
            #output.append([x[0] for x in fila])
            '''  
        else:
            vacio_mapa.append(index)
            y_target=y_target.drop(index)
            df = df.drop(index)
    vocabulario = convertirATupla(vocabulario)
    vocabulario = set(vocabulario)

    
    x_batch_rnn = np.reshape(output,[df.shape[0],largo_unidades])
    
    return df,x_batch_rnn, y_target, vacio_mapa, vocabulario

def transformarOneHot(y):
# encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoder_mapeo = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return encoder_mapeo, dummy_y




_,x_batch_rnn, y_target, vacio_mapa, vocabulario = inputRecurrenteMixto(X,y2,False,False,700)

encoder_mid, dummy_y_mid = transformarOneHot(y_target)
categorias = dummy_y_mid.shape[1]
print(encoder_mid.values())
print(categorias)

X_train,X_test,y_train, y_test=  train_test_split(x_batch_rnn,dummy_y_mid,test_size=0.20, random_state=59)

with open(ruta,"wb") as arc:
    pickle.dump(X_train,arc)

ruta = "X_test"

with open(ruta,"wb") as arc:
    pickle.dump(X_test,arc)


ruta = "y_train"
with open(ruta,"wb") as arc:
    pickle.dump(y_train,arc)

ruta = "y_test"
with open(ruta,"wb") as arc:
    pickle.dump(y_test,arc)

ruta = "vocabulario"

with open(ruta,"wb") as arc:
    pickle.dump(vocabulario,arc)
