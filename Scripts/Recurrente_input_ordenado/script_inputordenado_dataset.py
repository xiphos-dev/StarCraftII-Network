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


ruta = "../../"
archivo = "protoss.csv"
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


valores = ["1 base no tech", "2 base no tech", "3 base no tech"]
estructuras = ["Pylon",
               "Gateway",
               #"WarpGate",
               #"Battery",
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
               "Battery_tiempo",
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

mapeo_unidades_tiempo ={unidad:numero for numero,unidad in enumerate(unidades,1)}
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
numero = len(mapeo_unidades_tiempo.keys()) + 1
mapeo_mejora_numero = {mejora: numero for numero,mejora in enumerate(mejoras,numero)}
numero+= len(mapeo_mejora_numero.keys()) 

# In[25]:


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

mapeo_numero_estructura = {llave:numero for numero,llave in enumerate(estructuras,numero)}

mapeo_estructura_numero = {valor: llave for llave,valor in mapeo_numero_estructura.items()}

tier_1 = ["Stargate",'TwilightCouncil','RoboticsFacility','DarkShrine','TemplarArchives']

# In[26]:


def primeraTech(fila, cota_tiempo=336):
    #1:24 minutos in-replay time toma para que transcurra 1 minuto in game
    #1 minuto in game = 1:24 = 84 segundos reales
    #2 minutos in game = 2:48 = 168 segundos reales
    #3 minutos in game = 4:12 = 252 segundos reales
    #4 minutos in game = 5:36 = 336 segundos reales
    #5 minutos in game = 7:00 = 420 segundos reales
    #6 minutos in game = 8:24 = 504 segundos reales
    tech = ["Stargate",'TwilightCouncil','RoboticsFacility','DarkShrine','TemplarArchives']
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

valores = df["Label"].value_counts()
for llave, valor in valores.items():
    if valor < 500:
        del valores[llave]

builds = df[df["Label"].isin(valores.keys())]
builds = builds.Label.unique().tolist()

df = df[df["Label"].isin(builds)]

df["Primer_tech"] = df.apply(lambda fila: primeraTech(fila,336), axis=1)
df.head()

X = df.drop(["Label"], axis=1).drop("Replay", axis=1).drop(columnas_coordenadas, axis=1).drop(estructuras_tiempo, axis=1)
X["Probe"] = df.apply(lambda fila: 12 if fila["Probe"] < 12 else fila["Probe"], axis=1)
X.head()

y = df["Primer_tech"]
y2 = df["Label"]
#y = df["Label"]


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
    largo = 28
    largo_unidades = 50
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
                
            fila_unidades = sorted(fila_unidades, key=lambda x: x[1])
            
            for posicion in range(len(fila_unidades)):
                if fila_unidades[posicion][1] == 999:
                    fila_unidades[posicion][1] = 0
                fila_final_unidades.append(fila_unidades[posicion][0])
                vocabulario_unidades.append(fila_unidades[posicion][0])
            
            fila_mejoras = sorted(fila_mejoras, key=lambda x: x[1])
            
            for posicion in range(len(fila_mejoras)):
                if fila_mejoras[posicion][1] == 999:
                    fila_mejoras[posicion][1] = 0
                fila_final_mejoras.append(fila_mejoras[posicion][0])
                vocabulario_mejoras.append(fila_mejoras[posicion][0])
            #print(len(fila_final_mejoras))
            #output.append([x[0] for x in fila])
            output_rnn.append(fila_final)
            output_rnn_mejoras.append(fila_final_mejoras)
            output_rnn_unidades.append(fila_final_unidades)
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
    
    return df,x_batch_rnn, x_batch_rnn_unidades, x_batch_rnn_mejoras, y_target, vacio_mapa, vocabulario, vocabulario_unidades, vocabulario_mejoras


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

def inputRecurrenteUnicoOrdenadoB(df,y_target,main,natural,tiempo_limite,skip_probes=False):#Esta función retorna un único vector por ejemplo, incluyendo padding para los casos en que corresponda
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
        n_fila = 0
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
            if skip_probes:
                if estructura == "Probe":
                    #print("Probe detectada")
                    i+=1
                    continue
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
                        fila_unidades[pos] = [unidad,int(elemento)]
                        pos+=1
                        vacio_unidades = False
                else:
                    if tiempo <= tiempo_limite:
                        fila_unidades[pos] = [unidad,int(tiempo)]
                        pos+=1
                        vacio_unidades = False
            elif int(tiempo) > tiempo_limite:
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
            if tiempo <= tiempo_limite:
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
            fila_output = []
            for posicion in range(len(fila_unidades)):
                if fila_unidades[posicion][1] == 999:
                    fila_unidades[posicion][1] = 0
                vocabulario.append(fila_unidades[posicion][0])
                valor = 0
                if fila_unidades[posicion][1] != 0:
                    valor = fila_unidades[posicion][0]
                fila_output.append(valor)
            output.append(fila_output)
            n_fila +=1
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

    
    x_batch_rnn = np.asarray(output)
    
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




_,x_batch_rnn, y_mid, vacio_mapa, vocabulario = inputRecurrenteUnicoOrdenadoB(X,y2,False,False,700)

encoder_mid, dummy_y_mid = transformarOneHot(y_mid)
categorias = dummy_y_mid.shape[1]
print(encoder_mid.values())
print(categorias)

X_train_rnn, X_test_rnn, y_train, y_test=  train_test_split(x_batch_rnn, dummy_y_mid,test_size=0.20, random_state=59)

ruta = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"/data_rnn_1/X_train_rnn")

with open("X_train_unico","wb") as arc:
    pickle.dump(X_train_rnn,arc)

ruta = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"/data_rnn_1/X_test_rnn")

with open("X_test_unico","wb") as arc:
    pickle.dump(X_test_rnn,arc)

ruta = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"/data_rnn_1/y_train")
with open("y_train_unico","wb") as arc:
    pickle.dump(y_train,arc)

ruta = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"/data_rnn_1/y_test")
with open("y_test_unico","wb") as arc:
    pickle.dump(y_test,arc)

with open("vocabulario","wb") as arc:
    pickle.dump(vocabulario,arc)

with open("encoder_mid","wb") as arc:
    pickle.dump(encoder_mid,arc)

