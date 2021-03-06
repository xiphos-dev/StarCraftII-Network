#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


ruta = "./"
archivo = "interloper_protoss.csv"
file = ruta+archivo


# In[3]:


df = pd.read_csv(file, sep=',', dtype={"Label": str})
#df.head(20)


# In[4]:


#df = df[df["Probe"] != 0]

df.head(20)


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


'''
df = df[df["Label"].isin(builds_objetivo)]
X = df.drop(["Label"], axis=1).drop(estructuras_tiempo, axis=1)
y = df["Label"]
'''


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
            


# In[46]:


def extrapolarNuevoInput(dataset):
    #p_tech = 0.6 #refleja la probabilidad de observar una estructura clave, en este contexto, que perteneza al arbol tecnologico
    #p_standard = 0.9 #refleja la probabilidad de observar una estructura ordinaria 
    flag_csv = True
    for index, row in dataset.iterrows():
        memoria = np.zeros((len(estructuras_permutables),9))
        for i in range(1,10):#recorre desde *_x_1 hasta *_x_9
            for estructura in estructuras_permutables:
                if estructura+"_x_"+str(i) in row.keys() and row[estructura+"_x_"+str(i)] != 0:
                    canal = mapeo_numero_estructuras_permutables[estructura]#este valor representa la fila en el arreglo 'memoria' que corresponde a este tipo de estructura
                    #por ejemplo, todos los nexus en la primera fila, todos los robo en la segunda, etc.
                    memoria[canal][i-1]=1
            
        for i in range(len(estructuras_permutables)): 
            for k in range(1,9):
                valor = memoria[i][k-1]
                if valor == 0: 
                    break #esto indica que el k-esimo robo,cyber,gateway,etc no existe, por lo que se continua al siguiente tipo de estructura
                else:
                    for j in range(i+1,len(estructuras_permutables)):# se recorren las estructuras hacia abajo, comenzando desde el sucesor directo de la estructura actual
                        #la frase "hacia abajo" y "sucesor" se refieren a que el arreglo que recuerda las coordenadas existentes (variable 'memoria') es recorrido de manera secuencial y en un solo sentido
                        #por lo que nunca se volvera a visitar un nivel anterior, solo se permite continuar hacia un nivel posterior.
                        #Con 'nivel' me refiero a la fila de la  matriz que corresponde a cada estructura (variable 'canal').
                        for q in range(1,9):
                            valor_candidato_permutacion = memoria[j][q-1]
                            if valor_candidato_permutacion == 0: break #mismo caso del break anterior
                            else:
                                #print("Permutando: "+mapeo_estructuras_permutables_numero[i]+ " con " + mapeo_estructuras_permutables_numero[j])
                                dataframe = permutarCoordenadas((i,k),(j,q),row)
                                #print(row["Replay"])
                                #dataframe = pd.concat([row,dataframe])
                                dataframe.to_csv("./extension_dataset_p_interloper.csv", header=flag_csv, index=False, mode="a")
                                flag_csv = False
                                return 0
    
                                

def permutarCoordenadas(estructura_a, estructura_b, fila):
    
    estructura_a_nombre = mapeo_estructuras_permutables_numero[estructura_a[0]]
    estructura_b_nombre = mapeo_estructuras_permutables_numero[estructura_b[0]]
    
    print(estructura_a_nombre+str(estructura_a[1]))
    print(estructura_b_nombre+str(estructura_b[1]))
    
    coordenada_a_x = fila[estructura_a_nombre+"_x_"+str(estructura_a[1])]
    coordenada_a_y = fila[estructura_a_nombre+"_y_"+str(estructura_a[1])]
    
    coordenada_b_x = fila[estructura_b_nombre+"_x_"+str(estructura_b[1])]
    coordenada_b_y = fila[estructura_b_nombre+"_y_"+str(estructura_b[1])]

    df2 = {llave:[valor] for llave,valor in fila.items()}
    #print(df2[estructura_b_nombre+"_y_"+str(estructura_a[1])])
    
    df2[estructura_a_nombre+"_x_"+str(estructura_a[1])][0] = coordenada_b_x
    df2[estructura_a_nombre+"_y_"+str(estructura_a[1])][0] = coordenada_b_y
    
    df2[estructura_b_nombre+"_x_"+str(estructura_b[1])][0] = coordenada_a_x
    df2[estructura_b_nombre+"_y_"+str(estructura_b[1])][0] = coordenada_a_y
    
    df2["index"]=[0]
    
    return pd.DataFrame.from_dict(df2)
    
    


# In[47]:


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


# In[ ]:


extrapolarNuevoInput(df)


# In[14]:




