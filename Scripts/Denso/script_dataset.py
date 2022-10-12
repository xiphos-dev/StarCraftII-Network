#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np


# In[2]:


ruta = "../../"
archivo = "protoss.csv"
file = ruta+archivo


# In[3]:


df = pd.read_csv(file, sep=',', dtype={"Label": str})
#df.head(20)


# In[4]:


#df = df[df["Probe"] != 0]
#df.head(-20)


# ##  Preparando dataset

# In[5]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# In[6]:


import matplotlib.pyplot as plt
def graficar(historia):
    loss = historia.history['accuracy']
    val_loss = historia.history['val_accuracy']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'y', label='Training acc')
    plt.plot(epochs, val_loss, 'r', label="Validation acc")
    plt.title('Training and Validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()


# In[7]:


valores = ["1 base no tech", "2 base no tech", "3 base no tech"]
estructuras = ["Pylon",
               "Gateway",
               #"WarpGate",
              # "Battery",
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

mapeo_unidades_tiempo ={unidad:numero for numero,unidad in enumerate(unidades,20)}
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

tier_1 = ["Stargate",'TwilightCouncil','RoboticsFacility','DarkShrine','TemplarArchives']

mapeo_estructura_numero = {valor: llave for llave,valor in mapeo_numero_estructura.items()}

valores = df["Label"].value_counts()

for llave, valor in valores.items():
    if valor < 500 or "no tech" in llave or "4 base" in llave:
        del valores[llave]
        
#del valores["1 Hatch "] 
#del valores["2 Hatch "] 
#del valores["3 Hatch "] 



builds = df[df["Label"].isin(valores.keys())]

builds_objetivo = builds.Label.unique().tolist()

base_1 = [build for build in builds_objetivo if "3 base" in build]
print(base_1)


# In[10]:


def primeraTech(fila, cota_tiempo=336):
    #1:24 minutos in-replay time toma para que transcurra 1 minuto in game
    #1 minuto in game = 1:24 = 84 segundos reales
    #2 minutos in game = 2:48 = 168 segundos reales
    #3 minutos in game = 4:12 = 252 segundos reales
    #4 minutos in game = 5:36 = 336 segundos reales
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
            if minimo_tiempo < 256:#podria ser un stargate en el muro
                flag_wall = True
        elif fila[tech[i]+"_t_1"] < cota_tiempo and fila[tech[i]+"_t_1"] != 0:
            flag_wall = False
            tech_elegida = tech[i]
            minimo_tiempo = fila[tech[i]+"_t_1"]
    return tech_elegida


# In[11]:


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
    #print(listado_valores)
    for i in range(largo):
        if pos<len(listado_posiciones):
            if listado_posiciones[pos] == i:
                #output.append(listado_valores[0,pos])
                output.append(listado_valores[pos])
                pos+=1
            else:
                output.append(0)
        else:
            output.append(0)
    return output


# In[12]:



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


# In[13]:



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



def inputDensoRandomSelection(df, tiempo_limite, probabilidad_seleccion=0.8, filtrar_estructuras=True,filtrar_mejoras=False,filtrar_unidades=False):#esta funcion descarta aleatoriamente algunos datos para simular las observaciones incompletas dentro de una partida real
    n_unidades = 50
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


# In[14]:


def inputDensoDividido(df, tiempo_early,tiempo_limite):#esta funcion retorna los inputs dividos entre early y mid game. La division se hace en base a las cotas de tiempo
    output = []
    output_mid = []
    for index, row in df.iterrows():
            
        posiciones=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        posiciones_mid=[]
        k=0
        tiempos_primeros = []
        tiempos_primeros_mid = []
        for estructura in estructuras:
            if estructura not in (tier_1):
                tiempo_actual = row[estructura+"_t_1"]
                if tiempo_actual <= tiempo_limite and tiempo_actual != 0:
                   # print("Added:"+estructura)
                    tiempos_primeros.append(tiempo_actual)
                    posiciones.append(k)
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


# In[15]:


'''
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
'''


def inputDensoRandomSelectionConDatoClave(df, tiempo_limite, probabilidad_seleccion=0.8, filtrar_estructuras=True,filtrar_mejoras=False,filtrar_unidades=False):
    
    n_unidades = 300
    output = []

    for index, row in df.iterrows():
        escalados = [] #arreglo que contiene todos los tiempos de esta fila del dataset, usado para escalarlos todos juntos al final
        posiciones_estructuras=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_primeros = []
        for estructura in estructuras:
            if estructura not in (tier_1):
                tiempo_actual = row[estructura+"_t_1"]
                if tiempo_actual <= tiempo_limite: #tiempo actual puede ser cero
                   # print("Added:"+estructura)
                    if filtrar_estructuras:
                        umbral_seleccion = np.random.random_sample()
                        if umbral_seleccion <= probabilidad_seleccion:
                            tiempos_primeros.append(tiempo_actual)
                            posiciones_estructuras.append(k)
                            escalados.append(tiempos_primeros[-1])
                        else: 
                            tiempos_primeros.append(0)
                    else:
                        tiempos_primeros.append(tiempo_actual)
            k+=1

        #tiempos_primeros,scaler = escalar(tiempos_primeros)
        #output = recomponerListado(len(estructuras)-len(tier_1),posiciones,tiempos_primeros,output)
        

        posiciones_mejoras=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_mejora = []
        for mejora in mejoras:
            tiempo_mejora = row[mejora]
            if tiempo_mejora <= tiempo_limite:
                if filtrar_mejoras:
                    umbral_seleccion = np.random.random_sample()
                    if umbral_seleccion <= probabilidad_seleccion:
                        tiempos_mejora.append(tiempo_mejora)
                        posiciones_mejoras.append(k)
                        escalados.append(tiempos_mejora[-1])

                    else:
                        tiempos_mejora.append(0)
                else:
                    tiempos_mejora.append(tiempo_mejora)

            k+=1
            
        #tiempos_mejora,_ = escalar(tiempos_mejora,scaler)
        #output = recomponerListado(len(mejoras),posiciones,tiempos_mejora,output)
        
        orden_unidades=[0] * n_unidades # lista que contiene la secuencia de unidades creadas
        pos = 0
        cantidad_por_unidad = []
        for unidad in unidades:
            tiempo_primero = 0
            tiempo_ultimo = 0
            cantidad_por_unidad.append(row[unidad])
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
            
            escalados.append(tiempo_primero)
            escalados.append(tiempo_ultimo)

            
            #tiempo_primero,_ = escalar([tiempo_primero],scaler)
            #output.append(tiempo_primero[0,0])
            #tiempo_ultimo,_ = escalar([tiempo_ultimo],scaler)
            #output.append(tiempo_ultimo[0,0])

        
        escalados,_ = escalar(escalados)
        estructuras_final = escalados[0][:len(tiempos_primeros)]
        mejoras_final = escalados[0][len(tiempos_primeros):len(tiempos_primeros)+len(tiempos_mejora)]
        unidades_final = escalados[0][len(tiempos_primeros)+len(tiempos_mejora):]
        
        output = recomponerListado(len(estructuras)-len(tier_1),posiciones_estructuras,estructuras_final,output)
        output = recomponerListado(len(mejoras),posiciones_mejoras,mejoras_final,output) 
        i=0
        k=0
        while i < len(unidades_final):
            output.append(cantidad_por_unidad[k])
            output.append(unidades_final[i])
            output.append(unidades_final[i+1])
            i+=2
            k+=1
            
        for unidad in orden_unidades:
            output.append(unidad)

    #output = np.reshape(output,(df.shape[0],len(orden_unidades)+len(estructuras)-len(tier_1)+len(mejoras)))
    output = np.reshape(output,(df.shape[0],len(orden_unidades)+len(unidades)*3+len(estructuras)-len(tier_1)+len(mejoras)))
    return output


# In[16]:


#esta funcion tiene un problema desde el punto de vista filosofico: distinguir entre las categorias de existente y conocido requiere la presencia de un observador omnisciente que puede determinar cuales elementos son "0" y cuales son "-1". Para confirmar un "0" hace falta saber, con absoluta certeza, que cierto elemento no esta presente. Dicha certeza es imposible de obtener dentro de una partida, al menos en el mid game.
def inputDensoRandomSelectionConMascara(df, tiempo_limite, probabilidad_seleccion=0.8, filtrar_estructuras=True,filtrar_mejoras=False,filtrar_unidades=False):#esta funcion descarta aleatoriamente algunos datos para simular las observaciones incompletas dentro de una partida real
    
    n_unidades = 300
    output = []

    for index, row in df.iterrows():
        escalados = [] #arreglo que contiene todos los tiempos de esta fila del dataset, usado para escalarlos todos juntos al final
        posiciones_estructuras=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_primeros = []
        for estructura in estructuras:
            if estructura not in (tier_1):
                tiempo_actual = row[estructura+"_t_1"]
                if tiempo_actual <= tiempo_limite : #tiempo actual puede ser cero
                   # print("Added:"+estructura)
                    if filtrar_estructuras:
                        umbral_seleccion = np.random.random_sample()
                        if umbral_seleccion <= probabilidad_seleccion:
                            tiempos_primeros.append(tiempo_actual)
                            
                        else: #aplicacion de mascara
                            tiempos_primeros.append(-1)
                        posiciones_estructuras.append(k)
                        escalados.append(tiempos_primeros[-1])
                    else:
                        tiempos_primeros.append(tiempo_actual)
            k+=1

        #tiempos_primeros,scaler = escalar(tiempos_primeros)
        #output = recomponerListado(len(estructuras)-len(tier_1),posiciones,tiempos_primeros,output)
        

        posiciones_mejoras=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_mejora = []
        for mejora in mejoras:
            tiempo_mejora = row[mejora]
            if tiempo_mejora <= tiempo_limite:
                if filtrar_mejoras:
                    umbral_seleccion = np.random.random_sample()
                    if umbral_seleccion <= probabilidad_seleccion:
                        tiempos_mejora.append(tiempo_mejora)
                        
                    else:
                        tiempos_mejora.append(-1)
                    posiciones_mejoras.append(k)
                    escalados.append(tiempos_mejora[-1])

                else:
                    tiempos_mejora.append(tiempo_mejora)

            k+=1
            
        #tiempos_mejora,_ = escalar(tiempos_mejora,scaler)
        #output = recomponerListado(len(mejoras),posiciones,tiempos_mejora,output)
        
        orden_unidades=[0] * n_unidades # lista que contiene la secuencia de unidades creadas
        pos = 0
        cantidad_por_unidad = []
        for unidad in unidades:
            cantidad_por_unidad.append(row[unidad])
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
            
            
            
            escalados.append(tiempo_primero)
            escalados.append(tiempo_ultimo)

            
            #tiempo_primero,_ = escalar([tiempo_primero],scaler)
            #output.append(tiempo_primero[0,0])
            #tiempo_ultimo,_ = escalar([tiempo_ultimo],scaler)
            #output.append(tiempo_ultimo[0,0])

        
        escalados,_ = escalar(escalados)
        estructuras_final = escalados[0][:len(tiempos_primeros)]
        mejoras_final = escalados[0][len(tiempos_primeros):len(tiempos_primeros)+len(tiempos_mejora)]
        unidades_final = escalados[0][len(tiempos_primeros)+len(tiempos_mejora):]
        
        output = recomponerListado(len(estructuras)-len(tier_1),posiciones_estructuras,estructuras_final,output)
        output = recomponerListado(len(mejoras),posiciones_mejoras,mejoras_final,output) 
        
        i=0
        k=0
        
        while i < len(unidades_final):
            output.append(cantidad_por_unidad[k])
            output.append(unidades_final[i])
            output.append(unidades_final[i+1])
            i+=2
            k+=1
            
        for unidad in orden_unidades:
            output.append(unidad)

    #output = np.reshape(output,(df.shape[0],len(orden_unidades)+len(estructuras)-len(tier_1)+len(mejoras)))
    output = np.reshape(output,(df.shape[0],len(orden_unidades)+len(unidades)*3+len(estructuras)-len(tier_1)+len(mejoras)))
    return output


# In[17]:


#aqui se separa la informacion temporal de la categorica. La secuencia de unidades producidas es retornada en un vector separado
########
#revertido para incluir todo el output en un vector unico
def inputDensoRandomDividido(df, tiempo_limite, probabilidad_seleccion=0.8, filtrar_estructuras=True,filtrar_mejoras=False,filtrar_unidades=False):
    
    n_unidades = 300
    output = []
    output_unidades = []
    
    for index, row in df.iterrows():
        escalados = [] #arreglo que contiene todos los tiempos de esta fila del dataset, usado para escalarlos todos juntos al final
        posiciones_estructuras=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_primeros = []
        cantidad_tiempos_primeros = 0 #esta variable recuerda la cantidad de valores no cero que son agregados a la lista de tiempos primeros
        cantidad_mejoras = 0 #lo mismo que arriba, aplicado a mejoras
        cantidad_unidades = 0 #idem
        for estructura in estructuras:
            if estructura not in (tier_1):
                tiempo_actual = row[estructura+"_t_1"]
                if tiempo_actual <= tiempo_limite: #tiempo actual puede ser cero
                   # print("Added:"+estructura)
                    if filtrar_estructuras:
                        umbral_seleccion = np.random.random_sample()
                        if umbral_seleccion <= probabilidad_seleccion:
                            tiempos_primeros.append(tiempo_actual)
                            posiciones_estructuras.append(k)
                            escalados.append(tiempos_primeros[-1])
                            cantidad_tiempos_primeros+=1
                        else: 
                            tiempos_primeros.append(0)
                    else:
                        tiempos_primeros.append(tiempo_actual)
            k+=1

        #tiempos_primeros,scaler = escalar(tiempos_primeros)
        #output = recomponerListado(len(estructuras)-len(tier_1),posiciones,tiempos_primeros,output)
       
        #print("Escalados - Tiempos primeros:"+str(len(escalados)))
        #print("Tiempos primeros:"+str(len(tiempos_primeros)))
        #print("*"*10)

        posiciones_mejoras=[]#este arreglo recuerda las posiciones de los valores distintos a cero, asi permite reconstruir el orden original mas tarde
        k=0
        tiempos_mejora = []
        for mejora in mejoras:
            tiempo_mejora = row[mejora]
            if tiempo_mejora <= tiempo_limite:
                if filtrar_mejoras:
                    umbral_seleccion = np.random.random_sample()
                    if umbral_seleccion <= probabilidad_seleccion:
                        tiempos_mejora.append(tiempo_mejora)
                        posiciones_mejoras.append(k)
                        escalados.append(tiempos_mejora[-1])
                        cantidad_mejoras+=1

                    else:
                        tiempos_mejora.append(0)
                else:
                    tiempos_mejora.append(tiempo_mejora)

            k+=1
            
            
        #print("Escalados - Mejoras :"+str(len(escalados)))
        #print("Mejoras:"+str(len(tiempos_mejora)))
        #print("*"*10)
        #tiempos_mejora,_ = escalar(tiempos_mejora,scaler)
        #output = recomponerListado(len(mejoras),posiciones,tiempos_mejora,output)
        
        orden_unidades=[0] * n_unidades # lista que contiene la secuencia de unidades creadas
        pos = 0
        cantidad_por_unidad = []
        for unidad in unidades:
            cantidad_por_unidad.append(row[unidad])
            tiempo_primero = 0
            tiempo_ultimo = 0
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
                        print("break")
                        break

            escalados.append(tiempo_primero)
            escalados.append(tiempo_ultimo)
            cantidad_unidades+=2
            #print("Len escalados:"+str(len(escalados)))
            #tiempo_primero,_ = escalar([tiempo_primero],scaler)
            #output.append(tiempo_primero[0,0])
            #tiempo_ultimo,_ = escalar([tiempo_ultimo],scaler)
            #output.append(tiempo_ultimo[0,0])

        
        escalados,_ = escalar(escalados)
        estructuras_final = escalados[0][:cantidad_tiempos_primeros]
        mejoras_final = escalados[0][(cantidad_tiempos_primeros):(cantidad_tiempos_primeros)+(cantidad_mejoras)]
        unidades_final = escalados[0][(cantidad_tiempos_primeros)+(cantidad_mejoras):]
        '''
        print(len(tiempos_primeros))
        print(len(estructuras_final))
        print("*"*5)
        print(len(tiempos_mejora))
        print(len(mejoras_final))
        print("*"*5)
        print(len(unidades_final))
        print(len(unidades)*2)
        print("*"*5)
        print(len(escalados[0]))
        '''
        output = recomponerListado(len(estructuras)-len(tier_1),posiciones_estructuras,estructuras_final,output)
        output = recomponerListado(len(mejoras),posiciones_mejoras,mejoras_final,output) 
        
        i=0
        k=0
        #print("Len unidades_final:"+str(len(unidades_final)))
        while i < len(unidades_final):
            output.append(cantidad_por_unidad[k])
            output.append(unidades_final[i])
            output.append(unidades_final[i+1])
            i+=2
            k+=1
            
        for unidad in orden_unidades:
            #output_unidades.append(unidad)
            output.append(unidad)
    #output = np.reshape(output,(df.shape[0],len(orden_unidades)+len(estructuras)-len(tier_1)+len(mejoras)))
    output = np.reshape(output,(df.shape[0],len(unidades)*3+len(estructuras)-len(tier_1)+len(mejoras)+len(orden_unidades)))
    #output_unidades = np.reshape(output_unidades, (df.shape[0],len(orden_unidades)))
    return output


# In[18]:


columnas_coordenadas = [col for col in df.columns if "_x" in col or "_y" in col]
columnas_tiempo = [col for col in df.columns if "_t" in col]

df = df[df["Label"].isin(builds_objetivo)]
#df["Primer_tech"] = df.apply(lambda fila: primeraTech(fila), axis=1)

X = df.drop(columnas_coordenadas, axis=1).drop(["Replay","Label"],axis=1)
#y = df["Primer_tech"]
y = df["Label"]
X.head()

X["Probe"] = X.apply(lambda fila: 12 if fila["Probe"] < 12 else fila["Probe"], axis=1)
X.head()

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[23]:


print(dummy_y.shape)
categorias = dummy_y.shape[1]
print(dummy_y[0])


# In[24]:


input_random = inputDensoRandomDividido(X,700,0.7,True,True,True)
#input_random = inputDensoRandomSelectionConDatoClave(X,700,1,True,True,True)
#input_random = inputDensoRandomSelectionConMascara(X,700,0.7,True,True,True)


# In[25]:


#X_train, X_test, y_train, y_test = train_test_split(input_denso, dummy_y, test_size=0.20, random_state=98)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)
#del X, y, df

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





