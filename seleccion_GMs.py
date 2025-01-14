# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:13:41 2024

@author: HOME
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d

name = 'spectrums'
file = open(name, 'rb') 
GMSa = pickle.load(file)
file.close()

name = 'fourier'
file = open(name, 'rb') 
fourier = pickle.load(file)
file.close()

#%% Para escoger los GMs

choice = 'rand' # puede ser 'rand' o 'fourier'
GM_per_bin = 20 # cantidad de GM deseados por un bin dado
T = 0.30 # periodo de la estructura
#%%

def nbins(data):
    ''' Esta función recibe unos datos (data) y devuelve el número de bines óptimo según la regla de 
    Freedman-Diaconis'''
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    n = len(data)
    # Calculate the bin width using the Freedman-Diaconis rule
    bin_width = 2 * IQR / (n ** (1/3))
    # Calculate the range of the data
    data_range = np.max(data) - np.min(data)
    # Calculate the number of bins
    num_bins = int(np.ceil(data_range / bin_width))
    return num_bins

#%% Esta sección crea un DataFrame que contien la información de los registros 
 #GMSa contiene una lista con:
     #nombre de los records, intensidad de arias, periodos del espectro, espectros, periodos para espectro SaAvg, espectros SaAvg

Ta, Tb = max(0.3*T,0.04), 1.5*T # límites para examinar las frecuencias de Fourier
Ts2 = np.linspace(Ta,Tb,100)

GM_names = GMSa[0]
Ts = GMSa[2]
Tavg = GMSa[4]
Sp = GMSa[3]
SpAvg = GMSa[5]
SaT = np.zeros(len(Sp))
SaAvg = np.zeros(len(Sp))
fourie = np.zeros(len(fourier))

for ind,sp in enumerate(Sp):
    SaT[ind] = np.interp(T,Ts,sp)

for ind,saavg in enumerate(SpAvg):
    SaAvg[ind] = np.interp(T,Tavg,saavg)

for ind,four in enumerate(fourier):
    ff_f = interp1d(four[0], four[1])
    fou = ff_f(Ts2)
    aaa = np.trapz(fou,Ts2)
    fourie[ind] = aaa
    
dicc = {'Name': GM_names, 'Sa': SaT, 'SaAvg': SaAvg, 'Fourier': fourie}
df = pd.DataFrame(dicc)

sns.histplot(data=SaT)
plt.show()
# sns.histplot(data=SaAvg)
# plt.show()
# sns.histplot(data=fourie)
# plt.show()



#%% Esta sección se utiliza para hacer un bineado estadístico
# y extraer cuántos registros quedan en cada uno de los bins, así como su índice

conteo2, bins2 = np.histogram(df['Sa'],bins='fd')
bin_assignments = np.digitize(df['Sa'], bins2)

# Initialize a dictionary to hold indexes for each bin
indexes_per_bin = {bin_number: [] for bin_number in range(1, len(bins2))}

# Populate the dictionary with indexes of data points for each bin
for index, bin_number in enumerate(bin_assignments):
    if bin_number in indexes_per_bin:
        indexes_per_bin[bin_number].append(index)



#%% Exploracion de lo que pasa en un bin

bb = df.iloc[indexes_per_bin[1]]
bb.Sa.describe()

# bins2 devuelve los bordes del bin, es decir, en indexes_per_bin 
# están los índices que están entre los valores consecutivos de bins2


#%% Para seleccionar la lista de los 
index_lim = np.min(np.where(conteo2 < GM_per_bin))         # indice del bin hasta donde hay al menos los GM deseados
GMlist = []
bin_midpoint = []
if choice == 'rand':
    for i in range(index_lim):
        aa = random.sample(indexes_per_bin[i+1],GM_per_bin)
        midpoint = (bins2[i]+bins2[i+1])/2
        bin_midpoint.extend([midpoint]*GM_per_bin)
        GMlist.extend(aa)

if choice == 'fourier':
    for i in range(index_lim):
        # filtered_df = df[df.index.isin(indexes_per_bin[i+1])]
        # sorted_df = filtered_df.sort_values(by='Fourier')
        # first_20 = sorted_df.head(int(GM_per_bin))
        aa = df.loc[df.index.isin(indexes_per_bin[i+1])].sort_values('Fourier',ascending=False).index[:int(GM_per_bin)]
        midpoint = (bins2[i]+bins2[i+1])/2
        bin_midpoint.extend([midpoint]*GM_per_bin)
        GMlist.extend(aa)
# GMlist devuelve los índices de los GMs que se analizarán

#%% Finalmente se consolida todo en un DataFrame
mpoints = np.array(bin_midpoint)
df2 = df.iloc[GMlist]
df2['bin'] = mpoints

nsteps = np.loadtxt('Nsteps.txt')
dt = np.loadtxt('dt.txt')

df2['Nsteps'] = nsteps[GMlist]
df2['DTs'] = dt[GMlist]

print('Se han seleccionado',len(GMlist), 'registros', 'con IM entre', round(min(bin_midpoint),2), 'y', 
      round(max(bin_midpoint),2))


# df2.to_pickle('GM.pkl')
df2.to_pickle('GM'+'T'+str(T)+'.pkl')

#%%

sns.boxplot(data=df2, x = 'bin', y = 'Fourier')


#%% Test de lo de Fourier
