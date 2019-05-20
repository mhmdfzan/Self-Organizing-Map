import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
import random

dataset = pandas.read_csv("Dataset.csv", header=None)

def eucledian(x, y):
    return math.sqrt((x['x']-y['x'])**2 + (x['y']-y['y'])**2)

def normalisasi(data):
    nmin = {'x': min([n['x'] for n in data]), 'y': min([n['y'] for n in data])}
    nmax = {'x': max([n['x'] for n in data]), 'y': max([n['y'] for n in data])}
    for i in data:
        i['x'] = (i['x'] - nmin['x']) / (nmax['x'] - nmin['x']) 
        i['y'] = (i['y'] - nmin['y']) / (nmax['y'] - nmin['y']) 
    return data

def plot(koor,judul):
    koor.set(xlabel='X', ylabel='Y', title=judul)
    koor.grid()
    plt.axis([-0.2,1.2,-0.2,1.2])
    plt.show()

data = []
clusters = []
color = ['lawngreen', 'maroon', 'gray', 'hotpink', 'deepskyblue', 'purple', 'red', 'orange', 'yellow', 'aqua', 'brown']

for i in range(len(dataset)):
    data.append({'x' : dataset[0][i], 'y': dataset[1][i], 'color': "sienna"})

lr = 0.000001
tLr = 2
sig = 2
tsig = 2
dataTrain = normalisasi(data)

neuron = []
neuron_size = 1200
for i in range(neuron_size):
    neuron.append({'x': random.uniform(0, 1), 'y': random.uniform(0, 1), 'color': 'silver', 'status': 'not'})

fig, ax = plt.subplots()
for i in neuron:
    ax.plot(i['x'], i['y'], ".", color=i['color'])
for i in dataTrain:
    ax.plot(i['x'], i['y'], ".", color=i['color'])

plot(ax,'Mapping Awal')

iteration = 20
konvergen = 0.0000000000001
for t in range(iteration):
    rand = random.randint(1, len(data)-1)
    x = data[rand]
    
    win_neuron = neuron[0]
    for a in neuron:
        a['status'] = 'neuron'
        if eucledian(x, a) < eucledian(x, win_neuron): 
            win_neuron = a
    
    for b in neuron:
        if eucledian(b, win_neuron) < sig:
            b['status'] = 'neighborhood'
    
    for c in neuron:
        if (c['status'] == 'neighborhood'):
            s = eucledian(win_neuron, c)
            phi = np.exp(-(s**2 / (2*sig**2)))
            dWeight = lr * phi * eucledian(x, c)  #update weight
            c['x'] += dWeight
            c['y'] += dWeight
            
    if (dWeight < konvergen):
        print("Iterasi: ",t)
        break
    
    if (win_neuron not in clusters): 
        clusters.append(win_neuron)
        #update learning rate dan sigma, (tidak konvergen)
        lr *= np.exp(-t/tLr)
        tsig *= np.exp(-t/tsig)

for i in range(len(clusters)):
    clusters[i]['color'] = color[i]
    
#clusters dataset
for d in dataTrain:
    win_clusters = clusters[0]
    for clust in clusters:
        if eucledian(d, clust) < eucledian(d, win_clusters): 
            win_clusters = clust
    d['color'] = win_clusters['color']


print("Jumlah clusters: ", len(clusters))

fig, ax = plt.subplots()
for i in dataTrain:
    ax.plot(i['x'], i['y'], ".", color=i['color'])

plot(ax,'Mapping Hasil Akhir')
