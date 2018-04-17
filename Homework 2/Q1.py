import numpy as np
from scipy.special import comb
import networkx as nx 
import matplotlib.pyplot as plt
import itertools
import pandas as pd


#function to calculate probabilties from the uniform random variables
def calculateprobabilities(N,p):
    a=np.zeros(N)
    g=[]
    for j in range (0,N):
        a[j]=np.array(np.random.uniform(0,1))
        if a[j]<p:
            g.append(1)
        else:
            g.append(0)
    return(g)

  
def problem2(n,p):
    N=int(comb(n,2))
    g1=calculateprobabilities(N,p)
    edges=list(itertools.combinations(range(n), 2))  #To generate edges for each possible node

    edge_frames=pd.DataFrame(edges,columns=['Node1','Node2'])
    edge_frames['Edges']=edge_frames[['Node1','Node2']].apply(tuple, axis=1)
    edge_frames['Selection1']=g1
    edge_final1=list(edge_frames.loc[edge_frames['Selection1']==1,'Edges'])  #To find the location where value of edges is 1

    degree1=edge_frames.loc[edge_frames['Selection1']==1,'Node1']  #gives location of nodes where edges have value 1
    degree_count1=pd.Series.value_counts(degree1)  #counts degree of vertex for each node

    G1 = nx.Graph()
    G1.add_edges_from(edge_final1)
    G1.add_nodes_from(range(n))
    plt.figure()
    plt.suptitle('Graph for n=100 and p=0.06')
    nx.draw(G1)
    plt.figure()
    plt.suptitle('Histogram of degree of count for n=100 and p=0.06')
    plt.xlabel('Number of degrees')
    plt.ylabel('Count of each degree')
    plt.hist(degree_count1)
    
n=50
N=int(comb(n,2))
g1=calculateprobabilities(N,0.02)   #calculation with different probabilities
g2=calculateprobabilities(N,0.09)
g3=calculateprobabilities(N,0.12)
edges=list(itertools.combinations(range(n), 2))

edge_frames=pd.DataFrame(edges,columns=['Node1','Node2'])
edge_frames['Edges']=edge_frames[['Node1','Node2']].apply(tuple, axis=1)
edge_frames['Selection1']=g1
edge_frames['Selection2']=g2
edge_frames['Selection3']=g3
edge_final1=list(edge_frames.loc[edge_frames['Selection1']==1,'Edges']) #To find location where value of edge us 1 for 1st set of probabilities
edge_final2=list(edge_frames.loc[edge_frames['Selection2']==1,'Edges']) #To find location where value of edge us 1 for 2nd set of probabilities
edge_final3=list(edge_frames.loc[edge_frames['Selection3']==1,'Edges']) #To find location where value of edge us 1 for 3rd set of probabilities

degree1=edge_frames.loc[edge_frames['Selection1']==1,'Node1']
degree2=edge_frames.loc[edge_frames['Selection2']==1,'Node1']
degree3=edge_frames.loc[edge_frames['Selection3']==1,'Node1']

degree_count1=pd.Series.value_counts(degree1)
degree_count2=pd.Series.value_counts(degree2)
degree_count3=pd.Series.value_counts(degree3)

#plot for graphs for all probabilities
G1 = nx.Graph()
G1.add_edges_from(edge_final1)
G1.add_nodes_from(range(n))
plt.figure()
plt.suptitle('Graph for n=50 and p=0.02')
nx.draw(G1)

G2 = nx.Graph()
G2.add_edges_from(edge_final2)
G2.add_nodes_from(range(n))
plt.figure()
plt.suptitle('Graph for n=50 and p=0.09')
nx.draw(G2)

G3 = nx.Graph()
G3.add_edges_from(edge_final3)
G3.add_nodes_from(range(n))
plt.figure()
plt.suptitle('Graph for n=50 and p=0.12')
nx.draw(G3)

#Histograms for degree of vertex
plt.figure()
plt.hist(degree_count1)
plt.suptitle('Histogram of degree of count for n=50 and p=0.02')
plt.xlabel('Number of degrees')
plt.ylabel('Count of each degree')

plt.figure()
plt.suptitle('Histogram of degree of count for n=50 and p=0.09')
plt.xlabel('Number of degrees')
plt.ylabel('Count of each degree')
plt.hist(degree_count2)

plt.figure()
plt.suptitle('Histogram of degree of count for n=50 and p=0.12')
plt.xlabel('Number of degrees')
plt.ylabel('Count of each degree')
plt.hist(degree_count3)

#To calculate and plot hitogram for N=100 and p=0.06
problem2(100,0.06)
