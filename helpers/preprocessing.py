#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 22:38:14 2020

@author: ashik
"""

import pandas as pd
from pandas import DataFrame
import networkx as nx
import matplotlib.pyplot as plt
import collections
import json
import rgraph
import numpy as np
from rgraph import Graph, make_graph
import time
from rdflib import Graph as RDFGraph
import os
from numpy import save



'''Convert rdflib.Graph to networkx.Graph'''

def nt_to_networkxGraph(ntGraph):
    graph = nx.Graph()
    for sub, pred, obj in ntGraph:
        graph.add_edge(str(sub), str(obj), predicate=str(pred))

    return graph


'''
Returns networkx.Graph.
Input: Graph data in .nt format.
'''
def load_graph(filepath):

    filepath = 'graph_data/final_graph.nt'
    filename = filepath.split('/')[-1].split('.')[0]
    file = open(filepath)

    graph = nx.Graph()

    for line in file:
        line_split = line.split(' ')
        sub = line_split[0][1:-1]
        pred = line_split[1][1:-1]
        obj = line_split[2][1:-1]
        graph.add_edge(str(sub), str(obj), predicate=str(pred))
    

    print("networkx {} Graph loaded successfully with length {}".format(filename, len(graph)))

    return graph



def nodes(graph):
    nodes_dict = {}
    for index, item in enumerate(graph.nodes(), start=0):
        nodes_dict[item]=index
        print(item)
    
    nodes_json = json.dumps(nodes_dict)
    f = open('nodes.txt', 'w')
    f.write(nodes_json)
    f.close()
    return nodes_dict


# creating relations.txt 
def relations(graph):
    relations_dict = {}
    index=0
    for a, b in graph.edge.items():
        for c,d in b.items():
            if d['predicate'] not in relations_dict:
                relations_dict[d['predicate']] = index
                index = index+1
                
    relations_json = json.dumps(relations_dict)
    f = open('relations.txt', 'w')
    f.write(relations_json)
    f.close()
    return relations_dict


def generate_adj_list_npy(graph):
    
    nodes_json = open('nodes.txt', 'r')
    relations_json = open('relations.txt', 'r')
    node_list = json.load(nodes_json)
    pred_list = json.load(relations_json)
    adj_list = []
    for node1,node2 in graph.edges():
        edge = []
        edge.append(node_list[str(node1)])
        edge.append(node_list[str(node2)])
        edge.append(pred_list[graph[node1][node2]['predicate']])
        print(edge)
        adj_list.append(edge)
    adj_list = np.array(adj_list)
    save('adj_list.npy', adj_list)
    
    
def adj_shape():
    nodes_json = open('nodes.txt', 'r')
    node_list = json.load(nodes_json)
    relations_json = open('relations.txt', 'r')
    pred_list = json.load(relations_json)
    shape = (len(node_list), len(node_list), len(pred_list))
    return shape

def generate_binary():
    shape = adj_shape()
    adj_path =  os.path.join(os.path.abspath(os.curdir), 'adj_list.npy')
    adj_list = np.load(adj_path)
    adj_list = adj_list.astype(np.int32)

    T = Graph(adj_list, shape, sym=True)
    # save graph
    print('Saving graph..')
    t1 = time.time()
    dirpath = os.path.join(os.path.abspath(os.curdir), '_undir')
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print('* Created: %s' % dirpath)
    T.save_graph(dirpath)
    print('Graph saved in {:.4f} secs at: {} '.format(time.time() - t1, dirpath))
    
    
def main():
    filepath = 'graph_data/final_graph.nt'
    print('Loading graph...')
    # graph = nx.read_gpickle('graph_data/com_yago_facts_taxo_types')
    graph = load_graph(filepath=filepath)
    print('Graph load complete.')
    nodes(graph)
    relations(graph)
    generate_adj_list_npy(graph)
    generate_binary()
    
    
if __name__=="__main__":
    main()
