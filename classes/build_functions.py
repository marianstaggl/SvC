# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:13:59 2020

@author: 43681
"""
from classes import base_classes as bc
import numpy as np
import random as rd

'''
Hier sind die Koordinaten der knoten gespeichert. Diese werden in ein dict
eingebunden und dieses wird und alle dicts in einem dict gespeichert
'''
def get_nodes():
    # x-Koordinaten in Form einer Liste
    x_coord = [0.5, 0, -0.5, 0.5, 0, -0.5, -1.5, -1, -1, -1.5, -2.5, -2, -2, 
               -2.5, -1, -0.5, 0, -2, -1.5, -0.5, 0, 0.5, 0.5, -1.5, -1, 0, 
               -0.5, -1, -1.5, -2, 0.5, 0.5, 0, -0.5, -1, -1.5, 1, 1.5, 1.5, 
               1, 2, 2.5, 2.5, 2, 1, 1.5, 2, 1, 1.5, 1, 2, 1.5, 1.5, 1]
    
    # y-Koordinaten in Form einer Liste
    y_coord = [0.2887, 0.5774, 0.2887, -0.2887, -0.5774, -0.2887, 0.2887, 
               0.5774, -0.5774, -0.2887, 0.2887, 0.5774, -0.5774, -0.2887, 
               1.1547, 1.4434, 1.1547, 1.1547, 1.4434, 2.0207, 2.3094, 2.0207, 
               1.4434, 2.0207, 2.3094, -1.1547, -1.4434, -1.1547, -1.4434, 
               -1.1547, -1.4434, -2.0207, -2.3094, -2.0207, -2.3094, -2.0207, 
               0.5774, 0.2887, -0.2887, -0.5774, 0.5774, 0.2887, -0.2887, 
               -0.5774, 1.1547, 1.4434, 1.1547, 2.3094, 2.0207, -1.1547, 
               -1.1547, -1.4434, -2.0207, -2.3094]
    
    # Iteriert über die x-Koordinaten und ordnet ihnen eine y-Koordinate zu
    node_dict = {key: [] for key in range(54)}
    for node, x, y in zip(node_dict.keys(), x_coord, y_coord):
        node_dict[node] = {'coord': np.array([x,y]), 'building': None, 
                        'player': None, 'resourceSet': bc.ResourceSet()}
    return node_dict

'''
Hier sind die Kanten gespeichert, bzw aus welchen knoten sie bestehen
'''        
def get_edges():
    # die kanten als frozensets
    edges = [frozenset([25, 4]), frozenset([40, 46]), frozenset([52, 53]), frozenset([4, 5]), 
             frozenset([8, 27]), frozenset([40, 41]), frozenset([10, 13]), frozenset([14, 7]), 
             frozenset([49, 39]), frozenset([41, 42]), frozenset([42, 43]), frozenset([32, 33]), 
             frozenset([14, 15]), frozenset([10, 11]), frozenset([24, 19]), frozenset([40, 37]), 
             frozenset([11, 6]), frozenset([43, 38]), frozenset([3, 4]), frozenset([18, 14]), 
             frozenset([9, 6]), frozenset([35, 28]), frozenset([50, 43]), frozenset([51, 52]), 
             frozenset([18, 23]), frozenset([0, 3]), frozenset([49, 30]), frozenset([16, 22]), 
             frozenset([16, 15]), frozenset([12, 29]), frozenset([25, 30]), frozenset([27, 28]), 
             frozenset([8, 5]), frozenset([53, 31]), frozenset([9, 12]), frozenset([45, 46]), 
             frozenset([24, 23]), frozenset([0, 36]), frozenset([0, 1]), frozenset([6, 7]), 
             frozenset([36, 37]), frozenset([36, 44]), frozenset([17, 11]), frozenset([17, 18]), 
             frozenset([21, 22]), frozenset([33, 26]), frozenset([44, 22]), frozenset([33, 34]), 
             frozenset([16, 1]), frozenset([20, 21]), frozenset([26, 27]), frozenset([19, 20]), 
             frozenset([2, 7]), frozenset([38, 39]), frozenset([48, 45]), frozenset([44, 45]), 
             frozenset([37, 38]), frozenset([25, 26]), frozenset([32, 31]), frozenset([28, 29]), 
             frozenset([50, 51]), frozenset([1, 2]), frozenset([3, 39]), frozenset([8, 9]), 
             frozenset([19, 15]), frozenset([2, 5]), frozenset([49, 51]), frozenset([48, 47]), 
             frozenset([30, 31]), frozenset([21, 47]), frozenset([12, 13]), frozenset([34, 35])]
    
    # iterieren über die kanten und speichern in dict
    edge_dict = {key: [] for key in edges}
    for i,edge in enumerate(edges):
        edge_dict[edge] = {'ID': i, 'building': None, 'player': None}
    return edge_dict

'''
Hier sind die Koordinaten der knoten gespeichert. Diese werden in ein dict
eingebunden und dieses wird und alle dicts in einem dict gespeichert
'''
def get_fields():
    # x- Koordinaten
    x_coord = [0, -1, -2, -0.5, -1.5, 0, -1, -0.5, -1.5, 0, -1, 1, 2, 0.5, 
               1.5, 1, 0.5, 1.5, 1]
    # y-Koordinaten
    y_coord = [0, 0, 0, 0.8660, 0.8660, 1.7321, 1.7321, -0.8660, -0.8660, 
               -1.7321, -1.7321, 0, 0, 0.8660, 0.8660, 1.7321, -0.8660, 
               -0.8660, -1.7321]
    
    # iterieren über die einzelnen koordinatenpunkte
    field_dict = {key: [] for key in range(19)}
    for field, x, y in zip(field_dict.keys(), x_coord, y_coord):
        field_dict[field] = {'coord': np.array([x,y]), 'number': None, 
                             'resource': None,'resourceSet': bc.ResourceSet()}
    return field_dict

'''
Hier werden die Entwicklungskarten zusammengestellt und als Liste 
zurückgegeben
'''
def get_entw_cards():
    # die vorhandenen entwicklunskarten als dict
    entw = {'ritter': 14,'siegpunkt': 5,'entwicklung': 2,'straßenbau': 2}
    
    # eintragen in eine liste
    entw_cards = []
    for key, value in entw.items():
        for i in range(value):
                entw_cards.append(key)
    rd.shuffle(entw_cards)
    return entw_cards

'''
Hier werden die Resourcen für die Felder zusammengestellt, gemischt und als
Liste zurückgegeben
'''
def get_numbers():
    # rückgabe der nummern
    numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
    rd.shuffle(numbers)
    return [0] + numbers

'''
Hier werden die Nummern für die Felder zusammengestellt, gemischt und als
Liste zurückgegeben
'''
def get_resources():
    # rückgabe der resourcen
    field_resources = ['holz', 'holz', 'holz', 'holz', 'lehm','lehm', 'lehm',
                       'erz','erz','erz', 'wolle', 'wolle', 'wolle', 'wolle', 
                       'getreide', 'getreide', 'getreide', 'getreide']
    rd.shuffle(field_resources)
    return ['wüste'] + field_resources