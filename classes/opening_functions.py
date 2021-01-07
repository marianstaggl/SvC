# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:14:19 2020

@author: 43681
"""
from classes import base_classes as bc
import numpy as np

''' Eröffnungszug des Spielers '''
def init_version_0(matchfield, player):
    new_income = bc.ResourceSet()
    weights = np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]])
    
    best_node = 'none'
    best_nodevalue = 0
    for key in matchfield.free_nodes:
        node_income = matchfield.nodes[key]['resourceSet'].np_array(logical = False)
        node_logical = matchfield.nodes[key]['resourceSet'].np_array(logical = True)
        state_logical = player.avg_income.np_array(logical = True)
        
        value_1 = node_income @ weights[0,:]
        value_2 = np.abs((state_logical - node_logical) * node_logical) @ weights[1,:]
        current_nodevalue = value_1 + value_2
        
        if current_nodevalue > best_nodevalue:
            best_node = key
            best_nodevalue = current_nodevalue
            new_income.update_np(node_income)
    
    turn1 = bc.Turn({'type': 'siedlung', 'position': best_node, 'player': player,
                     'avg_income': new_income})
    turn2 = bc.Turn({'type': 'straße', 'position': matchfield.node2edge[best_node][0],
                     'player': player})
    
    return [turn1, turn2]