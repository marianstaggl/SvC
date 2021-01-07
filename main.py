# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:50:27 2020

@author: 43681
"""

import numpy as np
import collections as col
from classes import ai_classes as ai
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import time

# aufbauen des spiels
figure = plt.figure(figsize = (8, 8))
victorys = np.zeros(4)

# aufbauen des spiels
ai_models = ['simple_nn_version2', 'simple_nn_version2', 'simple_nn_version2', 'simple_nn_version2']
op_functions = ['init_version_0', 'init_version_0', 'init_version_0', 'init_version_0']
env = ai.Environment(ai_models, op_functions)
replay_memory = ai.ReplayMemory(1024)
exp = col.namedtuple('exp','player, state, action, next_state, reward, ri_bonus, \
                     str_bonus, sp_bonus')

# festlegen von einzelnen konstanten
batch_size = 1
num_games = 5

# spielen der spiele und eventuelles training der ai
for i in range(num_games):
    
    # zurücksetzen des spiels
    env.reset()
    env.setup()

    # initialisieren der spielerfahrung usw
    used_time = np.array([0, 0.1, 0.2, 0.3])
    game_exp = {'winner': np.zeros(4), 0: [], 1: [], 2: [], 3: []}
    
    while env.max_victorypoints < 10:
        
        # plotten des spielfelds
        env.plot(new_fig = False)
        plt.pause(0.5)
        
        # nächste züge der agenten
        color, needed_time = env.get_next_actions()
        
        # welcher spieler erreicht sein ziel zuerst
        temp_time = used_time + needed_time
        p_idx = temp_time.argmin()
        p_color = color[p_idx]
        
        # dieser spieler darf seine aktion ausführen
        old_state = env.state(p_color)
        action = env.agents[p_color].next_action.vector_form()
        reward, ri_bonus, str_bonus, sp_bonus = env.step(p_color)
        next_state = env.state(p_color)
        
        # speichern der spielerinnerung
        new_exp = exp(p_idx, old_state, action, next_state, reward, 
                      ri_bonus, str_bonus, sp_bonus)
        game_exp[p_idx].append(new_exp)
        used_time[p_idx] = temp_time[p_idx] 
        
    # vermerken wer gewonnen hat und abspeichern der spielerinnerung
    game_exp['winner'][p_idx] = 100/used_time[p_idx]
    replay_memory.push(game_exp)
    
    # vermerken wer wie oft gewonnen hat
    victorys[p_idx] += 1
    print(victorys)
    
    # falls genug spiele gespielt wurden werden die netzwerke trainiert
    if replay_memory.can_provide_sample(batch_size):
        game_exp = replay_memory.sample(batch_size)
        env.train_agents(game_exp)
        
# speichern der ai-Modelle
env.save_models()