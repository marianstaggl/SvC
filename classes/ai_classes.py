# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 09:52:41 2020

@author: 43681
"""
import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from classes import base_classes as bc
from classes import strategic_models as sm
from classes import opening_functions as of
from matplotlib import colors as mcolors

'''
Klasse der Spielespeichers: Hier werden Umgebungszustand, Spielzug, neuer
Umgebungszustand und die errungenen Punkte gespeichert
'''
class ReplayMemory:
    
    ''' Initialisieren des Speichers '''
    def __init__(self, capacity):
        self.winner_bonus = 10
        self.capacity = capacity
        self.counter = 0
        self.game_experience = []
        self.experience = []
    
    ''' Hinzufügen einer neuen Spieleerinnerung '''
    def push( self, game_exp):
        if self.counter < self.capacity:
            self.game_experience.append(game_exp)
            self.counter += 1
        else:
            replace_idx = rd.randint(0, self.capacity - 1)
            self.game_experience[replace_idx] = game_exp
    
    ''' Rückgabe eines zufälligen Samples'''
    def sample(self, batch_size):
        return rd.sample(self.game_experience, batch_size)
    
    ''' Sind genug Erinnerungen für das Training vorhanden '''
    def can_provide_sample(self, batch_size):
        return len(self.game_experience) >= batch_size
    
'''
Klasse eines Agenten: Hier wird der Agent definiert, der die Umgebung 
"erkundet" und dabei versucht eine optimale Strategie zu finden
'''
class AiAgent(bc.PlayerBase):
    
    ''' Initialisieren des Agenten '''
    def __init__(self, color, matchfield, ai_model, op_function):
        
        # initialisieren des spielers
        bc.PlayerBase.__init__(self, color, matchfield)
        
        # festlegen des ai_models und der bonuse
        self.set_ai_model(ai_model)
        self.set_op_function(op_function)
        self.bonus = {'straßenbau': 0, 'entwicklung': 0}
        
    ''' zurücksetzen des ai-agenten '''
    def reset_agent(self):
        self.reset_base()
        self.bonus = {'straßenbau': 0, 'entwicklung': 0}
        
    ''' Rückgabe des Eröffnungszuges '''
    def opening(self):
        return self.op_function(self.topology.matchfield, self)
        
    ''' Errechnet den nächsten Spielzug'''
    def set_next_action(self, state_vec):            
        # der index wird in eine aktion übersetzt
        policy_values = self.ai_model.forward(state_vec)
        self.next_action.reset(policy_values)
    
    ''' Festlegen des AI-Netzwerks '''
    def set_ai_model(self, model_type):
        # auswahl des zufälligen modells
        if model_type == 'random_choice':
            self.ai_model = sm.RandAI()
            
        # auswahl des ersten nn (kleinere version)
        elif model_type == 'simple_nn_version1':
            self.ai_model = sm.DQN_V1()
            
        # auswahl des zweiten nn (etwas größer vom umfang her)
        elif model_type == 'simple_nn_version2':
            self.ai_model = sm.DQN_V2()
            
        # auswahl des ersten cnn
        elif model_type == 'simple_cnn_version1':
            self.ai_model = sm.DQN_V3()
            
    ''' Festlegen der Funktion für die Eröffnungszüge '''
    def set_op_function(self, op_type):
        # auswahl der eröffnungsvariante
        if op_type == 'init_version_0':
            self.op_function = of.init_version_0
            
        else:
            raise
            
'''
Klasse der Spielumgebung: Hier wird die Spielumgebung für das training der 
strategischen modelle definiert.
'''
class Environment():
    
    ''' Initialisieren der Klasse '''
    def __init__(self, ai_models, op_functions):
        
        # speichern der farben in einem dict
        self.color_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        self.colors = ['red','gold','green','blue'] # kennzeichnung der agenten
        self.color_code = {'red': 0.25, 'gold': 0.5, 'green': 0.75, 'blue': 1} # notwendig für den state-vektor
        
        # initialisieren des spielfelds und der spieler
        self.agents = {}
        self.matchfield = bc.Matchfield(None, self.color_code)
        for c, ai, op_f in zip(self.colors, ai_models, op_functions):
            self.agents[c] = AiAgent(c, self.matchfield, ai, op_f)
        self.matchfield.topologies = [a.topology for a in self.agents.values()]
        
        # spielpunktwertungen initialisieren
        self.handelsstraße = {a: a.handelsstraße for a in self.agents.values()}
        self.längste_handelsstraße = [None, 4]
        self.größte_rittermacht = [None, 2]
        self.max_victorypoints = 0
        
    ''' Zurücksetzen der Spielumgebung'''
    def reset(self):
        # zurücksetzen des spielfelds und der spieler
        self.matchfield.reset_matchfield()
        [a.reset_agent() for a in self.agents.values()]
        
        # zurücksetzen der spielpunktbewertungen
        self.längste_handelsstraße = [None, 4]
        self.größte_rittermacht = [None, 2]
        self.max_victorypoints = 0
            
    ''' Hier wird das Spiel initialisiert und die Spieler wählen die Startpositionen'''
    def setup(self):
        # alle spieler setzten die erste siedlung
        for color in (self.colors):
            turns = self.agents[color].opening()
            self.agents[color].ex_turns(turns)
            self.matchfield.ex_turns(turns)
            
            # alle spieler setzen die zweite siedlung
        for color in reversed(self.colors):
            turns = self.agents[color].opening()
            self.agents[color].ex_turns(turns)
            self.matchfield.ex_turns(turns)
            
    ''' Bestimmen der nächsten Spielaktionen der Agenten '''
    def get_next_actions(self):
        
        # festlegen der next_action eines jeden spielers
        colors =[]
        needed_time = []
        for agent in self.agents.values():
            
            # bestimmen der nächsten spielaktion
            env_state = self.state(agent.color)
            agent.set_next_action(env_state)
            
            # erweitern der liste
            needed_time.append(agent.next_action.needed_time)
            colors.append(agent.color)
            
        # rückgabe dieser werte
        return self.colors, needed_time
        
    ''' Rückgabe des Spielfeldzustandes als Vektor (aus Spielerperspektive) '''
    def state(self, color):
        # die ersten einträge des vektors geben aufschluss über die erlaubten spielzüge
        s1 = self.agents[color].avai_straße()
        s2 = self.agents[color].avai_siedlung()
        s3 = self.agents[color].avai_stadt()
        s4 = np.array([not not self.matchfield.entw_cards])
        
        # die weiteren einträge über den spielfeldzustand
        s5 = self.matchfield.state(color)
        
        # umwandeln in numpy-array und rückgabe
        return np.concatenate((s1, s2, s3, s4, s5))
        
    ''' Durchführen einer Spielaktion '''
    def step(self, color):
        
        # shortcuts
        turn_list = self.agents[color].next_action.turn_list
        bonus_left = self.agents[color].next_action.bonus_left
        
        # durchführen der spielzüge
        self.matchfield.ex_turns(turn_list)
        self.agents[color].ex_turns(turn_list)
        self.agents[color].entw_bonus.update(bonus_left)
        
        # welchen bonus bekommt der spieler?
        reward = self.agents[color].next_action.victorypoints
        card_no = self.agents[color].next_action.cost.sum_up()
        ri_bonus, ri_num = self.update_rittermacht(self.agents[color])
        str_bonus, str_num = self.update_handelsstraße(self.agents[color])
        sp_bonus = turn_list[-1]['bonus_type'] == 'siegpunkt'
        
        # updaten der maximalen siegpunkte
        self.update_victorypoints()
        
        # rückgabe der belohnung
        return reward/card_no, ri_bonus/(3*ri_num), str_bonus/(2*str_num), sp_bonus/(3)
    
    ''' Updaten der Siegespunkte der einzelnen Spieler '''
    def update_victorypoints(self):
        
        # scheife über alle spieler
        for agent in self.agents.values():
            current_vp = agent.victorypoints + agent.entw_bonus['siegpunkt']
            
            # berücksichtigeung der größten rittermacht
            if agent == self.größte_rittermacht[0]:
                current_vp += 2
                
            # berücksichtigen der längsten handelsstraße
            if agent == self.längste_handelsstraße[0]:
                current_vp += 2
            
            # updaten der siegpunkte
            self.max_victorypoints = max(self.max_victorypoints, current_vp)
            
    ''' Trainieren der AI-Modelle '''
    def train_agents(self, game_exp):
        
        # trainieren der einzelnen agenten
        for agent in self.agents.values():
            agent.ai_model.train(game_exp)
    
    ''' Speichern der ai-Modelle '''
    def save_models(self):
        [agent.ai_model.save() for agent in self.agents.values()]
    
    ''' Updaten der Handelsstraße '''
    def update_handelsstraße(self, player):
        
        # aktualisieren bei den spielern
        [a.update_handelsstraße() for a in self.agents.values()]
        
        # nachsehen wer die längste straße hat
        for player, handelsstraße in self.handelsstraße.items():
            
            if handelsstraße[0] > self.längste_handelsstraße[1]:
                self.längste_handelsstraße[0] = player
                self.längste_handelsstraße[1] = handelsstraße[0]
                
        # falls die längste handelsstraße kürzer ist als 5
        if 5 > self.längste_handelsstraße[1]:
            self.längste_handelsstraße[0] = None
            self.längste_handelsstraße[1] = 4
            
        # ausgabe des bonus
        if self.längste_handelsstraße[0] == player:
            return (2, player.handelsstraße[0])
        else:
            return (0, 1)
    
    ''' Updaten der Rittermacht '''
    def update_rittermacht(self, player):
        
        # aktualisieren beim spieler
        player.update_rittermacht()
        
        # updaten nach dem spielzug
        if  player.rittermacht > self.größte_rittermacht[1]:
            self.größte_rittermacht[0] = player
            self.größte_rittermacht[1] = player.rittermacht
            
        # ausgabe des bonus
        if self.größte_rittermacht[0] == player:
            return (2, player.rittermacht)
        else:
            return (0, 1)
    
    ''' Plotten des Spielfeldes '''
    def plot(self, new_fig = True):
        if new_fig:
            plt.figure(figsize = (8,8))
        plt.cla()
        plt.axis('equal')
        
        # plotten der kanten
        for ID, edge in self.matchfield.edges.items():
            self.plot_edge(edge, ID)
            
        # plotten der knoten
        for ID, node in self.matchfield.nodes.items():
            self.plot_node(node, ID)
            
        # plotten der felder
        for ID, field in self.matchfield.fields.items():
            self.plot_field(field, ID)
            
    ''' Plotten eines Knoten '''
    def plot_node(self,node, ID):
        def plot_symbol(x, y, color, bg_style, fr_style, formation):
            # plotten des Hintergrunds
            #plt.scatter(x, y, marker = bg_style, c = 'white', s = 400)
            plt.scatter(x, y, marker = bg_style, edgecolors = color, facecolors = 'white', s = 400) 
            
            # plotten der formen
            shift = 0.05
            if formation == 's':
                x_vec = np.array([x + shift, x + shift, x - shift, x - shift])
                y_vec = np.array([y + shift, y - shift, y - shift, y + shift])
                ms = 20
            elif formation == '^':
                rad = math.pi/3
                x_vec = np.array([x, x - shift*math.sin(rad), x + shift*math.sin(rad)])
                y_vec = np.array([y + shift, y - shift*math.cos(rad), y - shift*math.cos(rad)])
                ms = 20
            plt.scatter(x_vec, y_vec, marker = fr_style, c = color, s = ms)
            
        if node['building'] == 'siedlung':
            color = self.color_dict[node['player'].color]
            plot_symbol(node['coord'][0], node['coord'][1], color, 'o', '^', 's')
        elif node['building'] == 'stadt':
            color = self.color_dict[node['player'].color]
            plot_symbol(node['coord'][0], node['coord'][1], color, 's', 's', 's')
        else:
            pass
        plt.text(node['coord'][0], node['coord'][1], str(ID), 
                 horizontalalignment = 'center', verticalalignment = 'center')
        
    ''' Plotten eines Feldes '''
    def plot_field(self, field, ID):
        def plot_symbol(x, y, bg_color, bg_style , fr_color, fr_style, fr_formation):
            # plotten des Hintergrunds
            plt.scatter(x, y, marker = bg_style, c = bg_color, s = 2000)
            
            # plotten der formen
            shift = 0.15
            if fr_formation == '^':
                rad = math.pi/3
                x_vec = np.array([x, x - shift*math.sin(rad), x + shift*math.sin(rad)])
                y_vec = np.array([y + shift, y - shift*math.cos(rad), y - shift*math.cos(rad)])
                ms = 70
            elif fr_formation == 's': 
                x_vec = np.array([x + shift, x + shift, x - shift, x - shift])
                y_vec = np.array([y + shift, y - shift, y - shift, y + shift])
                ms = 50
            plt.scatter(x_vec, y_vec, marker = fr_style, c = fr_color, s = ms)
            
        # plotten eines wolle feldes
        if field['resource'] == 'wolle':
            plot_symbol(field['coord'][0], field['coord'][1], 'yellowgreen', 
                        's', 'white', 'o', 's')
            
        elif field['resource'] == 'erz':
            plot_symbol(field['coord'][0], field['coord'][1], 'lightgray', 
                        'o', 'gray', '^', '^')
            
        elif field['resource'] == 'lehm':
            plot_symbol(field['coord'][0], field['coord'][1], 'peachpuff', 
                        'o', 'coral', 's', '^')
            
        elif field['resource'] == 'holz':
            plot_symbol(field['coord'][0], field['coord'][1], 'yellowgreen', 
                        's', 'forestgreen', '^', '^')
            
        elif field['resource'] == 'getreide':
            plot_symbol(field['coord'][0], field['coord'][1], 'sandybrown', 
                        's', 'gold', 'd', 's')
            
        #plt.scatter(field['x'], field['y'], marker = 'o', c = 'white', s = 150)
        plt.text(field['coord'][0], field['coord'][1], str(field['number']), weight = 'bold',
                 horizontalalignment = 'center', verticalalignment = 'center')
        
    ''' Plotten einer Kante '''
    def plot_edge(self, edge,ID):
        nodes = list(ID)
        coord = np.array([self.matchfield.nodes[nodes[0]]['coord'], 
                          self.matchfield.nodes[nodes[1]]['coord']])
        if edge['building'] == None:
            color = 'k'
            linewidth = 1
        else:
            color = self.color_dict[edge['player'].color]
            linewidth = 5
        
        plt.plot(coord[:,0], coord[:,1], c = color, linewidth = linewidth, zorder = -1)