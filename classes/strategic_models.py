# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:21:15 2020

@author: 43681
"""
import random as rd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

'''
Klasse der strategichen KI: Diese Klasse wird weitervererbt um einzelne
Unterklassen zu erstellen
'''
class StrategicAI:
    
    ''' Initialisieren des AI Modells '''
    def __init__(self):
        self.input_size = 582
        self.output_size = 181
        self.variante = 'none'
        
    ''' Auswählen eines Spielzugs aus dem state-Vektor '''
    def forward(self, state):
        print('Diese Methode fehlt in der KI-Klasse anscheinend noch...')
        
    ''' Auswählen eines Spielzugs aus dem state-Vektor '''
    def train(self, sample):
        print('Diese Methode fehlt in der KI-Klasse anscheinend noch...')
        
    ''' Darstellung des Modells '''
    def __repr__(self):
        return self.variante
    
    ''' Speichern des Modells '''
    def save(self):
        pass
        
'''
Klasse für eine KI, welche die Entscheidungen auf zufälliger Basis trifft
'''
class RandAI(StrategicAI):
    
    ''' Initialisieren des AI Modells '''
    def __init__(self):
        StrategicAI.__init__(self)
        self.variante = 'random_choice'
        
    ''' Auswählen des Spielzugs auf zufälliger Basis '''
    def forward(self, state):
        legal_moves = state[0:181]
        q_values = np.random.rand(181)
        
        action = legal_moves * q_values
        return action
    
    ''' bei der Lernmethode des Modells passiert nichts '''
    def train(self, sample):
        pass
    
'''
Klasse für eine KI, welche entscheidungen auf Basis eines einfachen neuronalen
Netzwerkes trifft
'''
class DQN(StrategicAI):
    
    ''' Initialisieren des AI Modells '''
    def __init__(self):
        StrategicAI.__init__(self)
        self.gamma = 0.9
        self.learning_counter = 0
        self.target_update_increment = 5
        
    ''' Erzeugen des Netzwerks '''
    def get_network(self):
        try:
            model = self.load()
        except:
            model = self.build_network()
        return model
            
    ''' Build Network-Funktion noch als abstrakte Funktion (Abstrakt)'''
    def build_network(self):
        raise ('Diese Funktion muss noch implementiert werden!')
        
    ''' Zuweisen der Bonuse an die einzelnen Spielzüge (Abstrakt) '''
    def account_reward(self, reward, ritter_bonus, straßen_bonus, sieger_bonus):
        raise ('Diese Funktion muss noch implementiert werden!')
        
    ''' Filtern des inputs, kann noch überschrieben werden '''
    def filter_input(self, states, isList = False):
        if isList:
            filtered_states = [self.filter_input(state) for state in states]
        else:
            filtered_states = np.expand_dims(states, axis = 0)
        return filtered_states
    
    ''' Filtern des Outputs, kann noch überschrieben werden '''
    def filter_output(self, q_values, legal_moves):
        return (q_values - np.min(q_values)) * legal_moves
    
    ''' Vorhersagen des Spielzuges aus dem Zustand '''
    def forward(self, state, policy_net = True):
        # filtern des inputs, errechnen der q-werte und filtern des outputs
        state = self.filter_input(state)
        q_values = self.policy_model.predict(state, batch_size = 1)
        action = self.filter_output(q_values.squeeze(), state[0,0:181])
        return action
    
    ''' trainieren des netzwerks '''
    def train(self, samples):
        # umformen der trainingssamples
        def extract_samples(input_filter, samples):
            tensors = [[],[],[],[]]
            for game_exp in samples:
                for i in range(4):
                    new_tensors = extract_tensors(game_exp[i],game_exp['winner'][i])
                    tensors = [t + nt for t, nt in zip(tensors, new_tensors)]
                    
            b_res = min([len(tensors[3]), 30])
            return (tensors[0][0:b_res], tensors[1][0:b_res], \
                    tensors[2][0:b_res], tensors[3][0:b_res])
        
        # zerlegen einer spielerfahrung in tensoren
        def extract_tensors(game_exp, winner_bonus):
            fields = ['state', 'action', 'next_state']
            tensors_sans = [extract_tensor(game_exp, field) for field in fields]
            
            rew_fields = ['reward', 'ri_bonus', 'str_bonus', 'sp_bonus']
            tensors_rew = [extract_tensor(game_exp, rew_field) for rew_field in rew_fields]
            reward = self.account_reward(tensors_rew[0], tensors_rew[1], tensors_rew[2], tensors_rew[3], winner_bonus)
            
            return (self.filter_input(tensors_sans[0], isList = True), tensors_sans[1], \
                    self.filter_input(tensors_sans[2], isList = True), reward)
        
        def extract_tensor(raw_exp, field_name):
            return [exp._asdict()[field_name] for exp in raw_exp] 
        
        # errechnen des q-target-Wertes
        def get_q_target(gamma, states, actions, next_states, rewards):
            q_val = self.policy_model.predict(np.array(states).squeeze())
            q_next = self.policy_model.predict(np.array(next_states).squeeze())
            q_val[np.array(actions) > 0] = np.array(rewards) + gamma*np.amax(q_next, axis = 1)
            return q_val
        
        # updaten des target-modells
        def update_target_model(self):
            self.learning_counter += 1
            if (self.learning_counter % self.target_update_increment) == 0:
                self.target_model = keras.models.clone_model(self.policy_model)
        
        # trainieren des netzwers mit den ermittelten q-targets
        states, actions, next_states, rewards = extract_samples(self.filter_input, samples)
        q_target = get_q_target(self.gamma, states, actions, next_states, rewards)
        self.policy_model.train_on_batch(np.array(states).squeeze(), q_target)
        update_target_model(self)
        
    ''' Rückgabe des Pfads '''
    def get_path(self):
        return 'ai_models\\' + self.variante + '.h5'
    
    ''' Speichern des Models '''
    def save(self):
        self.policy_model.save(self.get_path())
        
    ''' Laden des Modells '''
    def load(self):
        return load_model(self.get_path())
            
'''
Version 1 des DQN
'''
class DQN_V1(DQN):
    
    ''' initialisieren des Modells '''
    def __init__(self):
        DQN.__init__(self)
        self.variante = 'simple_nn_version1'
        self.policy_model = self.get_network()
        self.target_model = self.get_network()
        
    ''' Erstellen des Netzwerks '''
    def build_network(self):
        model = keras.Sequential([
            keras.layers.Dense(300, activation = 'relu', input_shape = (582,)),
            keras.layers.Dense(300, activation = 'relu'),
            keras.layers.Dense(self.output_size, activation = None)])
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        return model
    
    ''' Wie wird der Reward berechnet '''
    def account_reward(self, raw_reward, ri_bonus, str_bonus, sp_bonus, win_bonus):
        win_bonus = [win_bonus for i in range(len(raw_reward))]
        return [r_rew + r_bon + s_bon + sp_bon + w_bon for r_rew, r_bon, s_bon, sp_bon, w_bon in \
                  zip(raw_reward, ri_bonus, str_bonus, sp_bonus, win_bonus)]
            
'''
Version 2 des DQN
'''
class DQN_V2(DQN):
    
    ''' Initialisieren des Modells '''
    def __init__(self):
        DQN.__init__(self)
        self.variante = 'simple_nn_version2'
        self.policy_model = self.get_network()
        self.target_model = self.get_network()
        
    ''' Erstellen des Netzwerks '''
    def build_network(self):
        model = keras.Sequential([
            keras.layers.Dense(600, activation = 'relu', input_shape = (582,)),
            keras.layers.Dense(500, activation = 'tanh'),
            keras.layers.Dense(300, activation = 'relu'),
            keras.layers.Dense(self.output_size, activation = None)])
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        return model
    
    ''' Wie wird der Reward berechnet '''
    def account_reward(self, raw_reward, ri_bonus, str_bonus, sp_bonus, win_bonus):
        win_bonus = [win_bonus for i in range(len(raw_reward))]
        return [r_rew + r_bon + s_bon + sp_bon + w_bon for r_rew, r_bon, s_bon, sp_bon, w_bon in \
                  zip(raw_reward, ri_bonus, str_bonus, sp_bonus, win_bonus)]
            
