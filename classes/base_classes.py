# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:42:37 2020

@author: 43681
"""

import numpy as np
import networkx as nx
from classes import build_functions as bf

'''
Klasse der Spielfeld-Basis: Diese Klasse enthält die grundsätzlichen
Informationen zum Spielfeld, wie etwa die Position der Knoten, aus welchen
Knoten die Kanten bestehen usw. Diese werden beim mischen des Spielfeldes auch
nicht mehr geändert
'''
class MatchfieldBase:
    
    ''' Initialisieren der Klasse '''
    def __init__(self):
        # Laden der Koordinaten und Verknüpfungen
        self.nodes = bf.get_nodes() # die keys sind die ID-Zahlen von 1 - 54
        self.edges = bf.get_edges() # die keys sind frozen sets mit den knoten
        self.fields = bf.get_fields() # die keys sind die ID-Zahlen von 1 - 19
        
        # verlinken für einen späteren zugriff
        self.node2edge, self.node2edgeID = self.get_node2edge_ID()
        self.node2field = self.get_node2field()
        self.edge2node, self.edgeID2node = self.get_edge_ID2node()
        self.edge2field = self.get_edge2field()
        self.field2node = self.get_field2node()
        self.field2edge = self.get_field2edge()
        self.edge2ID, self.ID2edge = self.get_edge2ID()
        
        # zusätzliche darstellung als graph
        self.base_graph = self.get_base_graph()
        
    ''' Funktion zum zurücksetzen des matchfield-base '''
    @staticmethod
    def reset_node(node):
        # zurücksetzten des knoten
        node['building'] = None
        node['player'] = None
        node['resourceSet'].drain()
    
    @staticmethod    
    def reset_edge(edge):
        # zurücksetzen einer kante
        edge['building'] = None
        edge['player'] = None
        
    @staticmethod
    def reset_field(field):
        # zurücksetzen des feldes
        field['resource'] = None
        field['number'] = None
        field['resourceSet'].drain()
            
    def reset_base(self):
        # zurücksetzen aller knoten
        for node in self.nodes.values():
            self.reset_node(node)
            
        # zurücksetzen aller kanten
        for edge in self.edges.values():
            self.reset_edge(edge)
            
        # zurücksetzen aller felder
        for field in self.fields.values():
            self.reset_field(field)
        
    ''' Funktion um das node2edge dict zu erzeugen '''
    def get_node2edge_ID(self):
        # initialisieren des dict
        node2edge = {key: [] for key in range(54)}
        node2edgeID = {key: [] for key in range(54)}
        
        # befüllen des dict
        for node in self.nodes.keys():
            for edge, edge_data in self.edges.items():
                if node in edge:
                    node2edge[node].append(edge)
                    node2edgeID[node].append(edge_data['ID'])
        return node2edge, node2edgeID
            
    ''' Funktion um das node2field dict zu erzeugen '''
    def get_node2field(self):
        # initialisieren des dict
        node2field = {key: [] for key in range(54)}
        
        # befüllend es dict
        for node, node_data in self.nodes.items():
            for field, field_data in self.fields.items():
                n_coord = node_data['coord']
                f_coord = field_data['coord']
                dist = np.sqrt(np.sum((n_coord - f_coord)**2))
                if dist < 0.6:
                    node2field[node].append(field)
        return node2field
        
    ''' Funktion um das edge2node dict zu erzeugen '''
    def get_edge_ID2node(self):
        # initialisieren des dict
        edge2node = {key: [] for key in self.edges.keys()}
        edgeID2node = {key: [] for key in self.edges.keys()}
        
        # befüllen des dict
        for edge, edge_data in self.edges.items():
            edge2node[edge] = list(edge)
            edgeID2node[edge_data['ID']] = list(edge)
        return edge2node, edgeID2node
    
    ''' Funktion um das edge2field dict zu erzeugen (vorher node2field und edge2node) '''
    def get_edge2field(self):
        # initialisieren des dict
        edge2field = {key: [] for key in self.edges.keys()}
        
        # befüllen des dict
        for edge in self.edges.keys():
            nodes = self.edge2node[edge]
            f_n1 = set(self.node2field[nodes[0]])
            f_n2 = set(self.node2field[nodes[1]])
            edge2field[edge] += list(f_n1 & f_n2)
        return edge2field
    
    ''' Funktion um das field2node dict zu erzeugen (vorher node2field)'''
    def get_field2node(self):
        # initialisieren des dict
        field2node = {key: [] for key in range(19)}
        
        # befüllen des dict
        for node in self.nodes.keys():
            for field in self.node2field[node]:
                field2node[field].append(node)
        return field2node
            
    ''' Funktion um das field2edge dict zu erzeugen (vorher edge2field)'''
    def get_field2edge(self):
        # initialisieren des dict
        field2edge = {key: [] for key in range(19)}
        
        # befüllen des dict
        for edge in self.edges.keys():
            for field in self.edge2field[edge]:
                field2edge[field].append(edge)
        return field2edge
    
    ''' Funktion um das edge2ID dict zu erzeugen '''
    def get_edge2ID(self):
        # initialisie der dict
        edge2ID = {key: None for key in self.edges.keys()}
        ID2edge = {key: None for key in range(72)}
        
        # befüllen des dict
        for edge, edge_data in self.edges.items():
            edge2ID[edge] = edge_data['ID']
            ID2edge[edge_data['ID']] = edge
        return edge2ID, ID2edge
    
    ''' Funktion zum erstellen des Grapen des das Feld repräsentiert (vorher bf.get_edges und edge2node)'''
    def get_base_graph(self):
        # initialisieren des graphen
        graph = nx.Graph()
        
        # verlinken der knoten
        for edge in self.edges.keys():
            nodes = self.edge2node[edge]
            graph.add_edge(nodes[0], nodes[1])
        return graph

'''
Klasse eines Resourcensets: Abgeleitet von einem Dictionary, wird es auf die
oben angeführten Felder beschränkt und erweitert die Funktionalität um 
Addition, Subtraktion, Multiplikation und Division
'''
class ResourceSet(dict):
    __resource = ['lehm','holz','getreide','wolle','erz','wüste']
    
    ''' Initialisieren des Dict mit den Feldern lehm, holz,... '''
    def __init__(self, *argv):
        for i,res in enumerate(self.__resource):
            try:
                self[res] = argv[0][res]
            except:
                self[res] = 0   
                
    ''' Ausbalancieren der Resourcen, falls eine Resource im Minus ist '''
    def balance(self, bonus = 0, exchange_rate = 4):
        flag = True
        resources = self.np_array()
        while True:
            # überprüfen ob die resourcen alle im plus sind
            if all(resources>=0):
                break
            elif all(resources<0):
                flag = False
                resources = self.np_array()
                break
            
            # tauschen der resourcen oder einlösen der entwicklung
            if bonus > 0:
                resources[resources.argmin()] += 1
                bonus -= 1
            else :
                resources[resources.argmin()] += 1
                resources[resources.argmax()] -= exchange_rate
            
        # falls noch ein bonus übrig ist
        if bonus % 2 > 0:
            resources[resources.argmin()] += 1
            bonus -= 1
            
        self.update_np(resources)
        return (flag, bonus)
    
    ''' Updaten der Werte mithilfe eines np-Arrays '''
    def update_np(self, array):
        for i,res in enumerate(self.__resource):
            self[res] = array[i]
    
    ''' Umwandeln des Dictionary in ein np-Array '''
    def np_array(self, logical = False):
        np_array = np.zeros(len(self.__resource))
        for i,res in enumerate(self.__resource):
            if logical:
                np_array[i] = self[res]>0
            else:
                np_array[i] = self[res]
        return np_array
            
    ''' Erweitern der Funktionalität des Dictionary um die Addition '''
    def __add__(self, resource_set):
        new_instance = self.copy()
        for res in self.__resource:
            new_instance[res] += resource_set[res]
        return new_instance
            
    ''' Erweitern der Funktionalität des Dictionary um die Subtraktion '''
    def __sub__(self, resource_set):
        new_instance = self.copy()
        for res in self.__resource:
            new_instance[res] -= resource_set[res]
        return new_instance
            
    ''' Erweitern der Funktionalität des Dictionary um die Multiplikation '''
    def __mul__(self, constant):
        new_instance = self.copy()
        for res in self.__resource:
            new_instance[res] = self[res] * constant
        return new_instance
            
    ''' Erweitern der Funktionalität des Dictionary um die Division '''
    def __truediv__(self, constant):
        new_instance = self.copy()
        for res in self.__resource:
            new_instance[res] = self[res] / constant
        return new_instance
    
    ''' Aufsummieren aller resourcen '''
    def sum_up(self):
        sum_up = 0
        for res in self.__resource:
            sum_up += self[res]
        return sum_up
    
    ''' Überschreiben der copy() Methode '''
    def copy(self):
        return ResourceSet(self)
    
    ''' Ausleeren des dict '''
    def drain(self):
        for res in self.__resource:
            self[res] = 0
    
'''
Klasse eines Spielzuges: Abgeleitet von einem Dictionary, wird es auf die
angeführten Felder beschränkt. Die Funktionalität wird erweitert umd die
Baukosten.
'''
class Turn(dict):
    __turn = ['type', 'bonus_type', 'position', 'player', 'avg_income']
    
    ''' Initialisiert das Dict mit den Feldern typ, position,... '''
    def __init__(self, *argv):
        for att in self.__turn:
            try:
                self[att] = argv[0][att]
            except:
                self[att] = None
        if not self['avg_income']:
            self['avg_income'] = ResourceSet({})
    
    ''' gibt auskunft über die notwendigen Resourcen'''
    def cost(self):
        cost_dict = {'siedlung':    {'getreide':1, 'wolle':1, 'holz':1, 'lehm':1},
                     'stadt':       {'erz': 3, 'getreide': 2},
                     'straße':      {'lehm': 1, 'holz': 1},
                     'entwicklung': {'erz': 1, 'getreide': 1, 'wolle': 1}}
        return ResourceSet(cost_dict[self['type']])

'''
Klasse des Spielfeldes: Diese Klasse enthält neben den Positionen und
verlinkungen der basis die resourcen und die würfelzahlen. Daraus lässt sich 
auch das durchschnittliche Einkommen eines Konten errechnen
'''
class Matchfield(MatchfieldBase):
    
    ''' Initialisieren der Klasse '''
    def __init__(self, topologies, color_code):
        # basis für das spielfeld festlegen
        MatchfieldBase.__init__(self)
        self.field2number = {key: [] for key in range(19)}
        self.number2field = {key: [] for key in range(13)}
        self.entw_cards = []
        
        # topologie des feldes je nach spielerperspektive
        self.topologies = topologies
        self.free_nodes = set(range(54))
        self.free_edges = set(range(72))
        
        # vektorform des spielfeldes mit allen informationen
        self.color_code = color_code
        self.state_flag = False
        self.state_vector = np.zeros(401)
        
    ''' Initialisieren des Spielfeldes '''
    def reset_matchfield(self):
        # zurücksetzen der basis
        self.reset_base()
        
        # zurücksetzen der freien knoten und entwicklungskarten
        self.free_nodes.update(range(54))
        self.free_edges.update(range(72))
        self.entw_cards = bf.get_entw_cards()
        
        # verteilen der resourcen und würfelzahlens
        self.set_resources(bf.get_resources())
        self.set_numbers(bf.get_numbers())
        self.update_avg_income()
        
    ''' Funktion zum Setzen der Resourcen '''
    def set_resources(self, resource_list):
        # schleife über die felder und die resourcen (wüste immer in die mitte)
        for field, resource in zip(self.fields.values(), resource_list):
            field['resource'] = resource
            field['resourceSet'][resource] = 1
            
    ''' Funktion zum Setzen der Zahlen (zuerst set_resources)'''
    def set_numbers(self, number_list):
        # relative häufigkeit nach nummer
        rel_h = {2 : 1/36, 3 : 2/36, 4 : 3/36, 5 : 4/36, 6 : 5/36, 7 : 6/36,
                 8 : 5/36, 9 : 4/36, 10 : 3/36, 11 : 2/36, 12 : 1/36, 0 : 0}
        
        # schleife über die felder und die resourcen (wüste immer in die mitte)
        for field, number in zip(self.fields.items(), number_list):
            
            # festlegen der nummer und des durchschnittlichen einkommens des feldes
            field[1]['number'] = number
            field[1]['resourceSet'] *= rel_h[number]
            
            # aktualisieren der field2number und number2field dicts
            self.field2number[field[0]].append(number)
            self.number2field[number].append(field)
            
    ''' Funktion zum updaten des durchschnittlichen einkommens '''
    def update_avg_income(self):
        # schleife über die knoten
        for node, node_data in self.nodes.items():
            
            # schleife über die felder eines knoten
            for field in self.node2field[node]:
                node_data['resourceSet'] += self.fields[field]['resourceSet']
                
    ''' Funktion gibt den zustand des Spielfeldes in vektorform zurück '''
    def state(self, color):
        # falls das ganze noch nie abgerufen wurde
        if not self.state_flag:
            
            # schleife über alle knoten
            temp = np.array([node['resourceSet'].np_array() for node in self.nodes.values()])
            self.state_vector[0:270] = (np.reshape(temp[:,:-1].transpose(), (1,-1))).squeeze()
            self.state_vector[-1] = self.color_code[color]
            self.state_flag = True
            
        return self.state_vector
            
    ''' Verteilen der Resourcen nach einem würfelwurf (zuerst number2field!)'''
    def deal_resources(self, number):
        # schleife über die felder aus dem number2field dict
        for field in self.number2field[number]:
            
            # schleife über die knoten der felder
            for node in self.field2node[field]:
                
                # siedlung
                if self.nodes[node]['building'] == 'siedlung':
                    self.nodes[node]['player'].resources[field['resources']] += 1
                    
                # oder stadt
                elif self.nodes[node]['building'] == 'stadt':
                    self.nodes[node]['player'].resources[field['resources']] += 2
                    
    ''' Ausführen eines spielzuges '''            
    def build(self, location, building, player):
        # bauen eines gebäudes
        if building == 'straße':
            self.edges[location]['building'] = building
            self.edges[location]['player'] = player
            self.state_vector[self.edge2ID[location] + 270] = self.color_code[player.color] # eintragen in den state-vektor
        else:
            self.nodes[location]['building'] = building
            self.nodes[location]['player'] = player
            self.state_vector[location + 342] = self.color_code[player.color] # eintragen in den state-vektor
        
    def block(self, node):
        # bebauen eines knoten mit einer siedlung und entfernen der nachbarknoten
        for n in self.base_graph.neighbors(node):
            self.free_nodes.discard(n)
        self.free_nodes.remove(node)
        
    def ex_turn(self, turn):
        # falls eine entwicklung gekauft wird
        if turn['type'] == 'entwicklung':
            turn['bonus_type'] = self.entw_cards.pop(0)
            return
        
        # falls eine siedlung gebaut wird
        if turn['type'] == 'siedlung':
            self.block(turn['position'])
        
        # oder eine straße
        if turn['type'] == 'straße':
            self.free_edges.remove(self.edge2ID[turn['position']])
            
        self.build(turn['position'], turn['type'], turn['player'])
        [t.update(turn['type'], turn['position'], turn['player']) for t in self.topologies]
            
    def ex_turns(self, turns):
        # ausführen mehrerer spielzüge
        for turn in turns:
            self.ex_turn(turn)
            
'''
Klasse einer topologischen Perspektive eines Spielers: Diese Klasse enthält
die Informationen zu erreichbaren Knoten und der länge der längsten 
Handelsstraße eines jeden Mitspielers. Diese Objekte sind im Spielfeld
hinterlegt
'''
class Topology():
    
    ''' Initialisieren der Klasse '''
    def __init__(self, player, matchfield):
        # jede perspektive gehört einem spieler
        self.player = player
        self.matchfield = matchfield
        self.free_nodes = matchfield.free_nodes
        self.free_edges = matchfield.free_edges
        
        # graphen zur beschreibung der topologie
        self.own_graph = OwnGraph()
        self.reach_graph = ReachGraph(self.matchfield, self.matchfield.base_graph)
        
    ''' initialize funktion '''
    def reset(self):
        # graphen zur beschreibung der topologie
        self.reach_graph.reset(self.matchfield.base_graph) # alle freien knoten und kanten
        self.own_graph.reset() # alle eigenen knoten und kanten
        
    ''' updaten der topologie auf basis durchgeführter spielzüge '''
    def update(self, turn_type, turn_pos, player):
        
        # falls ein anderer spieler eine siedlung baut
        if turn_type == 'siedlung' and player != self.player:
            self.reach_graph.remove_node(turn_pos) # notwendig für erreichbare knoten
            self.own_graph.remove_node(turn_pos)
        
        # falls der spieler eine straße baut
        if turn_type == 'straße' and player == self.player:
            self.reach_graph.add_edge(turn_pos)
            self.own_graph.add_edge(turn_pos)
            
        # falls ein anderer spieler eine straße baut
        if turn_type == 'straße' and player != self.player:
            self.reach_graph.remove_edge(turn_pos)
        
    ''' rückgabe der verfügbaren knoten für eine siedlung '''
    def avai_siedlung(self):
        return list(self.reach_graph.get_nodes() & self.free_nodes)
    
    ''' rückgabe der verfügbaren kanten für eine straße '''
    def avai_straße(self):
        return list(self.reach_graph.get_edges() & self.free_edges)
    
    ''' rücgabe der länge der längsten handelsstraße '''
    def handelsstraße(self):
        return self.own_graph.longest_path()
            
'''
Basisklasse eines Spielers: Hier werden die grundlegenden Eigenschaften eines
Spielers festgelegt
'''
class PlayerBase():
    
    ''' Initialisieren der Klasse '''
    def __init__(self, color, matchfield):
        # festlegen der farbe des spielers
        self.color = color
        self.topology = Topology(self, matchfield)
        self.next_action = Action(self, matchfield)
        self.resources = ResourceSet()
        self.avg_income = ResourceSet()
        
        # festlegen der verschiedenen variablen
        self.rittermacht, self.handelsstraße, self.victorypoints = [0], [0], 0
        
    ''' Zurücksetzen des spielers '''
    def reset_base(self):
        # festlegen verfügbarer resourcen und gebäude sowie entwicklungen
        self.avai_buildings = {'siedlung': 5, 'stadt': 4, 'straße': 15}
        self.set_buildings = {'siedlung': [], 'stadt': [], 'straße': []}
        
        # zurücksetzen der resourcen und der topologie
        self.resources.drain(), self.avg_income.drain(), self.topology.reset()
        
        # längste handelsstraße, entwicklungen usw.
        self.entw_bonus = {'ritter': 0, 'straßenbau': 0, 'entwicklung': 0, 'siegpunkt': 0}
        self.rittermacht, self.handelsstraße[0], self.victorypoints = 0, 0, 0
        
    ''' Ausführen eines Spielzuges '''
    def ex_turn(self, turn):
        # kosten und zuverdienst
        self.resources -= turn.cost()
        self.avg_income += turn['avg_income']
        
        # falls eine entwicklung gekauft wird
        if turn['type'] == 'entwicklung':
            self.entw_bonus[turn['bonus_type']] += 1
            return
        
        # falls ein gebäude gebaut wird
        self.avai_buildings[turn['type']] -= 1
        self.set_buildings[turn['type']].append(turn['position'])
        if turn['type'] == 'stadt':
            self.avai_buildings['siedlung'] += 1
            self.set_buildings['siedlung'].remove(turn['position'])
            
        # raufzählen der siegpunkte
        if turn['type'] == 'siedlung' or turn['type'] == 'stadt':
            self.victorypoints += 1
            
    def ex_turns(self, turns):
        for turn in turns:
            self.ex_turn(turn)
    
    ''' Bestimmen der Ritteranzahl '''
    def update_rittermacht(self):
        self.rittermacht = self.entw_bonus['ritter']
    
    ''' Rückgabe der längsten Handelsstraße '''
    def update_handelsstraße(self):
        self.handelsstraße[0] = self.topology.handelsstraße()
    
    ''' Rückgabe der bebaubaren und erreichbaren knoten '''
    def avai_straße(self):
        avai_straße = np.zeros(72)
        avai_straße[self.topology.avai_straße()] = 1
        return avai_straße
    
    ''' Rückgabe der bebaubaren und erreichbaren knoten '''
    def avai_siedlung(self):
        avai_siedlung = np.zeros(54)
        avai_siedlung[self.topology.avai_siedlung()] = 1
        return avai_siedlung
    
    ''' Rückgabe der bebaubaren und erreichbaren knoten '''
    def avai_stadt(self):
        avai_stadt = np.zeros(54)
        avai_stadt[self.set_buildings['siedlung']] = 1
        return avai_stadt
    
''' 
Unterklasse der Topologie: Diese Klasse enthält alle notwendigen
Informationen um die längste Handelsstraße zu bestimmen
'''
class OwnGraph():
    
    ''' Initialisieren der Klasse '''
    def __init__(self):
        self.graph = nx.Graph()
        self.blocked_nodes = []
        self.path_len = 0
        self.up2date = False
        
    ''' Zurücksetzen der Klasse '''
    def reset(self):
        self.graph.clear()
        self.blocked_nodes.clear()
        self.path_len = 0
        self.up2data = False
        
    ''' Funktion zum hinzufügen einer Kante '''
    def add_edge(self, edge):
        # hinzufügen der knoten
        node_1, node_2 = edge
        self.graph.add_node(edge, weight = 1)
        if not node_1 in self.blocked_nodes:
            self.graph.add_node(node_1, weight = 0)
            self.graph.add_edge(node_1, edge)
            
        if not node_2 in self.blocked_nodes:
            self.graph.add_node(node_2, weight = 0)
            self.graph.add_edge(node_2, edge)
    
        # updaten des graphen
        self.up2date = False
        
    ''' Funktion zum entfernen eines Knoten '''
    def remove_node(self, node):
        # nachsehen ob etwas entfernt werden muss
        flag = self.graph.has_node(node)
        self.blocked_nodes.append(node)
        
        # entfernen eines blockierten knoten
        if flag:
            self.graph.remove_node(node)
            self.up2date = False
    
    ''' Funktion zum bestimmen des längsten Pfads '''
    def longest_path(self):
        
        # ist der berechnete pfad noch aktuell
        if not self.up2date:
            
            # vordefinieren von variablen
            longest_path = [0]
            
            # für jede mögliche startposition wird der längste pfad ermittelt
            for from_node, start_node, start_len in self.get_startpos(self.graph):
                self.step_path(self.graph, from_node, start_node, [], start_len, longest_path)
                
            # abspeicher des längsten pfads
            self.path_len = longest_path[0]
            self.up2date = True
            
        # rückgabe des längsten pfads
        return self.path_len
    
    ''' Funktion zum festlegen der Startpunkte '''
    @staticmethod
    def get_startpos(graph):        
        
        # vorbelegen von variablen
        start_pos = [[], [], []]
        
        # durchlaufen aller knoten
        for node in graph.nodes:
            
            # auslesen des knoten
            neighbors = [n for n in graph.neighbors(node)]
            from_node = node
            start_node = neighbors[0]
            start_len = graph.nodes[node]['weight']            
            
            # eintragen in die liste
            start_pos[graph.degree(node) - 1].append((from_node, start_node, start_len))
            
        # falls ein offenes ende vorliegt
        if not not start_pos[0]:
            start_pos = start_pos[0]
        elif not not start_pos[2]:
            start_pos = start_pos[2][0]
        else:
            start_pos = start_pos[1][0]
        return start_pos
    
    ''' Durchschreiten des Pfades '''
    @staticmethod
    def step_path(graph, from_node, start_node, visited_edges, current_len, max_len):
        
        # falls der neue knoten bereits besucht wurde
        if start_node in visited_edges:
            
            # falls der neue pfad länger ist als der bereits gefundene
            if current_len > max_len[0]:
                max_len[0] = current_len
                
            # abbruch der funktion
            return
        
        # nächsten knoten ermitteln
        next_nodes = [n for n in graph.neighbors(start_node)]
        next_nodes.remove(from_node)
        
        # falls eine sackgasse erreicht wurde
        if not next_nodes:
            
            # falls der neue pfad länger ist als der bereits gefundene
            if current_len > max_len[0]:
                max_len[0] = current_len
                
            # abbruch der funktion
            return
        
        # den pfad für jeden nächsten knoten gehen
        for next_node in next_nodes:
            
            # falls eine kante besucht wurde
            if graph.nodes[start_node]['weight'] > 0:
                next_len = current_len + 1
                visited_edges = visited_edges.copy() + [start_node]
            
            # falls ein knoten besucht wurde
            else:
                next_len = current_len
                visited_edges = visited_edges.copy()
                
            # rekursiver funktionsaufruf
            OwnGraph.step_path(graph, start_node, next_node, visited_edges, next_len, max_len)

'''
Unterklasse für die Topologie Klasse: Diese Klasse enthält die Funktionen und
Daten um die erreichbaren Knoten zu bestimmen
'''
class ReachGraph():
    
    ''' Initialisieren der Klasse '''
    def __init__(self, matchfield, graph):
        self.matchfield = matchfield
        self.graph = graph.copy()
        self.graph_components = []
        self.blocked_nodes = []
        self.own_nodes = set()
        self.reach_nodes = set()
        self.reach_edges = set()
        self.up2date = False
        
    ''' Zurücksetzen der Klasse '''
    def reset(self, graph):
        self.graph = graph.copy()
        self.graph_components.clear(), self.reach_nodes.clear(), 
        self.reach_edges.clear(), self.blocked_nodes.clear(), self.own_nodes.clear()
        self.up2date = False
        
    ''' Funktion zum entfernen eines Knoten '''
    def remove_node(self, node):
        
        # entfernen des knoten
        self.blocked_nodes.append(node)
        self.graph.remove_node(node)
            
        # ist der graph noch up2date
        self.up2date = False
        
    ''' Funktion zum entfernen einer Kante '''
    def remove_edge(self, edge):
        
        # entpacken der kante
        node_1, node_2 = edge
        
        # entfernen der kante
        if self.graph.has_edge(node_1, node_2):
            self.graph.remove_edge(node_1, node_2)
            self.up2date = False
        
    ''' Funktion zum hinzufügen einer kante '''
    def add_edge(self, edge):
        
        # entpacken der kante
        node_1, node_2 = edge
        
        # eintragen in die eigenen knoten
        if not node_1 in self.blocked_nodes:
            self.own_nodes.add(node_1)
            
        if not node_2 in self.blocked_nodes:
            self.own_nodes.add(node_2)
            
        # ist der graph noch up2date
        self.up2date = False
        
    ''' rückgabe aller erreichbaren knoten '''
    def get_nodes(self):
        # falls diese liste noch aktuell ist
        if not self.up2date:
            
            # ansonsten wird die liste neu berechnet
            self.reach_nodes.clear() # erreichbare knoten im gesamt-graphen
            self.graph_components.clear()
            for comp in nx.connected_components(self.graph):
                intersect = self.own_nodes & comp
                if not not intersect:
                    self.reach_nodes.update(comp)
                    self.graph_components.append((comp, intersect))
            
            # notieren dass diese arbeit erledigt wurde
            self.up2date = True
        
        # speichern der liste
        return self.reach_nodes
    
    ''' rückgabe aller erreichbaren kanten '''
    def get_edges(self):
        # falls die liste noch aktuell ist
        if not self.up2date:
        
            # durchlaufen aller erreichbaren knoten
            self.reach_edges.clear()
            for node in self.get_nodes():
                new_edges = self.matchfield.node2edgeID[node]
                self.reach_edges.update(new_edges)
            
            # notieren dass diese arbeit erledigt wurde
            self.up2date = True
            
        # speichern der neuen liste
        return self.reach_edges

'''
Klasse einer Spielaktion: Hier wird eine Spielaktion festgelegt die in einzelne
Spielzüge aufgeteilt werden kann. Ein Objekt dieses Typs wird bei jedem spieler
hinterlegt
'''
class Action():
    
    ''' Initialisieren der Klasse '''
    def __init__(self, player, matchfield):
        # zugriff auf spieler und spielfeld
        self.player = player
        self.matchfield = matchfield
        
        # initialisieren der anderen variablen
        self.street_pool = [Turn({'player': self.player, 'type': 'straße'}) for t in range(100)]
        self.target_turn = Turn({'player': self.player})
        self.action_vec = np.zeros(181)
        self.turn_list = []
        self.cost = ResourceSet()
        self.victorypoints = 0
        self.needed_time = 0
        self.bonus_left = {'entwicklung': 0, 'straßenbau': 0}
        
    ''' Zurücksetzen des next_turns '''
    def reset(self, policy_vector):
        # hier wird das ziel als Turn hinterlegt
        self.action_vec = policy_vector
        self.set_target_turn(policy_vector)
        
        # hier wird die variable turn_list und cost gesetzt
        self.set_turnlist_cost()
        
        # hier wird die variable victorypoints gesetzt
        self.set_victorypoints()
        
        # hier wird die variable needed_time gesetzt
        self.set_needed_time()
        
    ''' Rückgabe des policy_vectors als true-false vektor '''
    def vector_form(self):
        
        # umwandeln des policyvektors in eine true false form
        action_vec = np.zeros(181)
        action_vec[np.argmax(self.action_vec)] = 1
        
        # rückgabe des aktionsvektors
        return action_vec
        
    ''' Errechnen der Anzahl an notwendigen Zügen um die kosten decken zu können '''
    def set_needed_time(self):
        
        # shortcuts
        entw_bonus = 2 * self.player.entw_bonus['entwicklung']
        income = self.player.avg_income.np_array()
        cost = self.cost.np_array()
        
        # bestimmen der standartschrittweite
        valid_i = income > 0
        stepsize_base = np.ones(6) * 10000
        stepsize_base[valid_i] = np.ones(valid_i.sum()) / income[valid_i]
        exchange_rate = np.array([4, 4, 4, 4, 4, 1])
        
        # durchlaufen der schleife
        valid_c = cost > 0
        stepsize = stepsize_base * exchange_rate
        stepsize[valid_c] = stepsize_base[valid_c] * cost[valid_c]
        rounds, surplus = 0, 0
        while True:
            
            # sonst wird geschaut ob eine entwicklung vorhanden ist
            if entw_bonus > 0:
                surplus += 1
                entw_bonus -= 1
                
            # ansonsten wird 4:1 getauscht
            else:
                # bestimmen der nächsten schrittweite
                next_step = np.min(stepsize)
                rounds += next_step
                
                # updaten des stepsize-vektors
                step_sel = stepsize == next_step
                stepsize -= next_step
                stepsize[step_sel] = stepsize_base[step_sel] * exchange_rate[step_sel]
                
            # momentaner zustand
            state = income * rounds - cost
            deficit = state[state < 0].sum()
            sel_state = state >= 0
            surplus = (state[sel_state] // 4).sum()
                
            # falls der überschuss das defizit aufwiegt
            if surplus + deficit >= 0:
                break
            
        # abspeichern der notwendigen zeit
        self.bonus_left['entwicklung'] = entw_bonus//2
        self.needed_time = rounds
        
    ''' bestimmen der belohnung '''
    def set_victorypoints(self):
        # shortcuts
        target_type = self.target_turn['type']
        # falls eine stadt oder siedlung gebaut wird
        if target_type == 'siedlung' or target_type == 'stadt':
            self.victorypoints = 1            
        # ansonsten bekommt man keine unmittelbaren punkte
        else:
            self.victorypoints = 0
            
    ''' zusammenstellen der liste der spielzüge und der kosten '''
    def set_turnlist_cost(self):
        # shortcuts
        target_type = self.target_turn['type']
        spent_bonus = 0
        
        # löschen der züge
        self.turn_list.clear()
        self.cost.drain()
        
        # falls kein zug mehr möglich ist
        if target_type == 'none':
            return
        
        # falls eine entwicklung gekauft wird
        if target_type == 'entwicklung' or target_type == 'stadt':
            self.turn_list.append(self.target_turn)
            self.cost += self.target_turn.cost()
            return
        
        # falls eine siedlung gebaut wird
        if target_type == 'siedlung':
            t_node = self.target_turn['position']
            _, path = self.get_path(t_node)
            
        # falls eine straße gebaut wird
        if target_type == 'straße':
            t_edge = self.target_turn['position']
            t_node1, t_node2 = t_edge
            
            # pfad zu beiden knoten
            tu_1 = self.get_path(t_node1)
            tu_2 = self.get_path(t_node2)
            
            # der kürzere Pfad wird verwendet
            path = min(tu_1, tu_2)[1]
        
        # umformen des pfades in spielzüge
        bonus = self.player.entw_bonus['straßenbau'] *2
        for i, edge in enumerate(path):
            self.street_pool[i]['position'] = edge
            self.turn_list.append(self.street_pool[i])
            
            # falls ein bonus vorhanden ist wird er aufgebraucht
            if bonus > 0:
                bonus -= 1
            else:
                self.cost += self.street_pool[i].cost()
                
        # der letzte schritt ist der zielzug
        self.bonus_left['straßenbau'] = bonus // 2
        self.turn_list.append(self.target_turn)
        self.cost += self.target_turn.cost()
    
    def get_path(self, t_node):
        # vordefinieren von variablen und shortcuts
        shortest_path = []
        shortest_path_len = 10000
        reach_graph = self.player.topology.reach_graph
        
        # jede komponente wird nach dem zielknoten durchsucht
        for comp, intersect in reach_graph.graph_components:
            
            # falls die komponente den zielknoten enthält, wird der kürzeste
            # pfad zu einem der interset knoten gesucht
            if t_node in comp:
                
                # für jeden möglichen startknoten wird der kürzeste pfad gesucht
                for s_node in intersect:
                    current_path = nx.shortest_path(reach_graph.graph, 
                                                    s_node, t_node)
                    current_len = len(current_path)
                    
                    # falls der neue pfad kürzer ist dann ersetzen
                    if len(current_path) < shortest_path_len:
                        shortest_path = current_path.copy()
                        shortest_path_len = current_len
            
        # der kürzeste pfad wird umgewandelt in eine turn_list
        return (shortest_path_len, [frozenset([n,m]) for n,m in \
                                    zip(shortest_path[:-1], shortest_path[1:])])
        
    ''' updaten eines Action Objekts aus einem policy-vektor '''
    def set_target_turn(self, policy_vector):
        
        # zurücksetzen des Einkommens des arget_tuen
        self.target_turn['avg_income'].drain()
        
        # falls keine aktion möglich ist
        if not any(policy_vector > 0):
            self.target_turn['type'] = 'none'
            return
        
        # index des höchsten wertes ermitteln
        max_policy = policy_vector.argmax()
        max_policy = max_policy.item()
        
        # falls der index hier liegt, ist es eine straße
        if max_policy < 72:
            self.target_turn['position'] = self.matchfield.ID2edge[max_policy]
            self.target_turn['type'] = 'straße'
            
        # falls der index hier liegt, ist es eine siedlung
        elif 72 <= max_policy < 126:
            self.target_turn['position'] = max_policy - 72
            self.target_turn['type'] = 'siedlung'
            self.target_turn['resourceSet'] = self.matchfield.nodes[max_policy - 72]['resourceSet']
        
        # falls der index hier liegt, ist es eine stadt
        elif 126 <= max_policy < 180:
            self.target_turn['position'] = max_policy - 126
            self.target_turn['type'] = 'stadt'
            self.target_turn['resourceSet'] = self.matchfield.nodes[max_policy - 126]['resourceSet']
        
        # falls der index hier liegt, ist es eine entwicklung
        elif 180 <= max_policy:
            self.target_turn['type'] = 'entwicklung'