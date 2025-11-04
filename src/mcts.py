import numpy as np
import math
from copy import copy
from config import Config as cfg

class Node:
    def __init__(self, prior_prob, player, parent=None, action_index=None): # create a new node for the action
        self.state = None
        self.player = player
        self.total_visits_N = 0 # total visits to the node
        self.total_action_value_of_next_state_W = 0 # total action value of the next state
        self.mean_action_value_of_next_state_Q = 0 # mean action value of the next state
        self.prior_probs_P = prior_prob # prior probability of the action
        self.children = {} # children of the node
        self.parent = parent # parent of the node
        if action_index:            
            state = copy(parent.state) * -1
            state[action_index] = -1
            self.state = copy(state)

    def set_state(self,state):
        self.state = state

    def expand(self,action_probs,player,parent):
        for i,action_prob in enumerate(action_probs):
            if action_prob != 0:
                self.children[i] = Node(action_prob, player, parent, i) # create a new node for the action

    def is_leaf_node(self):
        return len(self.children)==0

    def select_best_child(self):
        best_uscore = float('-inf')
        for i, child in self.children.items():
            psa = child.prior_probs_P
            Ns = self.total_visits_N
            Nsa = child.total_visits_N
            Cs = 1 # exploration constant
            Q = child.mean_action_value_of_next_state_Q 
            Uscore = Q + Cs * psa * math.sqrt(Ns) / (1 + Nsa)
            if best_uscore < Uscore: # update the best child if the new child has a higher Uscore
                best_uscore = Uscore
                best_child_index = i
        return best_child_index,self.children[best_child_index]  
class MonteCarloTreeSearch:
    def __init__(self,game,policy_value_network):
        self.game = game
        self.policy_value_network = policy_value_network
        
    def init_root_node(self):
        root_state = np.zeros(cfg.ACTION_SIZE) 
        root_node = Node(prior_prob=0,player=1,action_index=None)
        root_node.set_state(root_state)
        return root_node
    def backup(self,mtc_steps,winner,player,value):
        for node in reversed(mtc_steps):
            node.total_visits_N+=1
            if winner==None:
                value = value
            elif winner ==0:
                value = 0
            else:
                value = -1 if winner == node.player else 1
            node.total_action_value_of_next_state_W = node.total_action_value_of_next_state_W + value
            node.mean_action_value_of_next_state_Q = node.total_action_value_of_next_state_W/node.total_visits_N

    def run_simulation(self, root_node, num_simulations=1600, player=1): # run the simulation for the given number of simulations
        root_state = root_node.state
        next_player = -1 * player # switch player
        value, action_probs = self.policy_value_network(root_state, player) #call the nn with player info
        valid_moves = self.game.get_valid_moves(root_state)
        action_probs = action_probs * valid_moves
        root_node.expand(action_probs=action_probs,player=next_player,parent=root_node)
        for _ in range(num_simulations):
            step_count = 0
            backup_steps = [root_node]
            node = root_node
            while node.is_leaf_node() == False: # keep traversing until we read a leaf node
                step_count += 1
                action_index, node = node.select_best_child()
                backup_steps.append(node)
            leaf_node = node 
            parent_node = backup_steps[-2]
            action = np.zeros(cfg.ACTION_SIZE)
            action[action_index] = 1
            leaf_node_state = self.game.get_next_state_from_next_player_prespective(parent_node.state, action, player) # get the next state from the next player's perspective
            leaf_node.set_state(leaf_node_state)

            value,action_probs = self.policy_value_network(leaf_node_state, leaf_node.player) # pass player for canonical representation
            winner = self.game.get_reward_for_next_player(leaf_node_state,leaf_node.player)
            
            if winner == None:
                valid_moves = self.game.get_valid_moves(leaf_node_state)
                action_probs = action_probs * valid_moves
                next_player = leaf_node.player*-1
                leaf_node.expand(action_probs=action_probs, player=next_player, parent=leaf_node)
            
            self.backup(backup_steps,winner,player,value)  
        return root_node
    def select_move(self,node,mode="exploit",temperature=1):
        visits = [(k, v.total_visits_N) for k, v in node.children.items()] # list of tuples (action_index, total_visits)
        # exploit: select the action with the most visits
        # explore: select the action with the probability of the visits
        if mode=="exploit":
            action_index = max(visits,key=lambda t: t[1])[0]
        elif mode=="explore":
            visit_options = [k for k, v in node.children.items()] # list of action indices
            probs = [v.total_visits_N ** (1 / temperature) for k, v in node.children.items()] # list of probabilities
            probs = [t / sum(probs) for t in probs] # normalize the probabilities
            action_index = np.random.choice(visit_options,1,p=probs)[0]
        action = np.zeros(cfg.ACTION_SIZE)
        action[action_index]=1
        subtree = node.children[action_index]
        action_probs = np.zeros(cfg.ACTION_SIZE)
        for k,v in node.children.items():
            action_probs[k] = v.total_visits_N
        action_probs = [action_prob/sum(action_probs) for action_prob in action_probs]
        return action, subtree, action_probs
        
            