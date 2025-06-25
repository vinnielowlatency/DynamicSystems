import numpy as np
from scipy.linalg import inv
from config import Config

class TennisMarkovChain: #main class to handle markov chain logic with all states
    def __init__(self):
        self.states = [
            '0-0', '15-0', '0-15', '15-15', '30-0', '30-15', '0-30', '15-30', '30-30',
            '40-0', '40-15', '40-30', '0-40', '15-40', '30-40', 'DEUCE', 'ADV-A', 'ADV-B',
            'A-WINS', 'B-WINS'
        ]
        self.n_transient = 18
        self.n_absorbing = 2
        
    def build_transition_matrix(self, p):
        n = len(self.states)
        P = np.zeros((n, n))
        state_idx = {state: i for i, state in enumerate(self.states)}
        
        #defining all state transitions
        transitions = {
            '0-0': [('15-0', p), ('0-15', 1-p)],
            '15-0': [('30-0', p), ('15-15', 1-p)],
            '0-15': [('15-15', p), ('0-30', 1-p)],
            '15-15': [('30-15', p), ('15-30', 1-p)],
            '30-0': [('40-0', p), ('30-15', 1-p)],
            '30-15': [('40-15', p), ('30-30', 1-p)],
            '0-30': [('15-30', p), ('0-40', 1-p)],
            '15-30': [('30-30', p), ('15-40', 1-p)],
            '30-30': [('40-30', p), ('30-40', 1-p)],
            '40-0': [('A-WINS', p), ('40-15', 1-p)],
            '40-15': [('A-WINS', p), ('40-30', 1-p)],
            '40-30': [('A-WINS', p), ('DEUCE', 1-p)],
            '0-40': [('15-40', p), ('B-WINS', 1-p)],
            '15-40': [('30-40', p), ('B-WINS', 1-p)],
            '30-40': [('DEUCE', p), ('B-WINS', 1-p)],
            'DEUCE': [('ADV-A', p), ('ADV-B', 1-p)],
            'ADV-A': [('A-WINS', p), ('DEUCE', 1-p)],
            'ADV-B': [('DEUCE', p), ('B-WINS', 1-p)],
            'A-WINS': [('A-WINS', 1.0)],
            'B-WINS': [('B-WINS', 1.0)]
        }
         # fill transition matrix
        for state, state_transitions in transitions.items():
            from_idx = state_idx[state]
            for next_state, prob in state_transitions:
                to_idx = state_idx[next_state]
                P[from_idx, to_idx] = prob
                
        return P
    
    def get_fundamental_matrix(self, p):
        P = self.build_transition_matrix(p)
        
        # extract Q (transient to transient) and R (transient to absorbing)
        Q = P[:self.n_transient, :self.n_transient]
        R = P[:self.n_transient, self.n_transient:]
        # calculate fundamental matrix
        I = np.eye(self.n_transient)
        N = inv(I - Q)
        # calculate absorption probabilities and times
        B = N @ R
        absorption_times = N.sum(axis=1)
        
        return {
            'fundamental_matrix': N,
            'absorption_probabilities': B,
            'absorption_times': absorption_times,
            'transition_matrix': P,
            'Q_matrix': Q,
            'R_matrix': R
        }
    
    def get_game_win_probability(self, p):
        analysis = self.get_fundamental_matrix(p)
        return analysis['absorption_probabilities'][0, 0]  # start which is 0-0 to win 
    
    def monte_carlo_validation(self, p, num_simulations=10000):
        a_wins = 0
        total_steps = 0
        
        for _ in range(num_simulations):
            current_state = 0  # start at state '0-0'
            steps = 0
            
            while current_state < self.n_transient:
                steps += 1
                P = self.build_transition_matrix(p)
                probs = P[current_state, :]
                current_state = np.random.choice(len(self.states), p=probs)
            total_steps += steps
            if current_state == self.n_transient: 
                a_wins += 1
        return {
            'empirical_prob_a_wins': a_wins / num_simulations,
            'empirical_mean_absorption_time': total_steps / num_simulations
        }