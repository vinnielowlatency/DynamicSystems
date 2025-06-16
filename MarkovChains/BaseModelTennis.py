import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class TennisGameMarkovChain:
    
    def __init__(self, p_A_wins_point=0.5, is_base_model=True):
        if is_base_model and p_A_wins_point != 0.5:
            print("Warning: Base model assumes equal players (p=0.5). Setting p=0.5.")
            p_A_wins_point = 0.5
            
        self.p = p_A_wins_point
        self.q = 1 - p_A_wins_point
        self.is_base_model = is_base_model
        
        self.states = [
            '0-0', '0-15', '15-0', '15-15', '30-0', '30-15', 
            '0-30', '15-30', '30-30', '40-0', '40-15', '40-30',
            '0-40', '15-40', '30-40', 'DEUCE', 'ADV-A', 'ADV-B',
            'A-WINS', 'B-WINS'
        ]
        self.n_states = len(self.states)
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.transition_matrix = self._build_transition_matrix()
        
    def _build_transition_matrix(self):
        n = self.n_states
        P = np.zeros((n, n))
        
        # Helper function to add transitions
        def add_transition(from_state, to_state_A, to_state_B):
            from_idx = self.state_to_idx[from_state]
            to_idx_A = self.state_to_idx[to_state_A]
            to_idx_B = self.state_to_idx[to_state_B]
            P[from_idx, to_idx_A] = self.p
            P[from_idx, to_idx_B] = self.q
        
        add_transition('0-0', '15-0', '0-15')
        add_transition('15-0', '30-0', '15-15')
        add_transition('0-15', '15-15', '0-30')
        add_transition('30-0', '40-0', '30-15')
        add_transition('15-15', '30-15', '15-30')
        add_transition('0-30', '15-30', '0-40')
        add_transition('30-15', '40-15', '30-30')
        add_transition('15-30', '30-30', '15-40')
        add_transition('30-30', '40-30', '30-40')
        
        add_transition('40-0', 'A-WINS', '40-15')
        add_transition('40-15', 'A-WINS', '40-30')
        add_transition('40-30', 'A-WINS', 'DEUCE')
        add_transition('0-40', '15-40', 'B-WINS')
        add_transition('15-40', '30-40', 'B-WINS')
        add_transition('30-40', 'DEUCE', 'B-WINS')
        
        add_transition('DEUCE', 'ADV-A', 'ADV-B')
        add_transition('ADV-A', 'A-WINS', 'DEUCE')
        add_transition('ADV-B', 'DEUCE', 'B-WINS')
        
        P[self.state_to_idx['A-WINS'], self.state_to_idx['A-WINS']] = 1.0
        P[self.state_to_idx['B-WINS'], self.state_to_idx['B-WINS']] = 1.0
        
        return P
    
    def get_Q_R_matrices(self):
        Q = self.transition_matrix[:-2, :-2]
        R = self.transition_matrix[:-2, -2:]
        return Q, R
    
    def game_win_probability(self):
        Q, R = self.get_Q_R_matrices()
        N = np.linalg.inv(np.eye(len(Q)) - Q)
        B = N @ R
        start_idx = self.state_to_idx['0-0']
        A_wins_col = 0
        return B[start_idx, A_wins_col]
    
    def expected_game_length(self):
        Q, R = self.get_Q_R_matrices()
        N = np.linalg.inv(np.eye(len(Q)) - Q)
        start_idx = self.state_to_idx['0-0']
        return np.sum(N[start_idx, :])
    
    def simulate_game(self):
        current_state = '0-0'
        points_played = 0
        
        while current_state not in ['A-WINS', 'B-WINS']:
            points_played += 1
            current_idx = self.state_to_idx[current_state]
            
            probs = self.transition_matrix[current_idx, :]
            next_idx = np.random.choice(len(probs), p=probs)
            current_state = self.states[next_idx]
        
        winner = 'A' if current_state == 'A-WINS' else 'B'
        return winner, points_played
    
    def create_matrix_table(self, figsize=(18, 12)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')
        
        matrix_data = []
        for i, from_state in enumerate(self.states):
            row = [from_state]
            for j, to_state in enumerate(self.states):
                val = self.transition_matrix[i, j]
                if val == 0:
                    row.append('0.0')
                elif val == 1:
                    row.append('1.0')
                else:
                    row.append(f'{val:.1f}')
            matrix_data.append(row)
        
        columns = ['From\\To'] + self.states
        
        table = ax.table(cellText=matrix_data,
                        colLabels=columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.2, 1.8)
        
        for i in range(len(matrix_data)):
            for j in range(len(columns)):
                cell = table[(i+1, j)]
                if j == 0:
                    cell.set_facecolor('#E6E6FA')
                elif j-1 < len(self.states) and matrix_data[i][j] not in ['0.0']:
                    if matrix_data[i][j] == '1.0':
                        cell.set_facecolor('#FFB6C1')
                    else:
                        cell.set_facecolor('#ADD8E6')
        
        for j in range(len(columns)):
            cell = table[(0, j)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        
        model_type = "Base Model" if self.is_base_model else "Extension"
        plt.title(f'Tennis Game Transition Matrix - {model_type} (p={self.p:.1f}, q={self.q:.1f})', 
                 fontsize=16, fontweight='bold', pad=20)
        return fig
    
    def display_matrix_summary(self):
        Q, R = self.get_Q_R_matrices()
        model_type = "Base Model" if self.is_base_model else "Extension"
        
        print(f"\n=== {model_type} - Transition Matrix Properties ===")
        print(f"  - Full matrix P: {self.transition_matrix.shape}")
        print(f"  - Q matrix (transient): {Q.shape}")
        print(f"  - R matrix (absorbing): {R.shape}")
        print(f"  - Non-zero entries: {np.count_nonzero(self.transition_matrix)}")
        print(f"  - Player A point probability: {self.p:.3f}")
        print(f"  - Game win probability (A): {self.game_win_probability():.4f}")
        print(f"  - Expected game length: {self.expected_game_length():.2f} points")


class TennisExtension:
    
    def __init__(self, skill_scale_factor=2.0):
        self.skill_scale_factor = skill_scale_factor
    
    def skill_to_probability(self, skill_A, skill_B):
        skill_diff = skill_A - skill_B
        probability = 1 / (1 + np.exp(-self.skill_scale_factor * skill_diff))
        return np.clip(probability, 0.01, 0.99)
    
    def create_skill_based_game(self, skill_A, skill_B):
        p = self.skill_to_probability(skill_A, skill_B)
        return TennisGameMarkovChain(p, is_base_model=False)
    
    def simulate_match(self, skill_A, skill_B, sets_to_win=2):
        sets_A = 0
        sets_B = 0
        total_games = 0
        
        while sets_A < sets_to_win and sets_B < sets_to_win:
            games_A = 0
            games_B = 0
            
            while True:
                game = self.create_skill_based_game(skill_A, skill_B)
                winner, _ = game.simulate_game()
                total_games += 1
                
                if winner == 'A':
                    games_A += 1
                else:
                    games_B += 1
                
                if games_A >= 6 and games_A - games_B >= 2:
                    sets_A += 1
                    break
                elif games_B >= 6 and games_B - games_A >= 2:
                    sets_B += 1
                    break
                elif games_A >= 8 or games_B >= 8:
                    if games_A > games_B:
                        sets_A += 1
                    else:
                        sets_B += 1
                    break
        
        winner = 'A' if sets_A > sets_B else 'B'
        return winner, sets_A, sets_B, total_games


def demonstrate_base_vs_extension():
    print("=" * 60)
    print("TENNIS MARKOV CHAIN: BASE MODEL VS EXTENSION")
    print("=" * 60)
    
    # Base Model - Equal players
    print("\n### BASE MODEL: Equal Players (p = 0.5) ###")
    base_game = TennisGameMarkovChain(p_A_wins_point=0.5, is_base_model=True)
    base_game.display_matrix_summary()
    
    # Extension - Skill-based
    print("\n### EXTENSION: Skill-Based Probabilities ###")
    extension = TennisExtension()
    
    # Example with different skill levels
    skill_scenarios = [
        (0.5, 0.5, "Equal skills"),
        (0.6, 0.4, "Moderate skill difference"), 
        (0.8, 0.3, "Large skill difference")
    ]
    
    for skill_A, skill_B, description in skill_scenarios:
        print(f"\n{description}: Skill A={skill_A}, Skill B={skill_B}")
        p = extension.skill_to_probability(skill_A, skill_B)
        ext_game = extension.create_skill_based_game(skill_A, skill_B)
        print(f"  Point probability: {p:.3f}")
        print(f"  Game win probability: {ext_game.game_win_probability():.3f}")


def demonstrate_amplification_effect():
    print("\n" + "=" * 50)
    print("AMPLIFICATION EFFECT ANALYSIS")
    print("=" * 50)
    print("Point Prob | Game Prob | Amplification Factor")
    print("-" * 45)
    
    for p in [0.50, 0.52, 0.55, 0.60, 0.65, 0.70]:
        game = TennisGameMarkovChain(p, is_base_model=False)
        game_prob = game.game_win_probability()
        if p == 0.5:
            amplification = 1.0
        else:
            point_advantage = p - 0.5
            game_advantage = game_prob - 0.5
            amplification = game_advantage / point_advantage
        print(f"   {p:.2f}    |   {game_prob:.3f}   |      {amplification:.3f}")


if __name__ == "__main__":
    demonstrate_base_vs_extension()
    
    demonstrate_amplification_effect()
    
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS...")
    print("=" * 50)
    
    base_game = TennisGameMarkovChain(0.5, is_base_model=True)
    fig_base = base_game.create_matrix_table()
    plt.show()
    
    extension = TennisExtension()
    ext_game = extension.create_skill_based_game(0.6, 0.4)
    fig_ext = ext_game.create_matrix_table()
    plt.show()