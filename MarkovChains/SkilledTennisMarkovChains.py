import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TennisGameMarkovChain:
    
    def __init__(self, p_A_wins_point=0.6):
        self.p = p_A_wins_point
        self.q = 1 - p_A_wins_point
        
        # state names for the matrix
        self.states = [
            '0-0', '0-15', '15-0', '15-15', '30-0', '30-15', 
            '0-30', '15-30', '30-30', '40-0', '40-15', '40-30',
            '0-40', '15-40', '30-40', 'DEUCE', 'ADV-A', 'ADV-B',
            'A-WINS', 'B-WINS'
        ]
        self.n_states = len(self.states)
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.transition_matrix = self._build_transition_matrix()
        
    def _build_transition_matrix(self): #clean transition matrix with clear state names
        n = self.n_states
        P = np.zeros((n, n))
        #helper function 
        def add_transition(from_state, to_state_A, to_state_B):
            from_idx = self.state_to_idx[from_state]
            to_idx_A = self.state_to_idx[to_state_A]
            to_idx_B = self.state_to_idx[to_state_B]
            P[from_idx, to_idx_A] = self.p
            P[from_idx, to_idx_B] = self.q
        
        # natural game progressions
        add_transition('0-0', '15-0', '0-15')
        add_transition('15-0', '30-0', '15-15')
        add_transition('0-15', '15-15', '0-30')
        add_transition('30-0', '40-0', '30-15')
        add_transition('15-15', '30-15', '15-30')
        add_transition('0-30', '15-30', '0-40')
        add_transition('30-15', '40-15', '30-30')
        add_transition('15-30', '30-30', '15-40')
        add_transition('30-30', '40-30', '30-40')
        
        # special transitions to winning/deuce
        add_transition('40-0', 'A-WINS', '40-15')
        add_transition('40-15', 'A-WINS', '40-30')
        add_transition('40-30', 'A-WINS', 'DEUCE')
        add_transition('0-40', '15-40', 'B-WINS')
        add_transition('15-40', '30-40', 'B-WINS')
        add_transition('30-40', 'DEUCE', 'B-WINS')
        
        # deuce and advantage
        add_transition('DEUCE', 'ADV-A', 'ADV-B')
        add_transition('ADV-A', 'A-WINS', 'DEUCE')
        add_transition('ADV-B', 'DEUCE', 'B-WINS')
        
        # absorbing states
        P[self.state_to_idx['A-WINS'], self.state_to_idx['A-WINS']] = 1.0
        P[self.state_to_idx['B-WINS'], self.state_to_idx['B-WINS']] = 1.0
        
        return P
    
    def get_Q_R_matrices(self):
        # last 2 states are absorbing
        Q = self.transition_matrix[:-2, :-2]
        R = self.transition_matrix[:-2, -2:]
        return Q, R
    
    def create_matrix_table(self, figsize=(18, 12)): # plotting functions
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')
        
        # create data to populate table
        matrix_data = []
        for i, from_state in enumerate(self.states):
            row = [from_state]  # first column is from state
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
        ## create table
        table = ax.table(cellText=matrix_data,
                        colLabels=columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.2, 1.8)
        
        # colour code
        for i in range(len(matrix_data)):
            for j in range(len(columns)):
                cell = table[(i+1, j)]  # +1 because of header row
                if j == 0:  # first column
                    cell.set_facecolor('#E6E6FA')
                elif j-1 < len(self.states) and matrix_data[i][j] not in ['0.0']:
                    if matrix_data[i][j] == '1.0':
                        cell.set_facecolor('#FFB6C1')
                    else:
                        cell.set_facecolor('#ADD8E6')  # red for absorpotion blue for transition
        for j in range(len(columns)):
            cell = table[(0, j)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        plt.title(f'Tennis Game Transition Matrix P (p={self.p:.1f}, q={self.q:.1f})', 
                 fontsize=16, fontweight='bold', pad=20)
        return fig
    
    def create_network_diagram(self, figsize=(16, 12)):
        fig, ax = plt.subplots(figsize=figsize)    
        # define positions for clear layout
        positions = {
            '0-0': (4, 0),
            '15-0': (2, 1), '0-15': (6, 1),
            '30-0': (0, 2), '15-15': (4, 2), '0-30': (8, 2),
            '40-0': (-1, 3), '30-15': (2, 3), '15-30': (6, 3), '0-40': (9, 3),
            '40-15': (0, 4), '30-30': (4, 4), '15-40': (8, 4),
            '40-30': (2, 5), '30-40': (6, 5),
            'DEUCE': (4, 6),
            'ADV-A': (2, 7), 'ADV-B': (6, 7),
            'A-WINS': (0, 8), 'B-WINS': (8, 8)
        }
        
        # Draw nodes
        for state, (x, y) in positions.items():
            if state in ['A-WINS', 'B-WINS']:
                circle = plt.Circle((x, y), 0.4, color='lightcoral', alpha=0.8, zorder=2)
            else:
                circle = plt.Circle((x, y), 0.4, color='lightblue', alpha=0.8, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, state, ha='center', va='center', fontsize=8, 
                   fontweight='bold', zorder=3)
        
        # key transitions
        def draw_arrow(from_state, to_state, prob, offset_factor=0.1):
            if from_state in positions and to_state in positions:
                x1, y1 = positions[from_state]
                x2, y2 = positions[to_state]
                # calculate direction offset
                dx = x2 - x1
                dy = y2 - y1
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    # offset from circle edges
                    offset = 0.45
                    start_x = x1 + (dx/length) * offset
                    start_y = y1 + (dy/length) * offset
                    end_x = x2 - (dx/length) * offset
                    end_y = y2 - (dy/length) * offset
                    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5))
                    mid_x = (start_x + end_x) / 2 #probability label
                    mid_y = (start_y + end_y) / 2
                    perp_x = -dy/length * offset_factor
                    perp_y = dx/length * offset_factor
                    ax.text(mid_x + perp_x, mid_y + perp_y, f'{prob:.1f}', 
                           fontsize=7, ha='center', va='center', 
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='yellow', 
                                   alpha=0.7, edgecolor='none'))
        
        # draw all non-zero transitions
        for i, from_state in enumerate(self.states):
            for j, to_state in enumerate(self.states):
                prob = self.transition_matrix[i, j]
                if prob > 0 and i != j:  # skip self loops for clarification
                    draw_arrow(from_state, to_state, prob)
        # plot properties
        ax.set_xlim(-2, 10)
        ax.set_ylim(-1, 9)
        ax.set_aspect('equal')
        ax.set_title(f'Tennis Game State Transition Network (p={self.p:.1f}, q={self.q:.1f})', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    def game_win_probability(self):
        Q, R = self.get_Q_R_matrices()
        N = np.linalg.inv(np.eye(len(Q)) - Q)  # fundamental matrix
        B = N @ R  # absorption probabilities
        start_idx = self.state_to_idx['0-0']
        A_wins_col = 0  # first column of R corresponds to A-WINS
        return B[start_idx, A_wins_col]
    
    def expected_game_length(self):
        Q, R = self.get_Q_R_matrices()
        N = np.linalg.inv(np.eye(len(Q)) - Q)
        start_idx = self.state_to_idx['0-0']
        return np.sum(N[start_idx, :])
    
    def display_matrix_summary(self):
        Q, R = self.get_Q_R_matrices()
        print(f"Transition Matrix Properties:")
        print(f"  - Full matrix P: {self.transition_matrix.shape}")
        print(f"  - Q matrix (transient): {Q.shape}")
        print(f"  - R matrix (absorbing): {R.shape}")
        print(f"  - Non-zero entries: {np.count_nonzero(self.transition_matrix)}")
        print(f"  - Game win probability (A): {self.game_win_probability():.4f}")
        print(f"  - Expected game length: {self.expected_game_length():.2f} points")

def demonstrate_amplification_effect():
    """Show how point probabilities amplify to game probabilities"""
    print("\n=== Tennis Scoring Amplification Effect ===")
    print("Point Prob | Game Prob | Amplification Factor")
    print("-" * 45)
    
    for p in [0.50, 0.52, 0.55, 0.60, 0.65, 0.70]:
        game = TennisGameMarkovChain(p)
        game_prob = game.game_win_probability()
        amplification = game_prob / p if p > 0 else 1
        print(f"   {p:.2f}    |   {game_prob:.3f}   |      {amplification:.3f}")


if __name__ == "__main__":
    # create tennis game with p=0.6
    tennis = TennisGameMarkovChain(0.6)
    
    # display matrix properties
    tennis.display_matrix_summary()
    
    # display amplification effect
    demonstrate_amplification_effect()
    
    fig_table = tennis.create_matrix_table()
    plt.show()
    
    fig_network = tennis.create_network_diagram()
    plt.show()