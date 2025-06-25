import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta, pearsonr
from scipy.linalg import inv

# Note: This model runs a large amount of season and match simulations and will require a few minutes to run

class TennisMarkovChain:
    def __init__(self):
        self.states = [
            '0-0', '15-0', '0-15', '15-15', '30-0', '30-15', '0-30', '15-30', '30-30',
            '40-0', '40-15', '40-30', '0-40', '15-40', '30-40', 'DEUCE', 'ADV-A', 'ADV-B',
            'A-WINS', 'B-WINS'  # Absorbing states
        ]

    def build_transition_matrix(self, p):
        n = len(self.states)
        P = np.zeros((n, n))
        state_idx = {state: i for i, state in enumerate(self.states)}
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
        for state, state_transitions in transitions.items():
            from_idx = state_idx[state]
            for next_state, prob in state_transitions:
                to_idx = state_idx[next_state]
                P[from_idx, to_idx] = prob
        return P
    
    def get_game_win_probability(self, p):
        P = self.build_transition_matrix(p)
        Q = P[:18, :18]  # Transient states
        R = P[:18, 18:]  # Transitions to absorbing states
        I = np.eye(18)
        N = inv(I - Q)  # Fundamental matrix
        B = N @ R
        return B[0, 0]  # Probability A wins from start state

class TennisAnalysis:
    def __init__(self, sensitivity_k=2.0):
        self.markov_chain = TennisMarkovChain()
        self.sensitivity_k = sensitivity_k

    def generate_player_skills(self, num_players, alpha=2, beta_param=5):
        skills = beta.rvs(alpha, beta_param, size=num_players)
        return np.clip(skills, 0.1, 0.9)

    def calculate_point_prob(self, skill_a, skill_b):
        return 1 / (1 + np.exp(-self.sensitivity_k * (skill_a - skill_b)))

    def simulate_match(self, skill_a, skill_b, sets_to_win=2):
        p = self.calculate_point_prob(skill_a, skill_b)
        game_win_prob = self.markov_chain.get_game_win_probability(p)
        sets_a = sets_b = 0
        while sets_a < sets_to_win and sets_b < sets_to_win:
            games_a = games_b = 0
            while True:
                if np.random.random() < game_win_prob:
                    games_a += 1
                else:
                    games_b += 1
                # Check for set completion
                if (games_a >= 6 or games_b >= 6) and abs(games_a - games_b) >= 2:
                    break
                # Tiebreaker at 6-6
                if games_a == 6 and games_b == 6:
                    if np.random.random() < game_win_prob:
                        games_a += 1
                    else:
                        games_b += 1
                    break
            if games_a > games_b:
                sets_a += 1
            else:
                sets_b += 1
        return 0 if sets_a > sets_b else 1

    def seed_tournament(self, skills, current_points):
        if np.sum(current_points) == 0:
            ranking = np.argsort(skills)[::-1]
        else:
            ranking = np.argsort(current_points)[::-1]
        # Alternate top and bottom seeds for bracket balance
        seeded_players = []
        n = len(ranking)
        for i in range(n // 2):
            seeded_players.extend([ranking[i], ranking[n - 1 - i]])
        return seeded_players

    def run_tournament(self, skills, sets_to_win=2, current_points=None, use_seeding=True):
        if use_seeding and current_points is not None:
            players = self.seed_tournament(skills, current_points)
        else:
            players = list(range(len(skills)))
            np.random.shuffle(players)
        while len(players) > 1:
            next_round = []
            for i in range(0, len(players), 2):
                if i + 1 < len(players):
                    player_a, player_b = players[i], players[i + 1]
                    winner_idx = self.simulate_match(skills[player_a], skills[player_b], sets_to_win)
                    winner = player_a if winner_idx == 0 else player_b
                    next_round.append(winner)
                else:
                    next_round.append(players[i])  # Bye
            players = next_round
        return players[0]

    def run_season(self, skills, num_tournaments, sets_to_win=2):
        num_players = len(skills)
        player_points = np.zeros(num_players)
        tournament_wins = np.zeros(num_players)
        points_structure = {  # double check if this makes sense compared to ATP conventions
            'winner': 100,
            'finalist': 60,
            'semifinalist': 30,
            'quarterfinalist': 15
        }
        season_data = []
        for tournament_num in range(num_tournaments):
            if tournament_num == 0:
                seeded_players = self.seed_tournament(skills, np.zeros(num_players))
            else:
                seeded_players = self.seed_tournament(skills, player_points)
            players = seeded_players
            round_results = {}
            round_num = 0
            # Run tournament rounds
            while len(players) > 1:
                next_round = []
                round_results[round_num] = []
                for i in range(0, len(players), 2):
                    if i + 1 < len(players):
                        player_a, player_b = players[i], players[i + 1]
                        winner_idx = self.simulate_match(skills[player_a], skills[player_b], sets_to_win)
                        winner = player_a if winner_idx == 0 else player_b
                        loser = player_b if winner_idx == 0 else player_a
                        next_round.append(winner)
                        round_results[round_num].append({
                            'winner': winner,
                            'loser': loser,
                            'winner_skill': skills[winner],
                            'loser_skill': skills[loser]
                        })
                    else:
                        next_round.append(players[i])
                
                players = next_round
                round_num += 1
            
            # Award points based on final positions
            winner = players[0]
            tournament_wins[winner] += 1
            player_points[winner] += points_structure['winner']
            if round_num >= 1 and round_results[round_num-1]:
                finalist = round_results[round_num-1][0]['loser']
                player_points[finalist] += points_structure['finalist']
            if round_num >= 2:
                for match in round_results[round_num-2]:
                    player_points[match['loser']] += points_structure['semifinalist']
            current_ranking = np.argsort(player_points)[::-1]  # make sure this is logic is sound
            season_data.append({
                'tournament': tournament_num + 1,
                'winner': winner,
                'winner_skill': skills[winner],
                'points_distribution': player_points.copy(),
                'current_ranking': current_ranking.copy()
            })
        
        return {
            'final_points': player_points,
            'tournament_wins': tournament_wins,
            'season_data': season_data,
            'final_ranking': np.argsort(player_points)[::-1]
        }

    def skill_ranking_analysis(self, num_simulations=200, num_players=64):
        skills = self.generate_player_skills(num_players)
        true_skill_ranking = np.argsort(skills)[::-1]
        season_results = []
        all_skill_percentiles = []
        all_final_ranking_percentiles = []
        mobility_data = []
        for sim in range(num_simulations):
            season_result = self.run_season(skills, num_tournaments=12, sets_to_win=2)
            final_ranking = season_result['final_ranking']
            # Calculate correlation between skill and ranking
            ranking_scores = np.zeros(num_players)
            for i, player in enumerate(final_ranking):
                ranking_scores[player] = num_players - i
            correlation, p_value = pearsonr(skills, ranking_scores)
            # Calculate percentiles
            skill_ranks = np.argsort(np.argsort(skills)[::-1])
            final_ranks = np.argsort(np.argsort(season_result['final_points'])[::-1])
            skill_percentiles = (1 - skill_ranks / (num_players - 1)) * 100
            ranking_percentiles = (1 - final_ranks / (num_players - 1)) * 100
            all_skill_percentiles.extend(skill_percentiles)
            all_final_ranking_percentiles.extend(ranking_percentiles)
            skill_tiers = np.digitize(skill_percentiles, [25, 50, 75, 100]) - 1
            ranking_tiers = np.digitize(ranking_percentiles, [25, 50, 75, 100]) - 1
            for player in range(num_players):
                mobility_data.append({
                    'skill_tier': skill_tiers[player],
                    'ranking_tier': ranking_tiers[player],
                    'skill_percentile': skill_percentiles[player],
                    'ranking_percentile': ranking_percentiles[player]
                })
            
            # Calculate top skill identification rates
            top_n_values = [5, 10, 15, 20]
            top_skill_representation = {}
            for top_n in top_n_values:
                top_skill_players = set(true_skill_ranking[:top_n])
                top_ranked_players = set(final_ranking[:top_n])
                overlap = len(top_skill_players & top_ranked_players)
                top_skill_representation[f'top_{top_n}'] = overlap / top_n
            season_results.append({
                'simulation': sim,
                'correlation': correlation,
                'p_value': p_value,
                'top_skill_representation': top_skill_representation,
                'winner_skill_percentile': np.percentile(skills, 
                    (skills >= skills[season_result['season_data'][-1]['winner']]).mean() * 100),
                'mean_winner_skill': np.mean([t['winner_skill'] for t in season_result['season_data']])
            })
        
        correlations = [r['correlation'] for r in season_results]
        return {
            'skills': skills,
            'season_results': season_results,
            'true_skill_ranking': true_skill_ranking,
            'all_skill_percentiles': all_skill_percentiles,
            'all_final_ranking_percentiles': all_final_ranking_percentiles,
            'mobility_data': mobility_data,
            'summary_stats': {
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'correlations': correlations
            }
        }

    def compare_formats(self, num_simulations=150, num_players=32):
        skills = self.generate_player_skills(num_players)
        true_skill_ranking = np.argsort(skills)[::-1]
        
        scenarios = [
            {'name': 'Short_Best3', 'tournaments': 6, 'sets_to_win': 2, 'description': 'Short Season (6 tournaments), Best-of-3'},
            {'name': 'Long_Best3', 'tournaments': 18, 'sets_to_win': 2, 'description': 'Long Season (18 tournaments), Best-of-3'},
            {'name': 'Short_Best5', 'tournaments': 6, 'sets_to_win': 3, 'description': 'Short Season (6 tournaments), Best-of-5'},
            {'name': 'Long_Best5', 'tournaments': 18, 'sets_to_win': 3, 'description': 'Long Season (18 tournaments), Best-of-5'},
            {'name': 'Medium_Best3', 'tournaments': 12, 'sets_to_win': 2, 'description': 'Medium Season (12 tournaments), Best-of-3'},
            {'name': 'Medium_Best5', 'tournaments': 12, 'sets_to_win': 3, 'description': 'Medium Season (12 tournaments), Best-of-5'}
        ]
        scenario_results = {}
        
        for scenario in scenarios:
            scenario_data = []
            
            for sim in range(num_simulations):
                season_result = self.run_season(
                    skills, 
                    scenario['tournaments'], 
                    scenario['sets_to_win']
                )
                final_ranking = season_result['final_ranking']
                
                # Calculate skill revelation metrics
                ranking_scores = np.zeros(num_players)
                for i, player in enumerate(final_ranking):
                    ranking_scores[player] = num_players - i
                
                correlation, p_value = pearsonr(skills, ranking_scores)
                
                # Top skill identification rates
                top_rates = {}
                for top_n in [5, 8, 10]:
                    top_skill_players = set(true_skill_ranking[:top_n])
                    top_ranked_players = set(final_ranking[:top_n])
                    overlap = len(top_skill_players & top_ranked_players)
                    top_rates[f'top_{top_n}_rate'] = overlap / top_n
                
                winner_skills = [t['winner_skill'] for t in season_result['season_data']]
                
                scenario_data.append({
                    'simulation': sim,
                    'correlation': correlation,
                    'p_value': p_value,
                    'top_rates': top_rates,
                    'mean_winner_skill': np.mean(winner_skills),
                    'std_winner_skill': np.std(winner_skills),
                    'winner_skill_range': np.max(winner_skills) - np.min(winner_skills)
                })
            
            scenario_results[scenario['name']] = {
                'data': scenario_data,
                'scenario_info': scenario
            }
        
        return {
            'skills': skills,
            'scenario_results': scenario_results
        }

def run_analysis():
    analyzer = TennisAnalysis()
    results1 = analyzer.skill_ranking_analysis(num_simulations=200, num_players=64)
    results2 = analyzer.compare_formats(num_simulations=150, num_players=32)
    create_plots(results1, results2)
    return results1, results2

def create_plots(q1_results, q2_results):
    # Figure 1: Skill-Ranking Analysis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    skill_percentiles = np.array(q1_results['all_skill_percentiles'])
    ranking_percentiles = np.array(q1_results['all_final_ranking_percentiles'])
    
    # Scatter plot: Skill vs Final Ranking
    ax1.scatter(skill_percentiles, ranking_percentiles, alpha=0.3, s=10, color='blue')
    ax1.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Correlation')
    ax1.set_xlabel('True Skill Percentile')
    ax1.set_ylabel('Final Ranking Percentile')
    ax1.set_title('Skill vs Final Ranking')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    corr_coef, _ = pearsonr(skill_percentiles, ranking_percentiles)
    ax1.text(5, 95, f'r = {corr_coef:.3f}', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Success rates by skill tier
    skill_tiers = ['Top 10%', 'Top 25%', 'Top 50%', 'Bottom 50%']
    success_in_top10 = []
    success_in_top25 = []
    
    for tier_name in skill_tiers:
        if tier_name == 'Top 10%':
            mask = skill_percentiles >= 90
        elif tier_name == 'Top 25%':
            mask = skill_percentiles >= 75
        elif tier_name == 'Top 50%':
            mask = skill_percentiles >= 50
        else:
            mask = skill_percentiles < 50
        
        if np.sum(mask) > 0:
            tier_rankings = ranking_percentiles[mask]
            success_in_top10.append(np.mean(tier_rankings >= 90) * 100)
            success_in_top25.append(np.mean(tier_rankings >= 75) * 100)
        else:
            success_in_top10.append(0)
            success_in_top25.append(0)
    
    x = np.arange(len(skill_tiers))
    width = 0.35
    bars1 = ax2.bar(x - width/2, success_in_top10, width, label='Finish Top 10%', 
                    color='darkblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, success_in_top25, width, label='Finish Top 25%', 
                    color='lightblue', alpha=0.8)
    
    ax2.set_xlabel('True Skill Tier')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rates by Skill Level')
    ax2.set_xticks(x)
    ax2.set_xticklabels(skill_tiers)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Mobility heatmap  -- needs consultation on interpretation (useful for visuals)
    mobility_df = pd.DataFrame(q1_results['mobility_data'])
    tier_labels = ['Bottom 25%', 'Lower-Mid 25%', 'Upper-Mid 25%', 'Top 25%']
    mobility_matrix = np.zeros((4, 4))
    
    for skill_tier in range(4):
        tier_data = mobility_df[mobility_df['skill_tier'] == skill_tier]
        if len(tier_data) > 0:
            for ranking_tier in range(4):
                count = len(tier_data[tier_data['ranking_tier'] == ranking_tier])
                mobility_matrix[skill_tier, ranking_tier] = count / len(tier_data) * 100
    
    im = ax3.imshow(mobility_matrix, cmap='Blues', aspect='auto')
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    ax3.set_xticklabels(tier_labels, rotation=45, ha='right')
    ax3.set_yticklabels(tier_labels)
    ax3.set_xlabel('Final Ranking Tier')
    ax3.set_ylabel('True Skill Tier')
    ax3.set_title('Skill-to-Ranking Mobility Matrix (%)')
    
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f'{mobility_matrix[i, j]:.0f}%',
                    ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax3, label='Percentage (%)')
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Format Comparison Analysis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    scenario_names = []
    mean_correlations = []
    std_correlations = []
    colors = []
    scenarios_data = {}
    
    for scenario_key, scenario_data in q2_results['scenario_results'].items():
        correlations = [r['correlation'] for r in scenario_data['data']]
        scenarios_data[scenario_key] = correlations
        scenario_names.append(scenario_data['scenario_info']['description'].replace(' (', '\n('))
        mean_correlations.append(np.mean(correlations))
        std_correlations.append(np.std(correlations))
        colors.append('skyblue' if 'Best-of-3' in scenario_data['scenario_info']['description'] else 'lightcoral')
    
    # Format effectiveness comparison
    bars = ax1.bar(range(len(scenario_names)), mean_correlations, yerr=std_correlations, 
                   capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Tournament Format')
    ax1.set_ylabel('Mean Skill-Ranking Correlation')
    ax1.set_title('Format Effectiveness Comparison')
    ax1.set_xticks(range(len(scenario_names)))
    ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, mean_val) in enumerate(zip(bars, mean_correlations)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_correlations[i] + 0.005,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Season length vs match length comparison
    season_lengths = [6, 12, 18]
    best3_means = []
    best5_means = []
    
    for length in season_lengths:
        for scenario_key, scenario_data in q2_results['scenario_results'].items():
            if scenario_data['scenario_info']['tournaments'] == length:
                correlations = [r['correlation'] for r in scenario_data['data']]
                if 'Best-of-3' in scenario_data['scenario_info']['description']:
                    best3_means.append(np.mean(correlations))
                else:
                    best5_means.append(np.mean(correlations))
    
    x = np.arange(len(season_lengths))
    width = 0.35
    bars1 = ax2.bar(x - width/2, best3_means, width, label='Best-of-3', 
                    color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, best5_means, width, label='Best-of-5', 
                    color='lightcoral', alpha=0.8, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
    
    ax2.set_xlabel('Number of Tournaments')
    ax2.set_ylabel('Mean Correlation')
    ax2.set_title('Season Length vs Match Length')
    ax2.set_xticks(x)
    ax2.set_xticklabels(season_lengths)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Effect sizes for format changes
    season_effect_best3 = np.mean([r['correlation'] for r in q2_results['scenario_results']['Long_Best3']['data']]) - \
                         np.mean([r['correlation'] for r in q2_results['scenario_results']['Short_Best3']['data']])
    season_effect_best5 = np.mean([r['correlation'] for r in q2_results['scenario_results']['Long_Best5']['data']]) - \
                         np.mean([r['correlation'] for r in q2_results['scenario_results']['Short_Best5']['data']])
    match_effect_short = np.mean([r['correlation'] for r in q2_results['scenario_results']['Short_Best5']['data']]) - \
                        np.mean([r['correlation'] for r in q2_results['scenario_results']['Short_Best3']['data']])
    match_effect_long = np.mean([r['correlation'] for r in q2_results['scenario_results']['Long_Best5']['data']]) - \
                        np.mean([r['correlation'] for r in q2_results['scenario_results']['Long_Best3']['data']])
    
    effects = [season_effect_best3, season_effect_best5, match_effect_short, match_effect_long]
    effect_names = ['Season Length\n(Best-of-3)', 'Season Length\n(Best-of-5)', 
                   'Match Length\n(Short)', 'Match Length\n(Long)']
    
    colors_effect = ['green' if e > 0 else 'red' for e in effects]
    bars = ax3.bar(range(len(effect_names)), effects, color=colors_effect, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax3.set_xlabel('Format Comparison')
    ax3.set_ylabel('Effect Size (Correlation Difference)')
    ax3.set_title('Effect Sizes of Format Changes')
    ax3.set_xticks(range(len(effect_names)))
    ax3.set_xticklabels(effect_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, effect) in enumerate(zip(bars, effects)):
        ax3.text(bar.get_x() + bar.get_width()/2, effect + (0.002 if effect >= 0 else -0.002), 
                f'{effect:.4f}', ha='center', va='bottom' if effect >= 0 else 'top', 
                fontsize=10, weight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results1, results2 = run_analysis()