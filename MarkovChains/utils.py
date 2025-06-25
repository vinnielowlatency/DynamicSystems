import numpy as np
from scipy.stats import beta, ttest_ind, pearsonr
from config import Config

def generate_beta_skills(num_players, alpha=None, beta_param=None,  #generate beta skill distribution
                        skill_min=None, skill_max=None):
    alpha = alpha or Config.BETA_ALPHA
    beta_param = beta_param or Config.BETA_BETA
    skill_min = skill_min or Config.SKILL_MIN
    skill_max = skill_max or Config.SKILL_MAX
    
    skills = beta.rvs(alpha, beta_param, size=num_players)
    return np.clip(skills, skill_min, skill_max)

def skill_to_probability(skill_a, skill_b, sensitivity_k=None):

    k = sensitivity_k or Config.SENSITIVITY_K
    return 1 / (1 + np.exp(-k * (skill_a - skill_b)))

def calculate_correlation_with_ranking(skills, final_ranking):

    num_players = len(skills)
    ranking_scores = np.zeros(num_players)
    
    # Convert ranking to scores (higher = better)
    for i, player in enumerate(final_ranking):
        ranking_scores[player] = num_players - i
    
    return pearsonr(skills, ranking_scores)

def calculate_top_skill_representation(skills, final_ranking, top_n_values=None):
    """
    Calculate how many top-skilled players appear in top rankings
    
    Args:
        skills (np.ndarray): Player skill levels
        final_ranking (np.ndarray): Final ranking
        top_n_values (list): List of top-N values to check
        
    Returns:
        dict: Top skill representation rates
    """
    top_n_values = top_n_values or Config.TOP_N_VALUES
    true_skill_ranking = np.argsort(skills)[::-1]  # Best to worst by skill
    
    representation = {}
    for top_n in top_n_values:
        if top_n <= len(skills):
            top_skill_players = set(true_skill_ranking[:top_n])
            top_ranked_players = set(final_ranking[:top_n])
            overlap = len(top_skill_players & top_ranked_players)
            representation[f'top_{top_n}'] = overlap / top_n
    
    return representation

def perform_statistical_tests(scenario_results):
    """
    Perform statistical tests comparing different scenarios
    
    Args:
        scenario_results (dict): Results from different scenarios
        
    Returns:
        dict: Statistical test results
    """
    tests = {}
    
    # Extract correlation data
    short_best3 = [r['correlation'] for r in scenario_results['Short_Season_Best3']['data']]
    long_best3 = [r['correlation'] for r in scenario_results['Long_Season_Best3']['data']]
    short_best5 = [r['correlation'] for r in scenario_results['Short_Season_Best5']['data']]
    long_best5 = [r['correlation'] for r in scenario_results['Long_Season_Best5']['data']]
    
    # Season length effects
    tests['season_effect_best3'] = ttest_ind(short_best3, long_best3)
    tests['season_effect_best5'] = ttest_ind(short_best5, long_best5)
    
    # Match length effects
    tests['match_effect_short'] = ttest_ind(short_best3, short_best5)
    tests['match_effect_long'] = ttest_ind(long_best3, long_best5)
    
    return tests

def save_results(data, filename, results_dir=None):
    """
    Save results to file
    
    Args:
        data: Data to save
        filename (str): Output filename
        results_dir (str): Results directory
    """
    import os
    import pickle
    
    results_dir = results_dir or Config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Results saved to {filepath}")