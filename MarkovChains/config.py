class Config: # beta distribution parameters
    BETA_ALPHA = 2
    BETA_BETA = 5
    SKILL_MIN = 0.1
    SKILL_MAX = 0.9
    SENSITIVITY_K = 2.0
    Q1_SIMULATIONS = 200
    Q1_PLAYERS = 64
    Q1_TOURNAMENTS = 12
    Q2_SIMULATIONS = 150
    Q2_PLAYERS = 32
    
    # tournament scenarios
    SCENARIOS = [
        {'name': 'Short_Season_Best3', 'tournaments': 6, 'sets_to_win': 2, 
         'description': 'Short Season (6 tournaments), Best-of-3'},
        {'name': 'Long_Season_Best3', 'tournaments': 18, 'sets_to_win': 2, 
         'description': 'Long Season (18 tournaments), Best-of-3'},
        {'name': 'Short_Season_Best5', 'tournaments': 6, 'sets_to_win': 3, 
         'description': 'Short Season (6 tournaments), Best-of-5'},
        {'name': 'Long_Season_Best5', 'tournaments': 18, 'sets_to_win': 3, 
         'description': 'Long Season (18 tournaments), Best-of-5'},
        {'name': 'Medium_Season_Best3', 'tournaments': 12, 'sets_to_win': 2, 
         'description': 'Medium Season (12 tournaments), Best-of-3'},
        {'name': 'Medium_Season_Best5', 'tournaments': 12, 'sets_to_win': 3, 
         'description': 'Medium Season (12 tournaments), Best-of-5'}
    ]
    
    # analysis params
    TOP_N_VALUES = [5, 8, 10, 15, 20]
    SIGNIFICANCE_LEVEL = 0.05
    
    # tbl setting
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
    RESULTS_DIR = 'results'