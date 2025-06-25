import os
import sys
from datetime import datetime

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from analysis import ComprehensiveTennisAnalysis
from visualization import create_all_figures, save_figures
from utils import save_results

def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("COMPREHENSIVE TENNIS MARKOV CHAIN ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Question 1: {Config.Q1_SIMULATIONS} simulations, {Config.Q1_PLAYERS} players")
    print(f"  Question 2: {Config.Q2_SIMULATIONS} simulations per scenario, {Config.Q2_PLAYERS} players")
    print(f"  Total scenarios: {len(Config.SCENARIOS)}")
    print(f"  Estimated runtime: 5-10 minutes\n")
    
    # Initialize analyzer
    analyzer = ComprehensiveTennisAnalysis()
    
    # Question 1 Analysis
    print("PHASE 1: Analyzing Question 1 (Skill Distribution Effects)")
    print("-" * 60)
    q1_results = analyzer.analyze_question_1_comprehensive(
        num_simulations=Config.Q1_SIMULATIONS,
        num_players=Config.Q1_PLAYERS
    )
    
    # Question 2 Analysis
    print("\nPHASE 2: Analyzing Question 2 (Format Effectiveness)")
    print("-" * 60)
    q2_results = analyzer.analyze_question_2_comprehensive(
        num_simulations=Config.Q2_SIMULATIONS,
        num_players=Config.Q2_PLAYERS
    )
    
    # Generate all figures
    print("\nPHASE 3: Generating Figures and Reports")
    print("-" * 60)
    figures = create_all_figures(q1_results, q2_results)
    save_figures(figures)
    
    # Save results
    save_results(q1_results, 'question1_results.pkl')
    save_results(q2_results, 'question2_results.pkl')
    
    # Generate final report
    analyzer.generate_final_report(q1_results, q2_results)
    
    print(f"\nAnalysis complete! Check the '{Config.RESULTS_DIR}' directory for:")
    print("  - Detailed figures (PNG format)")
    print("  - Raw data files (PKL format)")
    print("  - Summary report (TXT format)")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()