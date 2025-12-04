"""
Test script to run multiple iterations of each solver and collect statistics.
"""
import argparse
import json
import time
from typing import List, Dict
from data.load_dataset import load_historical_data
from main import solve_puzzles


def run_multiple_iterations(solver_type: str, num_puzzles: int, num_iterations: int, 
                           max_mistakes: int = 7) -> Dict:
    """
    Run solver multiple times and aggregate statistics.
    
    Args:
        solver_type: Type of solver ('csp' or 'kmeans')
        num_puzzles: Number of puzzles to test per iteration
        num_iterations: Number of iterations to run
        max_mistakes: Maximum mistakes allowed
        
    Returns:
        Dictionary with aggregated statistics
    """
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"TESTING {solver_type.upper()} SOLVER")
    print(f"Puzzles per iteration: {num_puzzles}, Iterations: {num_iterations}, Mistakes allowed: {max_mistakes}")
    print(f"{'='*80}\n")
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*80}\n")
        
        # Load puzzles
        all_puzzles = load_historical_data()
        test_puzzles = all_puzzles[:num_puzzles]
        
        # Run solver
        iteration_results = solve_puzzles(
            test_puzzles, 
            max_mistakes=max_mistakes, 
            solver_type=solver_type
        )
        
        all_results.append(iteration_results)
    
    # Aggregate statistics across all iterations
    total_puzzles = num_puzzles * num_iterations
    all_wins = []
    all_correct = []
    all_mistakes = []
    all_times = []
    
    for iteration_results in all_results:
        for result in iteration_results:
            all_wins.append(1 if result['is_won'] else 0)
            all_correct.append(len(result['solved_groups']))
            all_mistakes.append(result['mistakes'])
            all_times.append(result.get('timing', {}).get('total', 0))
    
    # Calculate averages
    total_wins = sum(all_wins)
    avg_wins = total_wins / num_iterations
    avg_correct = sum(all_correct) / total_puzzles
    avg_mistakes = sum(all_mistakes) / total_puzzles
    avg_time = sum(all_times) / total_puzzles
    
    # Calculate standard deviations
    import statistics
    std_correct = statistics.stdev(all_correct) if len(all_correct) > 1 else 0
    std_mistakes = statistics.stdev(all_mistakes) if len(all_mistakes) > 1 else 0
    std_time = statistics.stdev(all_times) if len(all_times) > 1 else 0
    
    summary = {
        'solver_type': solver_type,
        'num_puzzles_per_iteration': num_puzzles,
        'num_iterations': num_iterations,
        'max_mistakes': max_mistakes,
        'total_puzzles_tested': total_puzzles,
        'statistics': {
            'wins': {
                'total': total_wins,
                'average_per_iteration': avg_wins,
                'win_rate': total_wins / total_puzzles
            },
            'correct_guesses': {
                'average': avg_correct,
                'std_dev': std_correct
            },
            'mistakes': {
                'average': avg_mistakes,
                'std_dev': std_mistakes
            },
            'time': {
                'average': avg_time,
                'std_dev': std_time
            }
        }
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Test solvers with multiple iterations")
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=100,
        help="Number of puzzles to test per iteration (default: 100)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of iterations to run (default: 5)"
    )
    parser.add_argument(
        "--mistakes-allowed",
        type=int,
        default=7,
        help="Maximum mistakes allowed (default: 7)"
    )
    parser.add_argument(
        "--solvers",
        type=str,
        nargs='+',
        default=['csp', 'kmeans'],
        help="Solvers to test (default: csp kmeans)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format)"
    )
    
    args = parser.parse_args()
    
    all_summaries = []
    
    for solver_type in args.solvers:
        try:
            summary = run_multiple_iterations(
                solver_type=solver_type,
                num_puzzles=args.num_puzzles,
                num_iterations=args.num_iterations,
                max_mistakes=args.mistakes_allowed
            )
            all_summaries.append(summary)
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"FINAL SUMMARY - {solver_type.upper()} SOLVER")
            print(f"{'='*80}")
            print(f"Total puzzles tested: {summary['total_puzzles_tested']}")
            print(f"Total wins: {summary['statistics']['wins']['total']}")
            print(f"Average wins per iteration: {summary['statistics']['wins']['average_per_iteration']:.2f}")
            print(f"Win rate: {summary['statistics']['wins']['win_rate']:.2%}")
            print(f"Average correct guesses per puzzle: {summary['statistics']['correct_guesses']['average']:.2f} (±{summary['statistics']['correct_guesses']['std_dev']:.2f})")
            print(f"Average mistakes per puzzle: {summary['statistics']['mistakes']['average']:.2f} (±{summary['statistics']['mistakes']['std_dev']:.2f})")
            print(f"Average time per puzzle: {summary['statistics']['time']['average']:.3f}s (±{summary['statistics']['time']['std_dev']:.3f}s)")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"Error testing {solver_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Results saved to {args.output}")
    
    # Print comparison table
    if len(all_summaries) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'Solver':<15} {'Wins/Iter':<12} {'Win Rate':<12} {'Avg Correct':<15} {'Avg Mistakes':<15} {'Avg Time (s)':<15}")
        print(f"{'-'*80}")
        for summary in all_summaries:
            stats = summary['statistics']
            print(f"{summary['solver_type']:<15} "
                  f"{stats['wins']['average_per_iteration']:<12.2f} "
                  f"{stats['wins']['win_rate']:<12.2%} "
                  f"{stats['correct_guesses']['average']:<15.2f} "
                  f"{stats['mistakes']['average']:<15.2f} "
                  f"{stats['time']['average']:<15.3f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

