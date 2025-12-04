"""
Main solver orchestration for game mode.
"""
import argparse
import time
from typing import List, Dict
from solvers.k_means_solver import KMeansConnectionsSolver
from data.load_dataset import load_historical_data, Puzzle
from similarity.embedding_similarity import EmbeddingSimilarity
from evaluation.game_simulator import GameSimulator
from solvers.csp_solver import CSPSolver
from features.word_embeddings import WordEmbeddings
import config


def solve_with_csp(puzzle: Puzzle, similarity_fn: EmbeddingSimilarity, max_mistakes: int = 4) -> Dict:
    """
    Solve puzzle using CSP (Constraint Satisfaction Problem) approach.
    
    Args:
        puzzle: Puzzle to solve
        similarity_fn: Pre-initialized similarity function
        max_mistakes: Maximum number of mistakes allowed
        
    Returns:
        Dictionary with metrics on the model's performance
    """
    start_time = time.time()
    
    # game simulator class
    game = GameSimulator(puzzle, max_mistakes=max_mistakes)
    
    # CSP solver class
    solver = CSPSolver(similarity_fn)
    
    print(f"\npuzzle id: {puzzle.puzzle_id}")
    
    # solve the puzzle using the CSPSolver class
    result = solver.solve_with_feedback(game)
    
    total_time = time.time() - start_time
    
    # display results
    num_correct = len(result['solved_groups'])
    print(f"Correct guesses: {num_correct}")
    
    result['timing'] = {'total': total_time}
    
    return result

def solve_with_kmeans(puzzle: Puzzle, kmeans_solver: KMeansConnectionsSolver, 
                      max_mistakes: int = 4) -> Dict:
    """
    Solve puzzle using adaptive K-Means clustering approach.
    
    Args:
        puzzle: Puzzle to solve
        kmeans_solver: Pre-initialized K-Means solver
        max_mistakes: Maximum number of mistakes allowed
        
    Returns:
        Dictionary with metrics on the model's performance
    """
    # game simulator class
    game = GameSimulator(puzzle, max_mistakes=max_mistakes)
    
    print(f"\npuzzle id: {puzzle.puzzle_id}")
    
    # Use adaptive K-Means that re-clusters based on feedback
    result = kmeans_solver.solve_with_feedback(game)
    
    # Display results
    num_correct = len(result['solved_groups'])
    print(f"Correct guesses: {num_correct}")
    
    return result


def solve_puzzles(test_puzzles: List[Puzzle], max_mistakes: int = 4, 
                 solver_type: str = "csp"):
    """
    Solve a list of puzzles
    
    Args:
        test_puzzles: List of puzzles to solve
        max_mistakes: Maximum number of mistakes allowed per puzzle (default: 4)
        solver_type: Type of solver to use ("kmeans" or "csp")
    """
    results = []
    
    # Initialize solver based on type
    if solver_type == "kmeans":
        embeddings_fn = WordEmbeddings(config.EMBEDDING_MODEL)
        
        kmeans_solver = KMeansConnectionsSolver(embeddings_fn)
        
        print(f"\n{'='*70}")
        print("USING K-MEANS CLUSTERING SOLVER")
        print(f"{'='*70}\n")
        
        for puzzle in test_puzzles:
            result = solve_with_kmeans(puzzle, kmeans_solver, max_mistakes=max_mistakes)
            results.append(result)
    
    elif solver_type == "csp":
        similarity_fn = EmbeddingSimilarity()
        print(f"\n{'='*70}")
        print("USING CSP (CONSTRAINT SATISFACTION PROBLEM) SOLVER")
        print(f"{'='*70}\n")
        
        for puzzle in test_puzzles:
            result = solve_with_csp(puzzle, similarity_fn, max_mistakes=max_mistakes)
            results.append(result)
    
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. Choose from: 'kmeans', 'csp'")
    
    # Aggregate statistics
    total_puzzles = len(results)
    wins = sum(1 for r in results if r['is_won'])
    avg_correct = sum(len(r['solved_groups']) for r in results) / total_puzzles
    avg_submissions = sum(r['total_submissions'] for r in results) / total_puzzles
    avg_mistakes = sum(r['mistakes'] for r in results) / total_puzzles
    
    # Timing statistics
    avg_total_time = sum(r.get('timing', {}).get('total', 0) for r in results) / total_puzzles
    
    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS - {solver_type.upper()} SOLVER")
    print(f"{'='*70}")
    print(f"Total puzzles: {total_puzzles}")
    print(f"Wins: {wins} ({wins/total_puzzles:.1%})")
    print(f"Average correct guesses per puzzle: {avg_correct:.2f}")
    print(f"Average submissions per puzzle: {avg_submissions:.2f}")
    print(f"Average mistakes per puzzle: {avg_mistakes:.2f}")
    print(f"Average time per puzzle: {avg_total_time:.2f}s")
    print(f"{'='*70}\n")
    return results


def main():
    """let the agent play the game!"""
    parser = argparse.ArgumentParser(description="NYT Connections Solver Agent - Game Mode")
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=None,
        help="Number of puzzles to solve (default: all)"
    )
    parser.add_argument(
        "--mistakes-allowed",
        type=int,
        default=4,
        help="Maximum number of mistakes allowed in game mode (default: 4)"
    )
    parser.add_argument(
        "--solver-type",
        type=str,
        default="csp",
        help="Type of solver to use: 'kmeans' or 'csp' (default: csp)"
    )
    
    args = parser.parse_args()
    
    # Load puzzles
    all_puzzles = load_historical_data()
    test_puzzles = all_puzzles if args.num_puzzles is None else all_puzzles[:args.num_puzzles]
    
    # solve the test puzzles
    solve_puzzles(test_puzzles, max_mistakes=args.mistakes_allowed, solver_type=args.solver_type)


if __name__ == "__main__":
    main()
