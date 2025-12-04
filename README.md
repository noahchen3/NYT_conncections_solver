# NYT Connections Solver

An AI solver for the New York Times Connections puzzle game using multiple algorithmic approaches.

## About NYT Connections

NYT Connections is a word puzzle game where players are given 16 words and must group them into 4 categories of 4 words each. Each category has a common theme or connection. The challenge is to identify these connections and correctly group the words.

**Game Rules:**

- You have 16 words to organize into 4 groups of 4
- Each group shares a common theme (e.g., "Types of fruit", "Words that mean 'fast'", etc.)
- You can submit groups of 4 words to check if they're correct
- You get feedback: correct (all 4 words match a category), partial match (2-3 words are correct), or wrong (0-1 words correct)
- You have 4 mistakes allowed (configurable)
- The game ends when you find all 4 groups or run out of mistakes

## How This Project Works

This project implements AI solvers that automatically play the Connections game by:

1. **Loading historical puzzles** from a dataset of real NYT Connections puzzles
2. **Using word embeddings** (GloVe) to compute semantic similarity between words
3. **Applying different algorithms** to group words based on similarity and learned constraints
4. **Learning from feedback** - solvers adapt their strategies based on correct/wrong/partial matches
5. **Simulating the game** - provides the same feedback mechanism as the real game

### Solvers Implemented

**1. CSP Solver (Constraint Satisfaction Problem)**

- Uses weighted constraints to track which word pairs should/shouldn't be together
- Learns from feedback: updates constraint weights based on correct/wrong/partial matches
- Uses random sampling for exploration to find optimal groups
- Similarity-guided initialization helps bootstrap the constraint system

**2. K-Means Solver**

- Uses adaptive K-Means clustering on word embeddings
- Re-clusters remaining words after each submission based on feedback
- Updates similarity weights dynamically as it learns from game feedback
- Uses deterministic K-Means++ initialization for reproducibility

Both solvers use the same infrastructure (word embeddings, game simulator, similarity functions) but employ fundamentally different algorithmic approaches.

## Project Disclaimer

**Note:** This project uses infrastructure and architecture from a group project (data loading, game simulation, evaluation framework). However, the solver implementations (`solvers/csp_solver.py` and `solvers/k_means_solver.py`) are my own original work. The original repository can be found at https://github.com/derrickyg/CS4100-NYT-Connections-Project. The solvers use:

- Weighted constraints with similarity-guided initialization (CSP)
- Adaptive re-clustering with feedback-weighted similarity (K-Means)
- Random sampling for exploration (CSP)
- Deterministic initialization (K-Means)

## Installation

install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

run program:

```bash
python main.py --mistakes-allowed 4 --num-puzzles 10
```

^ where

`--mistakes-allowed` is the number of mistakes allowed in the game (default is 4, just like the real nyt connections game)

`--num-puzzles` is the number of puzzles to evaluate on (default is all ~652 puzzles in the dataset)

`--solver-type` is the type of solver to use: `'csp'` or `'kmeans'` (default: `'csp'`)

### Example Commands

```bash
# Test CSP solver on 10 puzzles with 4 mistakes
python main.py --solver-type csp --num-puzzles 10 --mistakes-allowed 4

# Test K-Means solver on 100 puzzles with 7 mistakes
python main.py --solver-type kmeans --num-puzzles 100 --mistakes-allowed 7

# Run comprehensive evaluation (5 iterations of 100 puzzles)
python test_solvers.py --num-puzzles 100 --num-iterations 5 --mistakes-allowed 7 --solvers csp kmeans
```

## Project Structure

```
├── solvers/              # Solver implementations
│   ├── csp_solver.py     # CSP solver with weighted constraints
│   └── k_means_solver.py # Adaptive K-Means solver
├── data/                 # Dataset loading
├── evaluation/           # Game simulation and feedback
├── features/            # Word embeddings
├── similarity/          # Similarity computation
├── main.py             # Main orchestration script
├── test_solvers.py     # Evaluation script
└── config.py           # Configuration
```

## Results

On a test set of 500 puzzles (5 iterations × 100 puzzles) with 7 mistakes allowed:

- **CSP Solver**: ~13% win rate, 0.85 average correct guesses per puzzle
- **K-Means Solver**: ~3.8% win rate, 0.71 average correct guesses per puzzle
