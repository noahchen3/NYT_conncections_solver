"""
Game simulator for NYT Connections that provides feedback like the real game.
"""
from typing import List, Dict, Optional
from data.load_dataset import Puzzle


class GameFeedback:
    """Represents feedback from a game submission."""
    
    def __init__(self, is_correct: bool, correct_words: int = 0, group_id: Optional[int] = None):
        """
        Initialize feedback.
        
        Args:
            is_correct: True if the group is correct
            correct_words: Number of words that are correct (0-4)
            group_id: Which group this matches (if correct)
        """
        self.is_correct = is_correct
        self.correct_words = correct_words
        self.group_id = group_id


class GameSimulator:
    """Simulates the Connections game with feedback."""
    
    def __init__(self, puzzle: Puzzle, max_mistakes: int = 4):
        """
        Initialize game simulator.
        
        Args:
            puzzle: Puzzle with ground truth groups
            max_mistakes: Maximum number of mistakes allowed before losing
        """
        self.puzzle = puzzle
        self.solved_groups: Dict[int, List[str]] = {}
        self.remaining_words = set(puzzle.words)
        self.mistakes = 0
        self.max_mistakes = max_mistakes
        self.submission_count = 0
        self.is_game_over = False
        self.is_won = False
        
        # Convert ground truth groups to sets for easier comparison
        self.truth_groups = {
            group_id: set(w.upper() for w in words)
            for group_id, words in puzzle.groups.items()
        }
    
    def submit_group(self, words: List[str]) -> GameFeedback:
        """
        Submit a group of 4 words and get feedback.
        
        Args:
            words: List of 4 words to submit
            
        Returns:
            GameFeedback object with feedback
        """
        if self.is_game_over:
            raise ValueError("Game is over!")
        
        if len(words) != 4:
            raise ValueError(f"Must submit exactly 4 words, got {len(words)}")
        
        self.submission_count += 1
        words_set = set(w.upper() for w in words)
        
        # Check if this matches any unsolved group exactly
        for group_id, truth_set in self.truth_groups.items():
            if group_id in self.solved_groups:
                continue  # Already solved
            
            if words_set == truth_set:
                # Correct! Group solved
                self.solved_groups[group_id] = list(words)
                self.remaining_words -= words_set
                
                # Check if game is won
                if len(self.solved_groups) == 4:
                    self.is_won = True
                    self.is_game_over = True
                
                return GameFeedback(is_correct=True, correct_words=4, group_id=group_id)
        
        # Not correct - count how many words match each unsolved group
        max_correct = 0
        
        for group_id, truth_set in self.truth_groups.items():
            if group_id in self.solved_groups:
                continue
            
            correct_count = len(words_set & truth_set)
            if correct_count > max_correct:
                max_correct = correct_count
        
        # Count mistake
        self.mistakes += 1
        
        # Check if game is lost
        if self.mistakes >= self.max_mistakes:
            self.is_game_over = True
        
        return GameFeedback(is_correct=False, correct_words=max_correct)
    
    def get_state(self) -> Dict:
        """
        Get current game state.
        
        Returns:
            Dictionary with game state information
        """
        return {
            "solved_groups": len(self.solved_groups),
            "remaining_words": len(self.remaining_words),
            "mistakes": self.mistakes,
            "mistakes_remaining": self.max_mistakes - self.mistakes,
            "submission_count": self.submission_count,
            "is_game_over": self.is_game_over,
            "is_won": self.is_won
        }
    
    def get_remaining_words(self) -> List[str]:
        """Get list of remaining unsolved words."""
        return list(self.remaining_words)
    
    def get_solved_groups(self) -> Dict[int, List[str]]:
        """Get dictionary of solved groups."""
        return self.solved_groups.copy()

