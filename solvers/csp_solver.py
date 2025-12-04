"""
CSP solver using weighted constraints and random sampling.
Alternative implementation with similarity-guided initialization and weighted constraint learning.
"""
from typing import List, Dict, Optional, Tuple
from itertools import combinations
import random
from similarity.embedding_similarity import EmbeddingSimilarity
from evaluation.game_simulator import GameSimulator, GameFeedback


class CSP:
    """Represents a Constraint Satisfaction Problem with weighted constraints."""
    
    def __init__(self, words: List[str]):
        """
        Initialize CSP for Connections puzzle.
        
        Args:
            words: List of words to assign to groups
        """
        self.variables = [w.upper() for w in words]  # Each word is a variable
        # Weighted constraints: pair -> weight (0.0 = forbidden, 1.0 = must be together)
        # Weight represents confidence that words belong together
        self.pair_weights: Dict[Tuple[str, str], float] = {}


class CSPSolver:
    """CSP solver with weighted constraints and random sampling."""
    
    def __init__(self, similarity_function: EmbeddingSimilarity):
        """
        Initialize CSP solver.
        
        Args:
            similarity_function: Function to compute word similarity
        """
        self.similarity_fn = similarity_function
        self.csp: Optional[CSP] = None
        self.submission_history: List[Dict] = []
        # Track groups we've already tried (to avoid repeats)
        self.tried_groups: set = set()
    
    def solve_with_feedback(self, game: GameSimulator) -> Dict:
        """
        Solve puzzle using weighted CSP constraint-guided search with feedback learning.
        
        Args:
            game: GameSimulator instance
            
        Returns:
            Dictionary with solution and statistics
        """
        submissions = []
        
        # Initialize CSP with remaining words
        remaining_words = game.get_remaining_words()
        if not remaining_words:
            return {
                "solved_groups": game.get_solved_groups(),
                "submissions": [],
                "total_submissions": 0,
                "mistakes": 0,
                "is_won": True
            }
        
        self.csp = CSP(remaining_words)
        self.submission_history.clear()
        self.tried_groups.clear()
        
        # Initialize with similarity-guided constraints
        self._initialize_similarity_constraints(remaining_words)
        
        while not game.is_game_over:
            remaining = game.get_remaining_words()
            if len(remaining) == 0:
                break
            
            # Update CSP with current remaining words
            remaining_upper = set(w.upper() for w in remaining)
            if self.csp is None or remaining_upper != set(self.csp.variables):
                # Preserve learned constraints before creating new CSP
                old_pair_weights = self.csp.pair_weights.copy() if self.csp else {}
                
                # Create new CSP
                self.csp = CSP(remaining)
                
                # Preserve learned constraints (only pairs where both words are still remaining)
                self.csp.pair_weights = {
                    (w1, w2): weight for (w1, w2), weight in old_pair_weights.items()
                    if w1 in remaining_upper and w2 in remaining_upper
                }
                
                # Re-initialize similarity constraints for remaining words
                self._initialize_similarity_constraints(remaining)
            
            # Strategy 1: Handle partial matches (3/4 or 2/4 correct)
            group = self._handle_partial_matches(remaining)
            
            # Strategy 2: Find next group using weighted CSP constraints
            if not group:
                group = self._find_next_group_weighted(remaining)
            
            # Strategy 3: Fallback to similarity-based selection
            if not group or len(group) != 4:
                group = self._get_best_guess_deterministic(remaining)
            
            # Ensure we haven't tried this group before
            group_tuple = self._normalize_group(group)
            max_attempts = 50
            attempts = 0
            while group_tuple in self.tried_groups and attempts < max_attempts:
                group = self._get_best_guess_deterministic(remaining)
                if group:
                    group_tuple = self._normalize_group(group)
                attempts += 1
            
            if group:
                self.tried_groups.add(group_tuple)
            
            # Submit group
            submission_num = len(submissions) + 1
            feedback = game.submit_group(group)
            submission_data = {
                "group": group,
                "feedback": feedback
            }
            submissions.append(submission_data)
            self.submission_history.append(submission_data)
            
            # Learn from feedback (update weights)
            self._learn_from_feedback(group, feedback)
            
            # Print correct guesses
            if feedback.is_correct:
                print(f"Correct guess #{submission_num}: {', '.join(group)}")
        
        # Get final state
        state = game.get_state()
        
        return {
            "solved_groups": game.get_solved_groups(),
            "submissions": submissions,
            "total_submissions": len(submissions),
            "mistakes": state["mistakes"],
            "is_won": state["is_won"]
        }
    
    def _normalize_group(self, group: List[str]) -> tuple:
        """Normalize group for comparison."""
        return tuple(sorted(w.upper() for w in group))
    
    def _initialize_similarity_constraints(self, words: List[str]):
        """
        Initialize weighted constraints based on similarity scores.
        High similarity pairs get initial soft constraints.
        
        Args:
            words: List of words to analyze
        """
        if len(words) < 2:
            return
        
        # Compute similarity for all pairs
        pair_similarities = []
        for w1, w2 in combinations(words, 2):
            sim = self.similarity_fn.similarity(w1, w2)
            pair_similarities.append(((w1.upper(), w2.upper()), sim))
        
        # Sort by similarity
        pair_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Assign initial weights based on similarity percentile
        # Top 20% get soft "known" constraint (weight 0.5-0.6)
        # Bottom 20% get soft "forbidden" constraint (weight 0.2-0.3)
        # Middle 60% get neutral weights (0.4-0.5)
        total_pairs = len(pair_similarities)
        top_threshold = int(total_pairs * 0.2)
        bottom_threshold = int(total_pairs * 0.8)
        
        for idx, ((w1, w2), sim) in enumerate(pair_similarities):
            if idx < top_threshold:
                # Top 20%: high similarity, soft known constraint
                weight = 0.5 + (sim * 0.1)  # 0.5 to 0.6
            elif idx >= bottom_threshold:
                # Bottom 20%: low similarity, soft forbidden constraint
                weight = 0.2 + (sim * 0.1)  # 0.2 to 0.3
            else:
                # Middle 60%: neutral
                weight = 0.4 + (sim * 0.1)  # 0.4 to 0.5
            
            # Only set if not already learned from feedback
            pair_key = (w1, w2) if w1 < w2 else (w2, w1)
            if pair_key not in self.csp.pair_weights:
                self.csp.pair_weights[pair_key] = weight
    
    def _handle_partial_matches(self, remaining: List[str]) -> Optional[List[str]]:
        """
        Handle partial matches (3/4 or 2/4 correct) - refine groups based on feedback.
        
        Args:
            remaining: Remaining words
            
        Returns:
            Refined group or None
        """
        if len(remaining) < 4:
            return None
        
        # Check recent submissions for partial matches
        for correct_count in [3, 2]:
            history_size = 10 if correct_count == 3 else 5
            for submission in reversed(self.submission_history[-history_size:]):
                if (not submission['feedback'].is_correct and 
                    submission['feedback'].correct_words == correct_count):
                    if correct_count == 3:
                        group = self._refine_partial_match(submission['group'], remaining)
                    else:
                        group = self._refine_two_correct(submission['group'], remaining)
                    
                    if group and self._normalize_group(group) not in self.tried_groups:
                        return group
        
        return None
    
    def _refine_partial_match(self, previous_group: List[str], remaining_words: List[str]) -> Optional[List[str]]:
        """Refine a group that had 3/4 correct by swapping the wrong word."""
        remaining_set = set(w.upper() for w in remaining_words)
        available_from_prev = [w for w in previous_group if w.upper() in remaining_set]
        
        if len(available_from_prev) < 3:
            return None
        
        # Try all combinations of 3 words from previous group
        best_group = None
        best_score = float('-inf')
        
        for three_words in combinations(available_from_prev, 3):
            three_words_set = set(w.upper() for w in three_words)
            candidates = [w for w in remaining_words if w.upper() not in three_words_set]
            
            # Find best completion for this combination
            group = self._find_best_completion_weighted(list(three_words), candidates)
            if group and self._normalize_group(group) not in self.tried_groups:
                # Score this group using weighted constraints
                score = self._score_group_weighted(group)
                if score > best_score:
                    best_score = score
                    best_group = group
        
        return best_group
    
    def _refine_two_correct(self, previous_group: List[str], remaining_words: List[str]) -> Optional[List[str]]:
        """When we got 2/4 correct, try different combinations of those 2 words."""
        remaining_set = set(w.upper() for w in remaining_words)
        available_from_prev = [w for w in previous_group if w.upper() in remaining_set]
        
        if len(available_from_prev) < 2:
            return None
        
        # Try all pairs from the previous group
        for pair in combinations(available_from_prev, 2):
            pair_set = set(w.upper() for w in pair)
            candidates = [w for w in remaining_words if w.upper() not in pair_set]
            
            if len(candidates) < 2:
                continue
            
            # Find best combination of 2 more words
            best_group = self._find_best_completion_weighted(list(pair), candidates)
            if best_group and self._normalize_group(best_group) not in self.tried_groups:
                return best_group
        
        return None
    
    def _find_next_group_weighted(self, remaining: List[str]) -> Optional[List[str]]:
        """
        Find next group using weighted CSP constraints.
        
        Args:
            remaining: Remaining words to choose from
            
        Returns:
            Group of 4 words or None
        """
        if len(remaining) < 4:
            return remaining if remaining else None
        
        # Find pairs with highest weights and try to build groups from them
        if not self.csp.pair_weights:
            return None
        
        # Sort pairs by weight (highest first)
        sorted_pairs = sorted(
            self.csp.pair_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Try building groups from high-weight pairs
        for (w1, w2), weight in sorted_pairs[:10]:  # Try top 10 pairs
            if w1 not in {w.upper() for w in remaining} or w2 not in {w.upper() for w in remaining}:
                continue
            
            base = [w1, w2]
            candidates = [w for w in remaining if w.upper() not in {w1, w2}]
            
            # Find best completion using weighted constraints
            best_completion = self._find_best_completion_weighted(base, candidates)
            if best_completion and self._normalize_group(best_completion) not in self.tried_groups:
                return best_completion
        
        return None
    
    def _find_best_completion_weighted(self, base: List[str], candidates: List[str]) -> Optional[List[str]]:
        """Find best completion for a base group using weighted constraints and similarity."""
        if len(base) >= 4:
            return base[:4]
        
        needed = 4 - len(base)
        if len(candidates) < needed:
            return None
        
        base_set = set(w.upper() for w in base)
        
        # Score candidates by weighted constraint satisfaction and similarity
        scored = []
        for candidate in candidates:
            candidate_upper = candidate.upper()
            
            # Compute weighted constraint score
            constraint_score = 0.0
            for base_word in base:
                pair_key = (candidate_upper, base_word) if candidate_upper < base_word else (base_word, candidate_upper)
                weight = self.csp.pair_weights.get(pair_key, 0.5)  # Default to neutral
                constraint_score += weight
            
            # Average similarity to base words
            avg_sim = sum(
                self.similarity_fn.similarity(candidate, base_word)
                for base_word in base
            ) / len(base) if base else 0.0
            
            # Combined score (weighted average: 60% constraints, 40% similarity)
            combined_score = 0.6 * (constraint_score / len(base)) + 0.4 * avg_sim
            scored.append((candidate, combined_score))
        
        if len(scored) < needed:
            return None
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return base + [w for w, _ in scored[:needed]]
    
    def _get_best_guess_deterministic(self, remaining: List[str]) -> List[str]:
        """
        Get best guess using random sampling for exploration.
        Scores sampled combinations and picks the best.
        
        Args:
            remaining: Remaining words to choose from
            
        Returns:
            Best group of 4 words
        """
        if len(remaining) < 4:
            return remaining
        
        # Generate all combinations
        all_combos = list(combinations(remaining, 4))
        max_combos = min(200, len(all_combos))  # Sample up to 200 combinations
        
        # Prioritize combinations with high-weight pairs
        combos_with_high_weights = []
        combos_without_high_weights = []
        
        # Threshold for "high weight" (pairs that are likely to be together)
        high_weight_threshold = 0.7
        
        for combo in all_combos:
            combo_set = set(w.upper() for w in combo)
            has_high_weight_pair = False
            
            # Check if combo has any high-weight pairs
            for w1, w2 in combinations(combo, 2):
                pair_key = (w1.upper(), w2.upper()) if w1.upper() < w2.upper() else (w2.upper(), w1.upper())
                weight = self.csp.pair_weights.get(pair_key, 0.5)
                if weight >= high_weight_threshold:
                    has_high_weight_pair = True
                    break
            
            if has_high_weight_pair:
                combos_with_high_weights.append(combo)
            else:
                combos_without_high_weights.append(combo)
        
        # Sample more from combos with high-weight pairs
        half_max = max_combos // 2
        prioritized = []
        
        if combos_with_high_weights:
            sample_size = min(half_max, len(combos_with_high_weights))
            prioritized.extend(random.sample(combos_with_high_weights, sample_size))
        
        # Fill remaining slots from combos without high-weight pairs
        remaining_slots = max_combos - len(prioritized)
        if remaining_slots > 0 and combos_without_high_weights:
            sample_size = min(remaining_slots, len(combos_without_high_weights))
            prioritized.extend(random.sample(combos_without_high_weights, sample_size))
        
        best_group = None
        best_score = float('-inf')
        
        for combo in prioritized:
            # Skip if already tried
            if self._normalize_group(list(combo)) in self.tried_groups:
                continue
            
            # Score this group
            score = self._score_group_weighted(list(combo))
            
            if score > best_score:
                best_score = score
                best_group = list(combo)
        
        # Fallback if no constrained group found
        if best_group is None:
            return self._get_best_guess_similarity(remaining)
        
        return best_group
    
    def _score_group_weighted(self, group: List[str]) -> float:
        """
        Score a group using weighted constraints and similarity.
        
        Args:
            group: Group of 4 words to score
            
        Returns:
            Score (higher is better)
        """
        if len(group) != 4:
            return float('-inf')
        
        group_upper = [w.upper() for w in group]
        group_set = set(group_upper)
        
        # Compute weighted constraint satisfaction
        constraint_score = 0.0
        pair_count = 0
        for w1, w2 in combinations(group_upper, 2):
            pair_key = (w1, w2) if w1 < w2 else (w2, w1)
            weight = self.csp.pair_weights.get(pair_key, 0.5)  # Default to neutral
            constraint_score += weight
            pair_count += 1
        
        avg_constraint = constraint_score / pair_count if pair_count > 0 else 0.5
        
        # Compute similarity score
        similarity_score = sum(
            self.similarity_fn.similarity(w1, w2)
            for w1, w2 in combinations(group, 2)
        ) / pair_count if pair_count > 0 else 0.0
        
        # Combined score (weighted average: 60% constraints, 40% similarity)
        combined_score = 0.6 * avg_constraint + 0.4 * similarity_score
        
        return combined_score
    
    def _get_best_guess_similarity(self, remaining: List[str]) -> List[str]:
        """Get best guess using similarity when weighted CSP fails."""
        if len(remaining) < 4:
            return remaining
        
        # Find group with highest pairwise similarity
        best_group = None
        best_score = float('-inf')
        
        max_combos = 100
        checked = 0
        
        for combo in combinations(remaining, 4):
            checked += 1
            if checked > max_combos:
                break
            
            score = sum(
                self.similarity_fn.similarity(w1, w2)
                for w1, w2 in combinations(combo, 2)
            )
            if score > best_score:
                best_score = score
                best_group = list(combo)
        
        if best_group is None:
            # Last resort: return first 4 words
            return remaining[:4]
        
        return best_group
    
    def _learn_from_feedback(self, group: List[str], feedback: GameFeedback):
        """
        Learn weighted constraints from feedback.
        
        Args:
            group: Submitted group
            feedback: Feedback received
        """
        if self.csp is None:
            return
        
        group_upper = [w.upper() for w in group]
        all_pairs = list(combinations(group_upper, 2))
        
        if feedback.is_correct:
            # All words in this group belong together - set weights to 1.0
            for w1, w2 in all_pairs:
                pair_key = (w1, w2) if w1 < w2 else (w2, w1)
                self.csp.pair_weights[pair_key] = 1.0
        elif feedback.correct_words == 0:
            # None of these words belong together - set weights to 0.0
            for w1, w2 in all_pairs:
                pair_key = (w1, w2) if w1 < w2 else (w2, w1)
                self.csp.pair_weights[pair_key] = 0.0
        elif feedback.correct_words >= 2:
            # Some words belong together - update weights based on similarity
            self._learn_from_partial_feedback_weighted(group_upper, all_pairs, feedback.correct_words)
    
    def _learn_from_partial_feedback_weighted(self, group_upper: List[str], all_pairs: List[Tuple[str, str]], correct_words: int):
        """
        Learn weighted constraints from partial feedback (2/4 or 3/4 correct).
        Uses similarity scores to assign confidence weights.
        
        Args:
            group_upper: Words in the group (uppercase)
            all_pairs: All pairs in the group
            correct_words: Number of correct words (2 or 3)
        """
        # Score all pairs by similarity
        pair_scores = [
            (pair, self.similarity_fn.similarity(pair[0], pair[1]))
            for pair in all_pairs
        ]
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Number of pairs we expect: C(correct_words, 2)
        num_correct_pairs = correct_words * (correct_words - 1) // 2
        
        # Assign weights based on similarity ranking
        for idx, (pair, sim) in enumerate(pair_scores):
            pair_key = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
            
            if idx < num_correct_pairs:
                # Top pairs (likely correct): high weight based on similarity
                # Map similarity [0, 1] to weight [0.7, 0.95]
                weight = 0.7 + (sim * 0.25)
            else:
                # Bottom pairs (likely incorrect): low weight based on similarity
                # Map similarity [0, 1] to weight [0.05, 0.3]
                weight = 0.05 + (sim * 0.25)
            
            # Update weight (average with existing if present, to avoid overwriting strong signals)
            if pair_key in self.csp.pair_weights:
                # If we already have a strong signal (very high or very low), keep it
                existing = self.csp.pair_weights[pair_key]
                if existing > 0.9 or existing < 0.1:
                    continue  # Don't overwrite strong signals
                # Otherwise, average with new weight
                self.csp.pair_weights[pair_key] = (existing + weight) / 2.0
            else:
                self.csp.pair_weights[pair_key] = weight
        
        # For 3/4 matches, also identify the best trio and strengthen those weights
        if correct_words >= 3:
            best_trio = self._find_best_trio(group_upper)
            if best_trio:
                # Strengthen weights for the trio
                for w1, w2 in combinations(best_trio, 2):
                    pair_key = (w1, w2) if w1 < w2 else (w2, w1)
                    # Boost weight to 0.85-0.95 range
                    sim = self.similarity_fn.similarity(w1, w2)
                    self.csp.pair_weights[pair_key] = 0.85 + (sim * 0.1)
                
                # Weaken weights involving the 4th word
                bottom_word = [w for w in group_upper if w not in best_trio][0]
                for w in best_trio:
                    pair_key = (bottom_word, w) if bottom_word < w else (w, bottom_word)
                    self.csp.pair_weights[pair_key] = 0.1  # Low weight
    
    def _find_best_trio(self, group_upper: List[str]) -> Optional[Tuple[str, str, str]]:
        """Find the 3 words that form the most cohesive group."""
        best_trio = None
        best_score = float('-inf')
        
        for trio in combinations(group_upper, 3):
            trio_score = sum(
                self.similarity_fn.similarity(w1, w2)
                for w1, w2 in combinations(trio, 2)
            ) / 3.0  # C(3,2) = 3 pairs
            
            if trio_score > best_score:
                best_score = trio_score
                best_trio = trio
        
        return best_trio
