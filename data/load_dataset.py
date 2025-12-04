# load the nyt connections dataset from huggingface
from datasets import load_dataset
from typing import List, Dict, Optional


class Puzzle:
    """Represents a NYT Connections puzzle."""
    
    def __init__(self, puzzle_id: int, words: List[str], 
                 groups: Optional[Dict[int, List[str]]] = None,
                 category_descriptions: Optional[Dict[int, str]] = None,
                 difficulty: Optional[float] = None,
                 contest: Optional[str] = None,
                 date: Optional[str] = None):
        self.puzzle_id = puzzle_id
        self.words = words
        self.groups = groups or {}
        self.category_descriptions = category_descriptions or {}
        self.difficulty = difficulty
        self.contest = contest
        self.date = date


def parse_puzzle_from_dataset(dataset_item, index: int) -> Puzzle:
    """
    parse an item from the dataset into a Puzzle object

    args:
        dataset_item: an item from the NYT Connections dataset
        index: the index of the puzzle in the dataset

    returns:
        a Puzzle object
    """
    words = dataset_item["words"]
    
    groups = {}
    category_descriptions = {}
    
    for i, answer in enumerate(dataset_item["answers"], 1):
        groups[i] = answer["words"]
        category_descriptions[i] = answer.get("answerDescription", "")
    
    return Puzzle(
        puzzle_id=index,
        words=words,
        groups=groups,
        category_descriptions=category_descriptions,
        difficulty=dataset_item.get("difficulty"),
        contest=dataset_item.get("contest"),
        date=str(dataset_item.get("date", ""))
    )


def load_historical_data() -> List[Puzzle]:
    """
    Load and parse puzzles from nyt connections dataset
    dataset url: https://huggingface.co/datasets/tm21cy/NYT-Connections
    """
    dataset = load_dataset("tm21cy/NYT-Connections", split="train")
    puzzles = []
    for i in range(len(dataset)):
        puzzles.append(parse_puzzle_from_dataset(dataset[i], i))
    return puzzles
