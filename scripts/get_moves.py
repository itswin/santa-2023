from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

def get_moves(puzzle_type: str) -> Dict[str, List[int]]:
    moves = eval(pd.read_csv("data/puzzle_info.csv").set_index("puzzle_type").loc[puzzle_type, "allowed_moves"])
    np_moves = {}
    for key in moves.keys():
        np_moves[key] = np.array(moves[key])
        np_moves["-" + key] = np.argsort(moves[key])
    return np_moves
