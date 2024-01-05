from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import re
from pathlib import Path
import itertools

def get_moves(puzzle_type: str) -> Dict[str, List[int]]:
    moves = eval(pd.read_csv("data/puzzle_info.csv").set_index("puzzle_type").loc[puzzle_type, "allowed_moves"])
    np_moves = {}
    for key in moves.keys():
        np_moves[key] = np.array(moves[key])
        np_moves["-" + key] = np.argsort(moves[key])
    return np_moves

def state_to_faces(state_string, n):
    n2 = n ** 2
    return {
        "U": list(state_string[0:n2]),
        "F": list(state_string[n2:2 * n2]),
        "R": list(state_string[2 * n2:3 * n2]),
        "B": list(state_string[3 * n2:4 * n2]),
        "L": list(state_string[4 * n2:5 * n2]),
        "D": list(state_string[5 * n2:6 * n2]),
    }

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def make_cubestring(faces):
    return "".join(faces["U"] + faces["R"] + faces["F"] + faces["D"] + faces["L"] + faces["B"])

def get_center_slice_moves(moves, n):
    half_n = str(n // 2)
    center_slice_moves = {}
    for move in moves.keys():
        if move.endswith(half_n):
            center_slice_moves[move] = moves[move]
    return center_slice_moves

def centers_aligned(state, n, solution=None):
    n2 = n ** 2
    if solution is None:
        solution = "A" * n2 + "B" * n2 + "C" * n2 + "D" * n2 + "E" * n2 + "F" * n2
    return \
        state[n2 // 2] == solution[n2 // 2] and \
        state[n2 + n2 // 2] == solution[n2 + n2 // 2] and \
        state[2 * n2 + n2 // 2] == solution[2 * n2 + n2 // 2] and \
        state[3 * n2 + n2 // 2] == solution[3 * n2 + n2 // 2] and \
        state[4 * n2 + n2 // 2] == solution[4 * n2 + n2 // 2] and \
        state[5 * n2 + n2 // 2] == solution[5 * n2 + n2 // 2]

def extend_move_seq(seq, moves):
    for move in moves:
        yield seq + [move]

def orient_centers(state, moves, n, solution=None):
    center_slice_moves = get_center_slice_moves(moves, n)
    print("Orienting centers")
    print(center_slice_moves)

    # Try longer sequences of moves if the centers are not aligned
    seqs = [[]]
    new_seq = []
    while not centers_aligned(state, n, solution):
        new_seqs = []
        for seq in seqs:
            for new_seq in extend_move_seq(seq, center_slice_moves.keys()):
                new_state = state.copy()
                for move in new_seq:
                    new_state = new_state[center_slice_moves[move]]
                if centers_aligned(new_state, n, solution):
                    print("Found", new_seq)
                    state = new_state
                    break
                else:
                    new_seqs.append(new_seq)
            if centers_aligned(state, n, solution):
                break
        seqs = new_seqs
    return state, new_seq

def get_move_map(n):
    base_moves = {
        "F": "f",
        "R": "r",
        "D": "d",
        "U": "d",
        "B": "f",
        "L": "r",
    }
    move_map = {}
    # "F": "f0",
    # "F'": "-f0",
    # "F2": "f0.f0",
    # "Fw": "f0.f1",
    # "Fw'": "-f0.-f1",
    # "Fw2": "f0.f0.f1.f1",
    # "2F": "f1",
    # "3F": "f2",
    for move in "DFR":
        # Number of layers
        for i in range(1, n // 2 + 1):
            if i == 1:
                move_map[f"{move}"] = f"{base_moves[move]}0"
                move_map[f"{move}'"] = f"-{base_moves[move]}0"
                move_map[f"-{move}"] = f"-{base_moves[move]}0"
                move_map[f"-{move}2"] = f"-{base_moves[move]}0.-{base_moves[move]}0"
                move_map[f"{move}2"] = f"{base_moves[move]}0.{base_moves[move]}0"
            elif i == 2:
                move_map[f"{move}w"] = f"{base_moves[move]}0.{base_moves[move]}1"
                move_map[f"{move}w'"] = f"-{base_moves[move]}0.-{base_moves[move]}1"
                move_map[f"{move}w2"] = f"{base_moves[move]}0.{base_moves[move]}0.{base_moves[move]}1.{base_moves[move]}1"

                move_map[f"2{move}"] = f"{base_moves[move]}1"
                move_map[f"2{move}2"] = f"{base_moves[move]}1.{base_moves[move]}1"
                move_map[f"-2{move}"] = f"-{base_moves[move]}1"
                move_map[f"-2{move}2"] = f"-{base_moves[move]}1.-{base_moves[move]}1"

                # For some reason it also has these
                move_map[f"2{move}w"] = f"{base_moves[move]}0.{base_moves[move]}1"
                move_map[f"2{move}w'"] = f"-{base_moves[move]}0.-{base_moves[move]}1"
                move_map[f"2{move}w2"] = f"{base_moves[move]}0.{base_moves[move]}0.{base_moves[move]}1.{base_moves[move]}1"
            else:
                move_map[f"{i}{move}w"] = ".".join([f"{base_moves[move]}{j}" for j in range(i)])
                move_map[f"{i}{move}w'"] = ".".join([f"-{base_moves[move]}{j}" for j in range(i)])
                move_map[f"{i}{move}w2"] = ".".join([f"{base_moves[move]}{j}" for j in range(i)] + [f"{base_moves[move]}{j}" for j in range(i)])

                move_map[f"{i}{move}"] = f"{base_moves[move]}{i - 1}"
                move_map[f"{i}{move}2"] = f"{base_moves[move]}{i - 1}.{base_moves[move]}{i - 1}"
                move_map[f"-{i}{move}"] = f"-{base_moves[move]}{i - 1}"
                move_map[f"-{i}{move}2"] = f"-{base_moves[move]}{i - 1}.-{base_moves[move]}{i - 1}"
    for move in "BUL":
        # Number of layers
        for i in range(1, n // 2 + 1):
            if i == 1:
                move_map[f"{move}"] = f"-{base_moves[move]}{n - 1}"
                move_map[f"{move}'"] = f"{base_moves[move]}{n - 1}"
                move_map[f"-{move}"] = f"{base_moves[move]}{n - 1}"
                move_map[f"-{move}2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}"
                move_map[f"{move}2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}"
            elif i == 2:
                move_map[f"{move}w"] = f"-{base_moves[move]}{n - 1}.-{base_moves[move]}{n - 2}"
                move_map[f"{move}w'"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}"
                move_map[f"{move}w2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}.{base_moves[move]}{n - 2}"

                move_map[f"2{move}"] = f"-{base_moves[move]}{n - 2}"
                move_map[f"2{move}2"] = f"-{base_moves[move]}{n - 2}.-{base_moves[move]}{n - 2}"
                move_map[f"-2{move}"] = f"{base_moves[move]}{n - 2}"
                move_map[f"-2{move}2"] = f"{base_moves[move]}{n - 2}.{base_moves[move]}{n - 2}"

                # For some reason it also has these
                move_map[f"2{move}w"] = f"-{base_moves[move]}{n - 1}.-{base_moves[move]}{n - 2}"
                move_map[f"2{move}w'"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}"
                move_map[f"2{move}w2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}.{base_moves[move]}{n - 2}"
            else:
                move_map[f"{i}{move}w"] = ".".join([f"-{base_moves[move]}{n - 1 - j}" for j in range(i)])
                move_map[f"{i}{move}w'"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in range(i)])
                move_map[f"{i}{move}w2"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in range(i)] + [f"{base_moves[move]}{n - 1 - j}" for j in range(i)])

                move_map[f"{i}{move}"] = f"{base_moves[move]}{n - i}"
                move_map[f"{i}{move}2"] = f"{base_moves[move]}{n - i}.{base_moves[move]}{n - i}"
                move_map[f"-{i}{move}"] = f"-{base_moves[move]}{n - i}"
                move_map[f"-{i}{move}2"] = f"-{base_moves[move]}{n - i}.-{base_moves[move]}{n - i}"
    return move_map

def print_faces(faces, n):
    for face in faces:
        print(face)
        for row in chunks(faces[face], n):
            print("\t", " ".join(row))

def get_edges(n, skip=2):
    edges = []

    n2 = n ** 2
    # U face: [0, n2)
    # F face: [n2, 2 * n2)
    # R face: [2 * n2, 3 * n2)
    # B face: [3 * n2, 4 * n2)
    # L face: [4 * n2, 5 * n2)
    # D face: [5 * n2, 6 * n2)

    # UF edge:
    #   U: [(n-1) * n + 1, n2 - 2]
    #   F: [n2 + 1, n2 + n - 2]
    edges.extend(list(zip(range((n-1) * n + 1, n2 - 1), range(n2 + 1, n2 + n - 1)))[::skip])

    # UR edge:
    #  U: [(n-1) + n * i for i in range(1, n - 1)] (Reversed)
    #  R: [2 * n2 + 1, 2 * n2 + n - 2]
    edges.extend(list(zip([(n-1) + n * i for i in range(1, n - 1)][::-1], range(2 * n2 + 1, 2 * n2 + n - 1)))[::skip])

    # UB edge:
    #  U: [1, n - 2] (Reversed)
    #  B: [3 * n2 + 1, 3 * n2 + n - 2]
    edges.extend(list(zip(range(1, n - 1)[::-1], range(3 * n2 + 1, 3 * n2 + n - 1)))[::skip])

    # UL edge:
    #  U: [n * i for i in range(1, n - 1)]
    #  L: [4 * n2 + 1, 4 * n2 + n - 2]
    edges.extend(list(zip([n * i for i in range(1, n - 1)], range(4 * n2 + 1, 4 * n2 + n - 1)))[::skip])

    # FR edge:
    #  F: [n2 + (n-1) + n * i for i in range(1, n - 1)]
    #  R: [2 * n2 + n * i for i in range(1, n - 1)]
    edges.extend(list(zip([n2 + (n-1) + n * i for i in range(1, n - 1)], [2 * n2 + n * i for i in range(1, n - 1)]))[::skip])

    # RB edge:
    #  R: [2 * n2 + (n-1) + n * i for i in range(1, n - 1)]
    #  B: [3 * n2 + n * i for i in range(1, n - 1)]
    edges.extend(list(zip([2 * n2 + (n-1) + n * i for i in range(1, n - 1)], [3 * n2 + n * i for i in range(1, n - 1)]))[::skip])

    # BL edge:
    #  B: [3 * n2 + (n-1) + n * i for i in range(1, n - 1)]
    #  L: [4 * n2 + n * i for i in range(1, n - 1)]
    edges.extend(list(zip([3 * n2 + (n-1) + n * i for i in range(1, n - 1)], [4 * n2 + n * i for i in range(1, n - 1)]))[::skip])

    # LF edge:
    #  L: [4 * n2 + (n-1) + n * i for i in range(1, n - 1)]
    #  F: [n2 + n * i for i in range(1, n - 1)]
    edges.extend(list(zip([4 * n2 + (n-1) + n * i for i in range(1, n - 1)], [n2 + n * i for i in range(1, n - 1)]))[::skip])

    # DF edge:
    #  D: [5 * n2 + 1, 5 * n2 + n - 2]
    #  F: [n2 + (n-1) * n + 1, n2 + n2 - 2]
    edges.extend(list(zip(range(5 * n2 + 1, 5 * n2 + n - 1), range(n2 + (n-1) * n + 1, n2 + n2 - 1)))[::skip])

    # DR edge:
    #  D: [5 * n2 + (n-1) + n * i for i in range(1, n - 1)]
    #  R: [2 * n2 + (n-1) * n + 1, 2 * n2 + n2 - 2]
    edges.extend(list(zip([5 * n2 + (n-1) + n * i for i in range(1, n - 1)], range(2 * n2 + (n-1) * n + 1, 2 * n2 + n2 - 1)))[::skip])

    # DB edge:
    #   D: [5 * n2 + (n-1) * n + 1, 5 * n2 + n2 - 2] (Reversed)
    #   B: [3 * n2 + (n-1) * n + 1, 3 * n2 + n2 - 2]
    edges.extend(list(zip(range(5 * n2 + (n-1) * n + 1, 5 * n2 + n2 - 1)[::-1], range(3 * n2 + (n-1) * n + 1, 3 * n2 + n2 - 1)))[::skip])

    # DL edge:
    #   D: [5 * n2 + n * i for i in range(1, n - 1)] (Reversed)
    #   L: [4 * n2 + (n-1) * n + 1, 4 * n2 + n2 - 2]
    edges.extend(list(zip([5 * n2 + n * i for i in range(1, n - 1)][::-1], range(4 * n2 + (n-1) * n + 1, 4 * n2 + n2 - 1)))[::skip])

    return edges

# Reskinned cube solution faces look like
#         A B A B A
#         B A B A B
#         A B A B A
#         B A B A B
#         A B A B A
# We want to return the indexes of the Xs on each face.
#         A B A B A
#         B A X A B
#         A X A X A
#         B A X A B
#         A B A B A
def get_diff_odd_centers(n):
    n2 = n ** 2
    odd_centers = []

    # Iterate through lines 2 to n-1 on each face
    # Even lines should take the odd "centers"
    # Odd lines should take the even "centers"
    for face_start in range(0, 6 * n2, n2):
        for row in range(1, n - 1):
            # Iterate over each row from 2 to n-1
            # Even lines should take the odd columns
            # Odd lines should take the even columns
            for col in range(1, n - 1):
                if row % 2 == 0 and col % 2 == 1:
                    odd_centers.append(face_start + row * n + col)
                elif row % 2 == 1 and col % 2 == 0:
                    odd_centers.append(face_start + row * n + col)

    return odd_centers

def make_edge_reskin_map(edges, reskin_solution, normal_solution):
    edge_map = {}
    for edge in edges:
        print(edge)
        reskin = reskin_solution[edge[0]] + reskin_solution[edge[1]]
        normal = normal_solution[edge[0]] + normal_solution[edge[1]]
        reskin_ = reskin_solution[edge[1]] + reskin_solution[edge[0]]
        normal_ = normal_solution[edge[1]] + normal_solution[edge[0]]

        if reskin in edge_map and edge_map[reskin] != normal:
            print("WARNING: Duplicate edge", reskin, normal)

        edge_map[reskin] = normal
        edge_map[reskin_] = normal_
        print(reskin, normal)

    return edge_map

def make_odd_center_reskin_map(odd_centers, reskin_solution, normal_solution):
    odd_center_map = {}
    for center in odd_centers:
        odd_center_map[reskin_solution[center]] = normal_solution[center]
    return odd_center_map

def invert(move):
    if move.startswith("-"):
        return move[1:]
    else:
        return "-" + move

class Commutator:
    def __init__(self, name, puzzle_moves, move_map=None):
        commutator_re = re.compile("\\[(.*),(.*)\\]")
        comm = commutator_re.match(name)
        X = comm.group(1).split(" ")
        Y = comm.group(2).split(" ")
        X_inv = list(map(invert, reversed(X)))
        Y_inv = list(map(invert, reversed(Y)))

        if move_map:
            X = list(map(lambda x: move_map[x], X))
            Y = list(map(lambda y: move_map[y], Y))
            X_inv = list(map(lambda x: move_map[x], X_inv))
            Y_inv = list(map(lambda y: move_map[y], Y_inv))
            name = f"[{'|'.join(X)},{'|'.join(Y)}]"

        self.name = name

        self.moves = []
        for x in X:
            self.moves.extend(x.split("."))
        for y in Y:
            self.moves.extend(y.split("."))
        for x in X_inv:
            self.moves.extend(x.split("."))
        for y in Y_inv:
            self.moves.extend(y.split("."))

        self.move = puzzle_moves[self.moves[0]]
        for i in range(1, len(self.moves)):
            self.move = self.move[puzzle_moves[self.moves[i]]]

        self.moves_named = ".".join(self.moves)
        self.length = len(self.moves)
    
    def count_pre_cancels(self, move_list):
        # Count the number of moves that cancel between the 
        # beginning of this commutator and the end of the other commutator

        for i in range(0, min(self.length, len(move_list))):
            if self.moves[i] != invert(move_list[-i - 1]):
                return i
   
        return min(self.length, min(self.length, len(move_list)))


def create_commutators(commutator_file, moves, move_map=None):
    commutator_re = re.compile(".*(\\[.*\\])")
    commutators = []
    print(move_map)
    with open(commutator_file, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            comm = commutator_re.match(line)
            if comm is None:
                continue
            commutator_name = comm.group(1)
            commutators.append(Commutator(commutator_name, moves, move_map))

    return commutators

def write_tws_file(puzzle, unique=False, commutators=None):
    full_moves = get_moves(puzzle["puzzle_type"])

    sol_state = puzzle["solution_state"].split(";")
    num_pieces = len(sol_state)

    if unique:
        mapped_sol_state = list(range(1, num_pieces + 1))
    else:
        piece_to_index = {}
        last_index = 1
        mapped_sol_state = []
        for piece in sol_state:
            if piece not in piece_to_index:
                piece_to_index[piece] = last_index
                last_index += 1
            mapped_sol_state.append(piece_to_index[piece])

    out = f"""
Name {puzzle["puzzle_type"]}

Set PIECE {num_pieces} 1

Solved
PIECE
{" ".join(map(str, mapped_sol_state))}
End

"""

    format = """
Move {}
PIECE
{}
End
"""

    for move, perm in full_moves.items():
        if move[0] == "-":
            continue
        l = list(perm)
        out += format.format(move, " ".join(map(lambda x: str(x+1), l)))

    if commutators:
        for move, perm in commutators.items():
            l = list(perm)
            out += format.format(move, " ".join(map(lambda x: str(x+1), l)))

    puzzle_name = puzzle["puzzle_type"].replace("/", "_")
    twsearch_puzzles = f"./data/tws_phases/{puzzle_name}/"

    Path(twsearch_puzzles).mkdir(parents=True, exist_ok=True)

    name = twsearch_puzzles + puzzle_name + \
        f"{'_unique' if unique else ''}" + \
        f"{'_commutators' if commutators else ''}" + \
        ".tws"
    with open(name, 'w+') as tws_file:
        tws_file.write(out)

    print("Wrote twsearch file to", name)
    return name

def write_piece_phases(puzzle, at_once=1):
    full_moves = get_moves(puzzle["puzzle_type"])

    sol_state = puzzle["solution_state"].split(";")
    num_pieces = len(sol_state)
    assert num_pieces % at_once == 0

    preamble = f"""
Name {puzzle["puzzle_type"]}

Set PIECE {num_pieces} 1

Solved
PIECE
{{solution_perm}}
End

"""

    ending = ""

    format = """
Move {}
PIECE
{}
End
"""

    for move, perm in full_moves.items():
        if move[0] == "-":
            continue
        l = list(perm)
        ending += format.format(move, " ".join(map(lambda x: str(x+1), l)))

    puzzle_name = puzzle["puzzle_type"].replace("/", "_")
    twsearch_puzzles = f"/Users/Win33/Documents/Programming/santa-2023/data/tws_phases/{puzzle_name}_pbp/"

    for i in range(1, num_pieces // at_once):
        start = []
        for _ in range(at_once):
            start.extend(list(range(2, 2 + i)))
        start.sort()
        solution_state = " ".join(map(str, start)) + " " + ("1 " * (num_pieces - at_once * i)).strip()
        out = preamble.format(solution_perm=solution_state) + ending
        name = twsearch_puzzles + puzzle_name + f"_pbp{at_once}_{i}.tws"
        with open(name, 'w+') as tws_file:
            tws_file.write(out)

    return name

def get_phase_list(file_name):
    with open(file_name, "r") as fp:
        lines = fp.readlines()
        return [line.strip() for line in lines]

def clear_line():
    print("*" * 80, end="\x1b[1K\r")

def evaluate_difference(current_state, final_state):
    return np.count_nonzero(current_state != final_state)
