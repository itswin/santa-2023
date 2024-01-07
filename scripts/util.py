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
    if "." in move:
        return ".".join(map(invert, move.split(".")))

    if move.startswith("-"):
        return move[1:]
    else:
        return "-" + move


def count_wrong(move):
    identity = np.arange(len(move))
    return np.count_nonzero(move != identity)

class Move:
    def __init__(self, name: str, move: List[int], moves: List[str]):
        self.name = name
        self.move = move
        self.moves = moves
        self.moves_named = ".".join(self.moves)
        self.length = len(moves)
        self.num_wrong = count_wrong(move)

    def count_pre_cancels(self, moves):
        # Count the number of moves that cancel between the 
        # beginning of this commutator and the end of the other commutator

        for i in range(0, min(self.length, len(moves))):
            if self.moves[i] != invert(moves[-i - 1]):
                return i

        return min(self.length, min(self.length, len(moves)))

    def compose(self, other):
        return Move(self.name + "." + other.name, self.move[other.move], self.moves + other.moves)

    def invert(self):
        return Move(
            invert(self.name) if "," not in self.name else "-" + self.name,
            np.argsort(self.move),
            list(map(invert, reversed(self.moves)))
        )

class Commutator(Move):
    def __init__(self, name, puzzle_moves, move_map=None):
        commutator_re = re.compile("\\[(.*),(.*)\\]")
        comm = commutator_re.match(name)
        delimiter = '|' if '|' in comm.group(1) else ' '
        X = comm.group(1).split(delimiter)
        Y = comm.group(2).split(delimiter)
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
        self.num_wrong = count_wrong(self.move)

# Formatted in (SETUP,COMMUTATOR)
# Performs moves SETUP COMMUTATOR SETUP'
class Conjugate(Move):
    def __init__(self, name, puzzle_moves, move_map=None):
        conjugate_re = re.compile("\\((.*),\\[(.*)\\]\\)")
        comm = conjugate_re.match(name)
        setup = comm.group(1).split('|')
        commutator = Commutator(f"[{comm.group(2)}]", puzzle_moves, move_map)

        self.name = name
        self.moves = setup + commutator.moves + list(map(invert, reversed(setup)))
        self.move = puzzle_moves[setup[0]]
        for i in range(1, len(self.moves)):
            self.move = self.move[puzzle_moves[self.moves[i]]]

        self.moves_named = ".".join(self.moves)
        self.length = len(self.moves)
        self.num_wrong = count_wrong(self.move)


def read_conjugates(moves_file, moves, move_map=None, max_wrong=5):
    conjugates = []
    all_move_set = set()
    identity = np.arange(len(moves[list(moves.keys())[0]]))

    with open(moves_file, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line[0] == "(":
                move = Conjugate(line, moves, move_map)
            elif line[0] == "[":
                move = Commutator(line, moves, move_map)
            else:
                print(f"Skipping {line} because it is not a commutator or conjugate")
                continue

            # Skip commutators which are the identity
            if (move.move == identity).all():
                continue

            # Skip commutators which are the same as any existing move
            if tuple(move.move) in all_move_set:
                continue

            if move.num_wrong > max_wrong:
                print(f"Skipping {move.name} because it commutes too many pieces. Commutes {move.num_wrong} > {max_wrong}.")
                continue

            conjugates.append(move)
            all_move_set.add(tuple(move.move))

    return conjugates

def create_commutators(commutator_file, moves, move_map=None, max_wrong=5):
    commutator_re = re.compile(".*(\\[.*\\])")
    commutators = []
    identity = np.arange(len(moves[list(moves.keys())[0]]))

    all_move_set = set()
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
            commutator = Commutator(commutator_name, moves, move_map)

            # Skip commutators which are the identity
            if (commutator.move == identity).all():
                continue

            # Skip commutators which are the same as any existing commutator
            if tuple(commutator.move) in all_move_set:
                continue

            if commutator.num_wrong > max_wrong:
                print(f"Skipping {commutator.name} because it commutes too many pieces. Commutes {commutator.num_wrong} > {max_wrong}.")
                continue

            commutators.append(commutator)
            all_move_set.add(tuple(commutator.move))

    # Create commutators back to back
    # Cm Cn
    # comms2 = []
    # for commutator_list in itertools.product(commutators, repeat=2):
    #     new_comm = commutator_list[0].compose(commutator_list[1])

    #     # Skip comms which are the identity
    #     if new_comm.num_wrong == 0 or new_comm.num_wrong > max_wrong:
    #         continue

    #     if tuple(new_comm.move) in all_move_set:
    #         continue

    #     comms2.append(new_comm)

    # commutators.extend(comms2)

    return commutators

def create_conjugates(commutators, moves, max_setup_moves=1, max_additional_pieces_changed=2, max_wrong=5):
    global identity
    identity = np.arange(len(moves[list(moves.keys())[0]]))

    all_move_set = set()
    for commutator in commutators:
        all_move_set.add(tuple(commutator.move))

    conjugates = []
    for commutator in commutators:
        for num_setup_moves in range(1, max_setup_moves + 1):
            for setup_moves in itertools.product(moves.keys(), repeat=num_setup_moves):
                setup_moves = list(setup_moves)
                # Skip setup moves which are the identity
                setup_move = moves[setup_moves[0]]
                for i in range(1, len(setup_moves)):
                    setup_move = setup_move[moves[setup_moves[i]]]

                if count_wrong(setup_move) == 0:
                    continue

                # Skip setup moves which cancel with the commutator
                if commutator.count_pre_cancels(setup_moves) > 0:
                    continue

                setup_moves_inv = list(map(invert, reversed(setup_moves)))
                setup_move_inv = moves[setup_moves_inv[0]]
                for i in range(1, len(setup_moves_inv)):
                    setup_move_inv = setup_move_inv[moves[setup_moves_inv[i]]]

                conjugate_moves = setup_moves + commutator.moves + setup_moves_inv
                conjugate_move = setup_move[commutator.move][setup_move_inv]

                # Skip conjugates which are the identity
                if count_wrong(conjugate_move) == 0:
                    continue

                # Skip conjugates which are the same as any existing commutator or conjugate
                if tuple(conjugate_move) in all_move_set:
                    continue

                # Format conjugate names as (setup,commutator)
                conjugate = Move(f"({'|'.join(setup_moves)},{commutator.name})", conjugate_move, conjugate_moves)
                if conjugate.num_wrong > commutator.num_wrong + max_additional_pieces_changed:
                    continue

                if conjugate.num_wrong > max_wrong:
                    continue

                conjugates.append(conjugate)
                all_move_set.add(tuple(conjugate_move))

    return conjugates

# Look at moves which overlap to some degree with each other to create new moves
def square_moves(moves):
    all_move_set = set()
    for move in moves:
        all_move_set.add(tuple(move.move))

    new_moves = []
    for move in moves:
        for move2 in moves:
            new_move = move.compose(move2)
            if tuple(new_move.move) in all_move_set:
                continue

            # Skip ones that commute too many pieces
            if new_move.num_wrong >= move.num_wrong + move2.num_wrong:
                # print(f"Skipping {new_move.name} because it commutes too many pieces. Commutes {new_move.num_wrong} >= {move.num_wrong} + {move2.num_wrong}")
                continue

            if new_move.num_wrong == 0:
                # print(f"Skipping {new_move.name} because it is the identity")
                continue

            new_moves.append(new_move)
            # print(f"Found new move. Commutes {new_move.num_wrong}", new_move.name)
            all_move_set.add(tuple(new_move.move))

    return new_moves

def invert_moves(moves):
    all_move_set = set()
    for move in moves:
        all_move_set.add(tuple(move.move))

    new_moves = []
    for move in moves:
        new_move = move.invert()
        if tuple(new_move.move) in all_move_set:
            continue

        new_moves.append(new_move)
        all_move_set.add(tuple(new_move.move))

    return new_moves

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

def create_normalize_inverted_cyclic(is_move_cyclic):
    def f(move):
        if move[0] != '-':
            return move
        elif is_move_cyclic[move[1:]]:
            return move[1:]
        else:
            return move
    return f

def get_cyclic_moves(moves):
    is_move_cyclic = {}
    identity = np.arange(len(moves[list(moves.keys())[0]]))
    for name, move in moves.items():
        m = move[move]
        is_move_cyclic[name] = (m == identity).all()
    return is_move_cyclic

def create_invert_if_not_cycle(is_move_cyclic):
    def f(move):
        if move[0] == '-':
            return move[1:]
        elif is_move_cyclic[move]:
            # Remove the "-" on cyclic moves
            if move[0] == '-':
                return move[1:]
            else:
                return move
        else:
            return "-" + move
    return f
