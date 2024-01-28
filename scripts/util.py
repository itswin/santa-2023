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

def expand_moves(moves: Dict[str, List[int]]) -> Dict[str, List[int]]:
    expanded_moves = {}
    identity = np.arange(len(moves[list(moves.keys())[0]]))
    for name, move in moves.items():
        expanded_moves[name] = move
        next_move = move[move]
        # Find the cycle length
        cycle_length = 1
        while not np.array_equal(next_move, identity):
            next_move = move[next_move]
            cycle_length += 1

        next_move = move[move]
        i = 1
        last_name = name
        while not np.array_equal(next_move, identity) and i < cycle_length / 2:
            last_name += "." + name
            expanded_moves[last_name] = next_move
            next_move = move[next_move]
            i += 1
    return expanded_moves

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

def get_3x3_faces(faces, n):
    assert n % 2 == 1
    n2 = n ** 2
    indices = np.asarray([
        0, n // 2, n - 1,
        n // 2 * n, n // 2 * n + n // 2, n // 2 * n + n - 1,
        n2 - n, n2 - n // 2 - 1, n2 - 1
    ])
    print(faces)
    return {
        "U": list(np.asarray(faces["U"])[indices]),
        "F": list(np.asarray(faces["F"])[indices]),
        "R": list(np.asarray(faces["R"])[indices]),
        "B": list(np.asarray(faces["B"])[indices]),
        "L": list(np.asarray(faces["L"])[indices]),
        "D": list(np.asarray(faces["D"])[indices]),
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
    # print(center_slice_moves)

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

# Cube notation to santa notation
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

    # Cube rotations
    move_map["y"] = ".".join(f"-d{j}" for j in range(n))
    move_map["y'"] = invert(move_map["y"])
    move_map["y2"] = move_map["y"] + "." + move_map["y"]
    move_map["x"] = ".".join(f"r{j}" for j in range(n))
    move_map["x'"] = invert(move_map["x"])
    move_map["x2"] = move_map["x"] + "." + move_map["x"]
    move_map["z"] = ".".join(f"f{j}" for j in range(n))
    move_map["z'"] = invert(move_map["z"])

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
        for i in range(1, n // 2 + 2):
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

                move_map[f"{move.lower()}"] = move_map[f"{move}w"]
                move_map[f"{move.lower()}'"] = move_map[f"{move}w'"]
                move_map[f"{move.lower()}2"] = move_map[f"{move}w2"]

                move_map[f"2{move}"] = f"{base_moves[move]}1"
                move_map[f"2{move}2"] = f"{base_moves[move]}1.{base_moves[move]}1"
                move_map[f"-2{move}"] = f"-{base_moves[move]}1"
                move_map[f"2{move}'"] = f"-{base_moves[move]}1"
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
                move_map[f"{i}{move}'"] = f"-{base_moves[move]}{i - 1}"
                move_map[f"-{i}{move}2"] = f"-{base_moves[move]}{i - 1}.-{base_moves[move]}{i - 1}"
    for move in "BUL":
        # Number of layers
        for i in range(1, n // 2 + 2):
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

                move_map[f"{move.lower()}"] = move_map[f"{move}w"]
                move_map[f"{move.lower()}'"] = move_map[f"{move}w'"]
                move_map[f"{move.lower()}2"] = move_map[f"{move}w2"]

                move_map[f"2{move}"] = f"-{base_moves[move]}{n - 2}"
                move_map[f"2{move}2"] = f"-{base_moves[move]}{n - 2}.-{base_moves[move]}{n - 2}"
                move_map[f"-2{move}"] = f"{base_moves[move]}{n - 2}"
                move_map[f"2{move}'"] = f"{base_moves[move]}{n - 2}"
                move_map[f"-2{move}2"] = f"{base_moves[move]}{n - 2}.{base_moves[move]}{n - 2}"

                # For some reason it also has these
                move_map[f"2{move}w"] = f"-{base_moves[move]}{n - 1}.-{base_moves[move]}{n - 2}"
                move_map[f"2{move}w'"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}"
                move_map[f"2{move}w2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}.{base_moves[move]}{n - 2}"
            else:
                move_map[f"{i}{move}w"] = ".".join([f"-{base_moves[move]}{n - 1 - j}" for j in range(i)])
                move_map[f"{i}{move}w'"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in range(i)])
                move_map[f"{i}{move}w2"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in range(i)] + [f"{base_moves[move]}{n - 1 - j}" for j in range(i)])

                move_map[f"{i}{move}"] = f"-{base_moves[move]}{n - i}"
                move_map[f"{i}{move}2"] = f"-{base_moves[move]}{n - i}.-{base_moves[move]}{n - i}"
                move_map[f"-{i}{move}"] = f"{base_moves[move]}{n - i}"
                move_map[f"{i}{move}'"] = f"{base_moves[move]}{n - i}"
                move_map[f"-{i}{move}2"] = f"{base_moves[move]}{n - i}.{base_moves[move]}{n - i}"
    return move_map

# Santa notation to cube notation
def get_inverse_move_map(n, use_minus=True):
    move_map = get_move_map(n)
    inverse_move_map = {}
    for move, inverse in move_map.items():
        if not use_minus:
            if "-" in move:
                move = move[1:] + "'"
        inverse_move_map[inverse] = move
    return inverse_move_map

def get_santa_to_sse_move_map(n):
    moves = get_moves(f"cube_{n}/{n}/{n}")
    santa_to_sse = {}

    flip_move = {
        "U": "D",
        "D": "U",
        "L": "R",
        "R": "L",
        "F": "B",
        "B": "F"
    }

    for move in moves:
        regex = re.compile(r"(-?)([a-z])(\d+)")
        match = regex.match(move)

        inverse = match.group(1) == "-"
        move_type = match.group(2).upper()
        layer = int(match.group(3))
        add_invert = "'" if inverse else ""

        if layer >= n // 2:
            move_type = flip_move[move_type]
            layer = n - layer - 1
            add_invert = "" if inverse else "'"

        if layer == 0:
            # r0 to R
            # r5 to L
            santa_to_sse[move] = move_type + add_invert
        elif layer == 1:
            # r1 to NR
            santa_to_sse[move] = "N" + move_type + add_invert
        else:
            santa_to_sse[move] = "N" + str(layer + 1) + move_type + add_invert

    return santa_to_sse

def get_sse_to_santa_move_map(n):
    sse_to_santa = {}

    base_moves = {
        "F": "f",
        "R": "r",
        "D": "d",
        "U": "d",
        "B": "f",
        "L": "r",
    }

    flip_move = {
        "U": "D",
        "D": "U",
        "L": "R",
        "R": "L",
        "F": "B",
        "B": "F"
    }

    # Normal moves
    for move in "FRD":
        sse_to_santa[move] = f"{base_moves[move]}0"
        sse_to_santa[move + "'"] = f"-{base_moves[move]}0"
        sse_to_santa[move + "2"] = f"{base_moves[move]}0.{base_moves[move]}0"
    for move in "ULB":
        sse_to_santa[move] = f"-{base_moves[move]}{n - 1}"
        sse_to_santa[move + "'"] = f"{base_moves[move]}{n - 1}"
        sse_to_santa[move + "2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}"

    # 4x4 biases towards the side given
    # 5x5 is only the center
    half_n_1 = (n - 1) // 2

    # Mid layer twists
    for move in "FRD":
        sse_to_santa["M" + move] = f"{base_moves[move]}{half_n_1}"
        sse_to_santa["M" + move + "'"] = f"-{base_moves[move]}{half_n_1}"
        sse_to_santa["M" + move + "2"] = f"{base_moves[move]}{half_n_1}.{base_moves[move]}{half_n_1}"
    for move in "ULB":
        sse_to_santa["M" + move] = f"-{base_moves[move]}{n - 1 - half_n_1}"
        sse_to_santa["M" + move + "'"] = f"{base_moves[move]}{n - 1 - half_n_1}"
        sse_to_santa["M" + move + "2"] = f"{base_moves[move]}{n - 1 - half_n_1}.{base_moves[move]}{n - 1 - half_n_1}"
    if n in (6, 7):
        if n == 6:
            mid_layer_map = {
                2: [2, 3],
                3: [1, 2, 3],
                # 4 is just a wide layer twist it shouldn't exist
                # 4: [1, 2, 3, 4]
            }
        else:
            mid_layer_map = {
                2: [2, 3],
                3: [2, 3, 4],
                4: [1, 2, 3, 4]
                # 5 is just a wide layer twist it shouldn't exist
                # 5: [1, 2, 3, 4, 5]
            }
        for move in "FRD":
            for i, layers in mid_layer_map.items():
                sse_to_santa[f"M{i}{move}"] = ".".join([f"{base_moves[move]}{j}" for j in layers])
                sse_to_santa[f"M{i}{move}'"] = ".".join([f"-{base_moves[move]}{j}" for j in layers])
                sse_to_santa[f"M{i}{move}2"] = sse_to_santa[f"M{i}{move}"] + "." + sse_to_santa[f"M{i}{move}"]
        for move in "ULB":
            for i, layers in mid_layer_map.items():
                sse_to_santa[f"M{i}{move}"] = ".".join([f"-{base_moves[move]}{n - 1 - j}" for j in layers])
                sse_to_santa[f"M{i}{move}'"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in layers])
                sse_to_santa[f"M{i}{move}2"] = sse_to_santa[f"M{i}{move}"] + "." + sse_to_santa[f"M{i}{move}"]


    # Numbered layer twists
    for i in range(2, n - 1):
        for move in "FRD":
            sse_name = f"N{i}{move}" if i > 2 else f"N{move}"
            sse_to_santa[sse_name] = f"{base_moves[move]}{i - 1}"
            sse_to_santa[sse_name + "'"] = f"-{base_moves[move]}{i - 1}"
            sse_to_santa[sse_name + "2"] = f"{base_moves[move]}{i - 1}.{base_moves[move]}{i - 1}"
        for move in "ULB":
            sse_name = f"N{i}{move}" if i > 2 else f"N{move}"
            sse_to_santa[sse_name] = f"-{base_moves[move]}{n - i}"
            sse_to_santa[sse_name + "'"] = f"{base_moves[move]}{n - i}"
            sse_to_santa[sse_name + "2"] = f"{base_moves[move]}{n - i}.{base_moves[move]}{n - i}"

    # Tier twists
    for i in range(2, n):
        for move in "FRD":
            if i == 2:
                sse_to_santa["T" + move] = f"{base_moves[move]}0.{base_moves[move]}1"
                sse_to_santa["T" + move + "'"] = f"-{base_moves[move]}0.-{base_moves[move]}1"
                sse_to_santa["T" + move + "2"] = f"{base_moves[move]}0.{base_moves[move]}1.{base_moves[move]}0.{base_moves[move]}1"
            else:
                sse_to_santa[f"T{i}{move}"] = ".".join([f"{base_moves[move]}{j}" for j in range(i)])
                sse_to_santa[f"T{i}{move}'"] = ".".join([f"-{base_moves[move]}{j}" for j in range(i)])
                sse_to_santa[f"T{i}{move}2"] = ".".join([f"{base_moves[move]}{j}" for j in range(i)] + [f"{base_moves[move]}{j}" for j in range(i)])

        for move in "ULB":
            if i == 2:
                sse_to_santa["T" + move] = f"-{base_moves[move]}{n - 1}.-{base_moves[move]}{n - 2}"
                sse_to_santa["T" + move + "'"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}"
                sse_to_santa["T" + move + "2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}.{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}"
            else:
                sse_to_santa[f"T{i}{move}"] = ".".join([f"-{base_moves[move]}{n - 1 - j}" for j in range(i)])
                sse_to_santa[f"T{i}{move}'"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in range(i)])
                sse_to_santa[f"T{i}{move}2"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in range(i)] + [f"{base_moves[move]}{n - 1 - j}" for j in range(i)])

    # Slice twists
    for i in range(1, n // 2):
        for move in "FRD":
            if i == 1:
                slice = "S" + move
                sse_to_santa[slice] = f"{sse_to_santa[move]}.{invert(sse_to_santa[flip_move[move]])}"
                sse_to_santa[slice + "'"] = f"{sse_to_santa[move + "'"]}.{invert(sse_to_santa[flip_move[move] + "'"])}"
                sse_to_santa[slice + "2"] = f"{sse_to_santa[slice]}.{sse_to_santa[slice]}"
            else:
                slice = f"S{i}{move}"

                tier_twist = f"T{i}{move}" if i > 2 else f"T{move}"
                tier_twist_flipped = f"T{i}{flip_move[move]}" if i > 2 else f"T{flip_move[move]}"

                sse_to_santa[slice] = f"{sse_to_santa[tier_twist]}.{sse_to_santa[tier_twist_flipped]}"
                sse_to_santa[slice + "'"] = f"{sse_to_santa[tier_twist + "'"]}.{sse_to_santa[tier_twist_flipped + "'"]}"
                sse_to_santa[slice + "2"] = f"{sse_to_santa[tier_twist]}.{sse_to_santa[tier_twist]}"
        for move in "ULB":
            if i == 1:
                slice = "S" + move
                sse_to_santa[slice] = f"{sse_to_santa[move]}.{invert(sse_to_santa[flip_move[move]])}"
                sse_to_santa[slice + "'"] = f"{sse_to_santa[move + "'"]}.{invert(sse_to_santa[flip_move[move] + "'"])}"
                sse_to_santa[slice + "2"] = f"{sse_to_santa[slice]}.{sse_to_santa[slice]}"
            else:
                slice = f"S{i}{move}"

                tier_twist = f"T{i}{move}" if i > 2 else f"T{move}"
                tier_twist_flipped = f"T{i}{flip_move[move]}" if i > 2 else f"T{flip_move[move]}"

                sse_to_santa[slice] = f"{sse_to_santa[tier_twist]}.{sse_to_santa[tier_twist_flipped]}"
                sse_to_santa[slice + "'"] = f"{sse_to_santa[tier_twist + "'"]}.{sse_to_santa[tier_twist_flipped + "'"]}"
                sse_to_santa[slice + "2"] = f"{sse_to_santa[tier_twist]}.{sse_to_santa[tier_twist]}"

    # Wide layer twists all inner layers
    for move in "FRD":
        sse_to_santa["W" + move] = ".".join([f"{base_moves[move]}{j}" for j in range(1, n - 1)])
        sse_to_santa["W" + move + "'"] = invert(sse_to_santa["W" + move])
        sse_to_santa["W" + move + "2"] = sse_to_santa["W" + move] + "." + sse_to_santa["W" + move]
    for move in "ULB":
        sse_to_santa["W" + move] = ".".join([f"-{base_moves[move]}{j}" for j in range(1, n - 1)])
        sse_to_santa["W" + move + "'"] = invert(sse_to_santa["W" + move])
        sse_to_santa["W" + move + "2"] = sse_to_santa["W" + move] + "." + sse_to_santa["W" + move]

    # Void twists
    # Tier twist without the outer face
    for i in range(2, n - 1):
        for move in "FRD":
            if i == 2:
                sse_to_santa["V" + move] = f"{base_moves[move]}1.{base_moves[move]}2"
                sse_to_santa["V" + move + "'"] = f"-{base_moves[move]}1.-{base_moves[move]}2"
                sse_to_santa["V" + move + "2"] = f"{base_moves[move]}1.{base_moves[move]}2.{base_moves[move]}1.{base_moves[move]}2"
            else:
                sse_to_santa[f"V{i}{move}"] = ".".join([f"{base_moves[move]}{j}" for j in range(1, i + 1)])
                sse_to_santa[f"V{i}{move}'"] = ".".join([f"-{base_moves[move]}{j}" for j in range(1, i + 1)])
                sse_to_santa[f"V{i}{move}2"] = ".".join([f"{base_moves[move]}{j}" for j in range(1, i + 1)] + [f"{base_moves[move]}{j}" for j in range(1, i + 1)])

        for move in "ULB":
            if i == 2:
                sse_to_santa["V" + move] = f"-{base_moves[move]}{n - 2}.-{base_moves[move]}{n - 3}"
                sse_to_santa["V" + move + "'"] = f"{base_moves[move]}{n - 2}.{base_moves[move]}{n - 3}"
                sse_to_santa["V" + move + "2"] = f"{base_moves[move]}{n - 2}.{base_moves[move]}{n - 3}.{base_moves[move]}{n - 2}.{base_moves[move]}{n - 3}"
            else:
                sse_to_santa[f"V{i}{move}"] = ".".join([f"-{base_moves[move]}{n - 1 - j}" for j in range(1, i + 1)])
                sse_to_santa[f"V{i}{move}'"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in range(1, i + 1)])
                sse_to_santa[f"V{i}{move}2"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in range(1, i + 1)] + [f"{base_moves[move]}{n - 1 - j}" for j in range(1, i + 1)])

    # Cube rotations
    # These really should not exist...
    for move in "FRD":
        sse_to_santa["C" + move] = ".".join([f"{base_moves[move]}{j}" for j in range(n)])
        sse_to_santa["C" + move + "'"] = ".".join([f"-{base_moves[move]}{j}" for j in range(n)])
        sse_to_santa["C" + move + "2"] = sse_to_santa["C" + move] + "." + sse_to_santa["C" + move]
    for move in "ULB":
        sse_to_santa["C" + move] = ".".join([f"-{base_moves[move]}{j}" for j in range(n)])
        sse_to_santa["C" + move + "'"] = ".".join([f"{base_moves[move]}{j}" for j in range(n)])
        sse_to_santa["C" + move + "2"] = sse_to_santa["C" + move] + "." + sse_to_santa["C" + move]

    return sse_to_santa

def print_faces(faces, n):
    for face in faces:
        print(face)
        for row in chunks(faces[face], n):
            print("\t", " ".join(row))

def get_edges(n, skip=1):
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

def get_corners(n):
    corners = []

    n2 = n ** 2
    # UFR corner:
    #   U: [n2 - 1]
    #   F: [n2 + n - 1]
    #   R: [2 * n2]
    corners.append((n2 - 1, n2 + n - 1, 2 * n2))

    # URB corner:
    #   U: [n - 1]
    #   R: [2 * n2 + n - 1]
    #   B: [3 * n2]
    corners.append((n - 1, 2 * n2 + n - 1, 3 * n2))

    # UBL corner:
    #   U: [0]
    #   B: [3 * n2 + n - 1]
    #   L: [4 * n2]
    corners.append((0, 3 * n2 + n - 1, 4 * n2))

    # ULF corner:
    #   U: [n2 - n]
    #   L: [4 * n2 + n - 1]
    #   F: [n2]
    corners.append((n2 - n, 4 * n2 + n - 1, n2))

    # DRF corner:
    #   D: [5 * n2 + n - 1]
    #   R: [2 * n2 + (n-1) * n]
    #   F: [2 * n2 - 1]
    corners.append((5 * n2 + n - 1, 2 * n2 + (n-1) * n, 2 * n2 - 1))

    # DFL corner:
    #   D: [5 * n2]
    #   F: [n2 + (n-1) * n]
    #   L: [5 * n2 - 1]
    corners.append((5 * n2, n2 + (n-1) * n, 5 * n2 - 1))

    # DLB corner:
    #   D: [5 * n2 + n * (n-1)]
    #   L: [4 * n2 + (n-1) * n]
    #   B: [4 * n2 - 1]
    corners.append((5 * n2 + n * (n-1), 4 * n2 + (n-1) * n, 4 * n2 - 1))

    # DBR corner:
    #   D: [6 * n2 - 1]
    #   B: [3 * n2 + (n-1) * n]
    #   R: [3 * n2 - 1]
    corners.append((6 * n2 - 1, 3 * n2 + (n-1) * n, 3 * n2 - 1))

    return corners

def get_centers(n):
    edge_pairs = get_edges(n)
    corner_triplets = get_corners(n)
    other = []
    for edge_pair in edge_pairs:
        other.extend(edge_pair)
    for corner_triplet in corner_triplets:
        other.extend(corner_triplet)
    return list(set(range(6 * n ** 2)) - set(other))

def remove_invert(move):
    if move[0] == "-":
        return move[1:]
    else:
        return move

def invert(move):
    if isinstance(move, list):
        return [invert(m) for m in reversed(move)]

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
        if moves is None:
            return 0

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

def append_move_cancel(moves: List[str], move: Move):
    cancels = move.count_pre_cancels(moves)
    if cancels == 0:
        return moves + move.moves
    else:
        return moves[:-cancels] + move.moves[cancels:]

class Commutator(Move):
    def __init__(self, name, puzzle_moves, move_map=None):
        # Type 1: [X,Y]

        reverse_X = name[0] == "]"
        reverse_Y = name[-1] == "["

        delimiter = '|' if '|' in name else ' '
        split_idx = name.find(",")
        X = name[1:split_idx].split(delimiter)
        Y = name[split_idx + 1:-1].split(delimiter)
        X_inv = list(map(invert, reversed(X)))
        Y_inv = list(map(invert, reversed(Y)))

        if reverse_X:
            X_inv = list(reversed(X_inv))
        if reverse_Y:
            Y_inv = list(reversed(Y_inv))

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
        conjugate_re = re.compile("\\([^,],(.*)\\)")
        comm = conjugate_re.match(name)
        split_idx = name.find(",")
        setup = name[1:split_idx].split("|")
        comm_name = name[split_idx + 1:-1]
        commutator = Commutator(comm_name, puzzle_moves, move_map)

        self.name = name
        self.moves = setup + commutator.moves + list(map(invert, reversed(setup)))
        self.move = puzzle_moves[setup[0]]
        for i in range(1, len(self.moves)):
            self.move = self.move[puzzle_moves[self.moves[i]]]

        self.moves_named = ".".join(self.moves)
        self.length = len(self.moves)
        self.num_wrong = count_wrong(self.move)


def read_conjugates(moves_file, moves, move_map=None, max_wrong=5, announce_skip=True):
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
                if announce_skip:
                    print(f"Skipping {move.name} because it commutes too many pieces. Commutes {move.num_wrong} > {max_wrong}.")
                continue

            conjugates.append(move)
            all_move_set.add(tuple(move.move))

    return conjugates

def create_commutators(commutator_file, moves, move_map=None, max_wrong=5, announce_skip=True):
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
                if announce_skip:
                    print(f"Skipping {commutator.name} because it commutes too many pieces. Commutes {commutator.num_wrong} > {max_wrong}.")
                continue

            commutators.append(commutator)
            all_move_set.add(tuple(commutator.move))

    return commutators

def create_algs(alg_file, moves, move_map=None, max_wrong=5, announce_skip=True):
    algs = []
    identity = np.arange(len(moves[list(moves.keys())[0]]))

    all_move_set = set()
    with open(alg_file, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue

            alg_moves = line.split(" ")
            if move_map:
                alg_moves = list(map(lambda x: move_map[x], alg_moves))
            move = moves[alg_moves[0]]
            for i in range(1, len(alg_moves)):
                for m in alg_moves[i].split("."):
                    move = move[moves[m]]

            alg = Move(line, move, alg_moves)

            # Skip algs which are the identity
            if (alg.move == identity).all():
                continue

            # Skip algs which are the same as any existing alg
            if tuple(alg.move) in all_move_set:
                continue

            if alg.num_wrong > max_wrong:
                if announce_skip:
                    print(f"Skipping {alg.name} because it commutes too many pieces. Commutes {alg.num_wrong} > {max_wrong}.")
                continue

            algs.append(alg)
            all_move_set.add(tuple(alg.move))

    return algs

def get_setup_moves(moves, max_setup_moves):
    setup_move_set = set()

    # Add the identity
    identity = np.arange(len(moves[list(moves.keys())[0]]))
    setup_move_set.add(tuple(identity))

    inverter = lambda x: x if "." in x else invert(x)

    setups = []

    for num_setup_moves in range(1, max_setup_moves + 1):
        for setup_moves in itertools.product(moves.keys(), repeat=num_setup_moves):
            setup_moves = list(setup_moves)
            setup_move = moves[setup_moves[0]]
            for i in range(1, len(setup_moves)):
                setup_move = setup_move[moves[setup_moves[i]]]

            if tuple(setup_move) in setup_move_set:
                continue

            setup_move_set.add(tuple(setup_move))

            if count_wrong(setup_move) == 0:
                continue

            setup_moves_inv = list(map(inverter, reversed(setup_moves)))
            setup_move_inv = moves[setup_moves_inv[0]]
            for i in range(1, len(setup_moves_inv)):
                setup_move_inv = setup_move_inv[moves[setup_moves_inv[i]]]

            setup = Move("|".join(setup_moves), setup_move, setup_moves)
            setup_inv = Move("|".join(setup_moves_inv), setup_move_inv, setup_moves_inv)

            setups.append((setup, setup_inv))

    return setups

def create_conjugates(commutators, moves, max_setup_moves=1, max_additional_pieces_changed=2, max_wrong=5):
    global identity
    identity = np.arange(len(moves[list(moves.keys())[0]]))

    all_move_set = set()
    all_move_set.add(tuple(identity))
    for commutator in commutators:
        all_move_set.add(tuple(commutator.move))

    setup_moves = get_setup_moves(moves, max_setup_moves)
    print(f"Number of setup moves: {len(setup_moves)}")

    conjugates = []
    for commutator in commutators:
        for setup, setup_inv in setup_moves:
            # Skip setup moves which cancel with the commutator
            # if commutator.count_pre_cancels(setup_moves) > 0:
            #     continue

            conjugate_moves = setup.moves + commutator.moves + setup_inv.moves
            conjugate_move = setup.move[commutator.move][setup_inv.move]

            # Skip conjugates which are the same as any existing commutator or conjugate
            if tuple(conjugate_move) in all_move_set:
                continue

            all_move_set.add(tuple(conjugate_move))

            # Format conjugate names as (setup,commutator)
            conjugate = Move(f"({setup.name},{commutator.name})", conjugate_move, conjugate_moves)

            if conjugate.num_wrong > commutator.num_wrong + max_additional_pieces_changed:
                continue

            if conjugate.num_wrong > max_wrong:
                continue

            conjugates.append(conjugate)

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

def print_wrong_stickers(initial_state, solution_state):
    for i in range(len(solution_state)):
        if solution_state[i] != initial_state[i]:
            print(f"\tSticker {i}: {initial_state[i]} -> {solution_state[i]}")

def identify_cycles(initial_state, solution_state):
    # Find the cycles
    piece_to_cycle = {}
    cycle = 0
    for i in range(len(solution_state)):
        if solution_state[i] != initial_state[i]:
            if i in piece_to_cycle:
                continue

            piece_to_cycle[i] = cycle
            j = int(initial_state[i][1:])
            while j != i:
                piece_to_cycle[j] = cycle
                j = int(initial_state[j][1:])
            
            cycle += 1

    return piece_to_cycle

def decompose(solution_state, moves):
    piece_to_set = {i: i for i in range(len(solution_state))}

    def change_all_sets(old_set, new_set):
        for i in range(len(solution_state)):
            if piece_to_set[i] == old_set:
                piece_to_set[i] = new_set

    for i in range(len(solution_state)):
        for name, move in moves.items():
            new_piece = move[i]
            if piece_to_set[i] != piece_to_set[new_piece]:
                merged_set = min(piece_to_set[i], piece_to_set[new_piece])
                change_all_sets(piece_to_set[i], merged_set)
                change_all_sets(piece_to_set[new_piece], merged_set)

    # Print each set
    sets = {}
    for i in range(len(solution_state)):
        set = piece_to_set[i]
        if set not in sets:
            sets[set] = []
        sets[set].append(i)

    for set, pieces in sets.items():
        print(f"{set}: {pieces}")

    # Assign each piece an index in the set
    piece_to_set_index = {}
    for set, pieces in sets.items():
        for i, piece in enumerate(pieces):
            piece_to_set_index[piece] = i + 1

    # Convert solution state pieces within each set to indexes
    set_to_sol_piece_to_index = {}
    set_to_last_index = {}

    for i, piece in enumerate(solution_state):
        set_num = piece_to_set[i]
        if set_num not in set_to_sol_piece_to_index:
            set_to_sol_piece_to_index[set_num] = {}
            set_to_last_index[set_num] = 1
        if piece not in set_to_sol_piece_to_index[set_num]:
            set_to_sol_piece_to_index[set_num][piece] = str(set_to_last_index[set_num])
            set_to_last_index[set_num] += 1

    return sets, piece_to_set_index, set_to_sol_piece_to_index

def reskin(state, edges, edge_map, odd_centers, odd_center_reskin_map):
    new_state = state.copy()

    for edge in edges:
        print(edge, state[edge[0]] + state[edge[1]])
        new_state[edge[0]] = edge_map[state[edge[0]] + state[edge[1]]][0]
        new_state[edge[1]] = edge_map[state[edge[0]] + state[edge[1]]][1]
        print(new_state[edge[0]] + new_state[edge[1]])
    
    for odd_center in odd_centers:
        new_state[odd_center] = odd_center_reskin_map[state[odd_center]]

    return new_state
