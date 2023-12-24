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

def centers_aligned(state, n):
    n2 = n ** 2
    return \
        state[n2 // 2] == "A" and \
        state[n2 + n2 // 2] == "B" and \
        state[2 * n2  + n2 // 2] == "C" and \
        state[3 * n2 + n2 // 2] == "D" and \
        state[4 * n2 + n2 // 2] == "E" and \
        state[5 * n2 + n2 // 2] == "F"

def extend_move_seq(seq, moves):
    for move in moves:
        yield seq + [move]

def orient_centers(state, moves, n):
    center_slice_moves = get_center_slice_moves(moves, n)
    print("Orienting centers")
    print(center_slice_moves)

    # Try longer sequences of moves if the centers are not aligned
    seqs = [[]]
    new_seq = []
    while not centers_aligned(state, n):
        new_seqs = []
        for seq in seqs:
            for new_seq in extend_move_seq(seq, center_slice_moves.keys()):
                new_state = state.copy()
                for move in new_seq:
                    new_state = new_state[center_slice_moves[move]]
                if centers_aligned(new_state, n):
                    print("Found solution", new_seq)
                    state = new_state
                    break
                else:
                    new_seqs.append(new_seq)
            if centers_aligned(state, n):
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
    for move in "DFR":
        # Number of layers
        for i in range(1, n // 2 + 1):
            if i == 1:
                move_map[f"{move}"] = f"{base_moves[move]}0"
                move_map[f"{move}'"] = f"-{base_moves[move]}0"
                move_map[f"{move}2"] = f"{base_moves[move]}0.{base_moves[move]}0"
            elif i == 2:
                move_map[f"{move}w"] = f"{base_moves[move]}0.{base_moves[move]}1"
                move_map[f"{move}w'"] = f"-{base_moves[move]}0.-{base_moves[move]}1"
                move_map[f"{move}w2"] = f"{base_moves[move]}0.{base_moves[move]}0.{base_moves[move]}1.{base_moves[move]}1"

                # For some reason it also has these
                move_map[f"2{move}w"] = f"{base_moves[move]}0.{base_moves[move]}1"
                move_map[f"2{move}w'"] = f"-{base_moves[move]}0.-{base_moves[move]}1"
                move_map[f"2{move}w2"] = f"{base_moves[move]}0.{base_moves[move]}0.{base_moves[move]}1.{base_moves[move]}1"
            else:
                move_map[f"{i}{move}w"] = ".".join([f"{base_moves[move]}{j}" for j in range(i)])
                move_map[f"{i}{move}w'"] = ".".join([f"-{base_moves[move]}{j}" for j in range(i)])
                move_map[f"{i}{move}w2"] = ".".join([f"{base_moves[move]}{j}" for j in range(i)] + [f"{base_moves[move]}{j}" for j in range(i)])
    for move in "BUL":
        # Number of layers
        for i in range(1, n // 2 + 1):
            if i == 1:
                move_map[f"{move}"] = f"-{base_moves[move]}{n - 1}"
                move_map[f"{move}'"] = f"{base_moves[move]}{n - 1}"
                move_map[f"{move}2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}"
            elif i == 2:
                move_map[f"{move}w"] = f"-{base_moves[move]}{n - 1}.-{base_moves[move]}{n - 2}"
                move_map[f"{move}w'"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}"
                move_map[f"{move}w2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}.{base_moves[move]}{n - 2}"

                # For some reason it also has these
                move_map[f"2{move}w"] = f"-{base_moves[move]}{n - 1}.-{base_moves[move]}{n - 2}"
                move_map[f"2{move}w'"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}"
                move_map[f"2{move}w2"] = f"{base_moves[move]}{n - 1}.{base_moves[move]}{n - 1}.{base_moves[move]}{n - 2}.{base_moves[move]}{n - 2}"
            else:
                move_map[f"{i}{move}w"] = ".".join([f"-{base_moves[move]}{n - 1 - j}" for j in range(i)])
                move_map[f"{i}{move}w'"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in range(i)])
                move_map[f"{i}{move}w2"] = ".".join([f"{base_moves[move]}{n - 1 - j}" for j in range(i)] + [f"{base_moves[move]}{n - 1 - j}" for j in range(i)])
    return move_map

def print_faces(faces, n):
    for face in faces:
        print(face)
        for row in chunks(faces[face], n):
            print("\t", " ".join(row))

def get_edges(n):
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
    edges.extend(list(zip(range((n-1) * n + 1, n2 - 1), range(n2 + 1, n2 + n - 1)))[::2])

    # UR edge:
    #  U: [(n-1) + n * i for i in range(1, n - 1)] (Reversed)
    #  R: [2 * n2 + 1, 2 * n2 + n - 2]
    edges.extend(list(zip([(n-1) + n * i for i in range(1, n - 1)][::-1], range(2 * n2 + 1, 2 * n2 + n - 1)))[::2])

    # UB edge:
    #  U: [1, n - 2] (Reversed)
    #  B: [3 * n2 + 1, 3 * n2 + n - 2]
    edges.extend(list(zip(range(1, n - 1)[::-1], range(3 * n2 + 1, 3 * n2 + n - 1)))[::2])

    # UL edge:
    #  U: [n * i for i in range(1, n - 1)]
    #  L: [4 * n2 + 1, 4 * n2 + n - 2]
    edges.extend(list(zip([n * i for i in range(1, n - 1)], range(4 * n2 + 1, 4 * n2 + n - 1)))[::2])

    # FR edge:
    #  F: [n2 + (n-1) + n * i for i in range(1, n - 1)]
    #  R: [2 * n2 + n * i for i in range(1, n - 1)]
    edges.extend(list(zip([n2 + (n-1) + n * i for i in range(1, n - 1)], [2 * n2 + n * i for i in range(1, n - 1)]))[::2])

    # RB edge:
    #  R: [2 * n2 + (n-1) + n * i for i in range(1, n - 1)]
    #  B: [3 * n2 + n * i for i in range(1, n - 1)]
    edges.extend(list(zip([2 * n2 + (n-1) + n * i for i in range(1, n - 1)], [3 * n2 + n * i for i in range(1, n - 1)]))[::2])

    # BL edge:
    #  B: [3 * n2 + (n-1) + n * i for i in range(1, n - 1)]
    #  L: [4 * n2 + n * i for i in range(1, n - 1)]
    edges.extend(list(zip([3 * n2 + (n-1) + n * i for i in range(1, n - 1)], [4 * n2 + n * i for i in range(1, n - 1)]))[::2])

    # LF edge:
    #  L: [4 * n2 + (n-1) + n * i for i in range(1, n - 1)]
    #  F: [n2 + n * i for i in range(1, n - 1)]
    edges.extend(list(zip([4 * n2 + (n-1) + n * i for i in range(1, n - 1)], [n2 + n * i for i in range(1, n - 1)]))[::2])

    # DF edge:
    #  D: [5 * n2 + 1, 5 * n2 + n - 2]
    #  F: [n2 + (n-1) * n + 1, n2 + n2 - 2]
    edges.extend(list(zip(range(5 * n2 + 1, 5 * n2 + n - 1), range(n2 + (n-1) * n + 1, n2 + n2 - 1)))[::2])

    # DR edge:
    #  D: [5 * n2 + (n-1) + n * i for i in range(1, n - 1)]
    #  R: [2 * n2 + (n-1) * n + 1, 2 * n2 + n2 - 2]
    edges.extend(list(zip([5 * n2 + (n-1) + n * i for i in range(1, n - 1)], range(2 * n2 + (n-1) * n + 1, 2 * n2 + n2 - 1)))[::2])

    # DB edge:
    #   D: [5 * n2 + (n-1) * n + 1, 5 * n2 + n2 - 2] (Reversed)
    #   B: [3 * n2 + (n-1) * n + 1, 3 * n2 + n2 - 2]
    edges.extend(list(zip(range(5 * n2 + (n-1) * n + 1, 5 * n2 + n2 - 1)[::-1], range(3 * n2 + (n-1) * n + 1, 3 * n2 + n2 - 1)))[::2])

    # DL edge:
    #   D: [5 * n2 + n * i for i in range(1, n - 1)] (Reversed)
    #   L: [4 * n2 + (n-1) * n + 1, 4 * n2 + n2 - 2]
    edges.extend(list(zip([5 * n2 + n * i for i in range(1, n - 1)][::-1], range(4 * n2 + (n-1) * n + 1, 4 * n2 + n2 - 1)))[::2])

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

        if reskin in edge_map and edge_map[reskin] != normal:
            print("WARNING: Duplicate edge", reskin, normal)

        edge_map[reskin] = normal
        print(reskin, normal)

    # Insert the reverse of each edge
    for edge in list(edge_map.keys()):
        edge_map[edge[::-1]] = edge_map[edge][::-1]

    return edge_map

def make_odd_center_reskin_map(odd_centers, reskin_solution, normal_solution):
    odd_center_map = {}
    for center in odd_centers:
        odd_center_map[reskin_solution[center]] = normal_solution[center]
    return odd_center_map
