#!/usr/bin/env python3
import pandas as pd
import json

# File paths
base_path = '~/Documents/Programming/santa-2023'
puzzle_info_path = f'{base_path}/data/puzzle_info.csv'
puzzles_path = f'{base_path}/data/puzzles.csv'
sample_submission_path = f'{base_path}/data/sample_submission.csv'

# Loading the data
puzzle_info_df = pd.read_csv(puzzle_info_path)
puzzles_df = pd.read_csv(puzzles_path)
sample_submission_df = pd.read_csv(sample_submission_path)

# Displaying the first few rows of each dataframe
print(puzzle_info_df.head())

# Parsing the initial_state and solution_state columns
# Converting the semicolon-separated string values into lists of colors
puzzles_df['parsed_initial_state'] = puzzles_df['initial_state'].apply(lambda x: x.split(';'))
puzzles_df['parsed_solution_state'] = puzzles_df['solution_state'].apply(lambda x: x.split(';'))

# Displaying the modified dataframe with parsed states
puzzles_df[['id', 'puzzle_type', 'parsed_initial_state', 'parsed_solution_state']].head()

# Converting the string representation of allowed_moves to dictionary
puzzle_info_df['allowed_moves'] = puzzle_info_df['allowed_moves'].apply(lambda x: json.loads(x.replace("'", '"')))

# Selecting an example puzzle type and displaying its allowed moves
example_puzzle_type = puzzle_info_df['puzzle_type'].iloc[0]
example_allowed_moves = puzzle_info_df[puzzle_info_df['puzzle_type'] == example_puzzle_type]['allowed_moves'].iloc[0]

print(example_puzzle_type)

print(pd.DataFrame(example_allowed_moves))
