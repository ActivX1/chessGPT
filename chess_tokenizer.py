import json
import os
import pandas as pd
from collections import Counter

def save_tokenizer(token_to_idx, idx_to_token, move_counts, save_dir='./'):
    # Save token mappings
    with open(f'{save_dir}token_to_idx.json', 'w') as f:
        json.dump(token_to_idx, f)
    
    with open(f'{save_dir}idx_to_token.json', 'w') as f:
        json.dump({str(k): v for k, v in idx_to_token.items()}, f)  # Convert int keys to str
    
    # Save move statistics
    with open(f'{save_dir}move_counts.json', 'w') as f:
        json.dump({k: v for k, v in move_counts.items()}, f)

def load_tokenizer(load_dir='./'):
    with open(f'{load_dir}token_to_idx.json', 'r') as f:
        token_to_idx = json.load(f)
    
    with open(f'{load_dir}idx_to_token.json', 'r') as f:
        idx_to_token = {int(k): v for k, v in json.load(f).items()}  # Convert str keys back to int
    
    with open(f'{load_dir}move_counts.json', 'r') as f:
        move_counts = Counter(json.load(f))
    
    return token_to_idx, idx_to_token, move_counts


def create_chess_move_tokenizer():
    # Read the CSV file
    df = pd.read_csv('games.csv')
    
    # Get all moves from the 'moves' column and split them
    all_moves = ['[PAD]']
    for index, row in df.iterrows():
        game_moves = row['moves'].split()
        all_moves.append('[BOG]')  # Add beginning token
        all_moves.extend(game_moves)
        # Add appropriate end token based on winner
        if row['winner'] == 'white':
            all_moves.append('[WWIN]')
        elif row['winner'] == 'black':
            all_moves.append('[BWIN]')
        else:  # draw
            all_moves.append('[DRAW]')
    
    # Get unique moves and create vocabulary
    move_counts = Counter(all_moves)
    unique_moves = sorted(move_counts.keys()) # Sort for consistency
    
    # Define special tokens order
    special_tokens = ['[BOG]', '[WWIN]', '[BWIN]', '[DRAW]', '[PAD]']
    # Remove special tokens from unique_moves if present
    unique_moves = [move for move in unique_moves if move not in special_tokens]
    # Add special tokens at the beginning
    unique_moves = special_tokens + unique_moves
    
    # Create token to index mapping
    token_to_idx = {token: idx for idx, token in enumerate(unique_moves)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    
    # Print vocabulary statistics
    print(f"Vocabulary size: {len(unique_moves)}")
    print("\nMost common moves:")
    for move, count in move_counts.most_common(10):
        print(f"{move}: {count}")
    
    return token_to_idx, idx_to_token, move_counts

def tokenize_game(game_moves, winner, token_to_idx):
    """
    Tokenize a single game
    Args:
        game_moves: string of space-separated moves
        winner: 'white', 'black', or 'draw'
        token_to_idx: dictionary mapping tokens to indices
    """
    moves = game_moves.split()
    tokens = [token_to_idx['[BOG]']]  # Start with BOG token
    tokens.extend([token_to_idx[move] for move in moves])
    
    # Add appropriate end token
    if winner == 'white':
        tokens.append(token_to_idx['[WWIN]'])
    elif winner == 'black':
        tokens.append(token_to_idx['[BWIN]'])
    else:
        tokens.append(token_to_idx['[DRAW]'])
    
    return tokens

if __name__ == "__main__":
    # Check if tokenizer files exist
    if not all(os.path.exists(f) for f in ['token_to_idx.json', 'idx_to_token.json', 'move_counts.json']):
        print("Creating new tokenizer...")
        token_to_idx, idx_to_token, move_counts = create_chess_move_tokenizer()
        save_tokenizer(token_to_idx, idx_to_token, move_counts)
        print("\nTokenizer saved to JSON files")
    else:
        print("Loading existing tokenizer...")
        token_to_idx, idx_to_token, move_counts = load_tokenizer()
        print("Tokenizer loaded successfully")

    # Example: Tokenize a single game
    df = pd.read_csv('games.csv')
    sample_game = df['moves'].iloc[0]
    sample_winner = df['winner'].iloc[0]
    tokens = tokenize_game(sample_game, sample_winner, token_to_idx)
    
    print("\nExample tokenization:")
    print("Original moves:", sample_game)
    print("Winner:", sample_winner)
    print("Tokens:", tokens)
    print("Decoded:", [idx_to_token[idx] for idx in tokens])