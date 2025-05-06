import torch
import chess
import numpy as np
from chess_gpt import ChessGPT, ChessGPTConfig
from chess_tokenizer import load_tokenizer, tokenize_game
from helpers import load_latest_checkpoint

class ChessRewardFunction:
    """Reward function for chess games"""
    
    @staticmethod
    def calculate_reward(board: chess.Board) -> float:
        """Calculate reward ensuring winner has higher material score"""
        
        # First calculate material balance
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0
        }
        
        material_score = 0.0
        for piece_type in piece_values:
            material_score += (
                len(board.pieces(piece_type, chess.WHITE)) -
                len(board.pieces(piece_type, chess.BLACK))
            ) * piece_values[piece_type]

        # Check game outcome
        if board.is_game_over():
            outcome = board.outcome()
            if outcome is None:
                return 0.0
            
            if outcome.winner is None:  # Draw
                return 0.0
            elif outcome.winner:  # White wins
                # Ensure positive score for white win
                return max(1.0, material_score + 15.0)
            else:  # Black wins
                # Ensure negative score for black win
                return min(-1.0, material_score - 15.0)

        # For ongoing games, use the existing scoring system
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        position_score = sum(1.0 for sq in center_squares if board.piece_at(sq))
        
        def king_safety(color):
            king_square = board.king(color)
            if king_square is None:
                return 0.0
            safety = 0.0
            for sq in chess.SQUARES:
                if chess.square_distance(king_square, sq) <= 2:
                    if not board.is_attacked_by(not color, sq):
                        safety += 0.1
            return safety

        king_safety_score = king_safety(chess.WHITE) - king_safety(chess.BLACK)
        
        # Combine scores with weights
        total_reward = (
            0.5 * np.tanh(0.2 * material_score) +
            0.3 * np.tanh(0.3 * position_score) +
            0.2 * np.tanh(0.5 * king_safety_score)
        )
        
        return total_reward

class SelfPlayTrainer:
    def __init__(self, model, device, token_to_idx, idx_to_token):
        self.model = model
        self.device = device
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        self.reward_function = ChessRewardFunction()

    def generate_move(self, board, color, temperature=1.0, top_k=5):
        # Convert move stack to PGN moves
        moves = []
        temp_board = chess.Board()
        for move in board.move_stack:
            san_move = temp_board.san(move)
            moves.append(san_move)
            temp_board.push(move)

        move_str = ' '.join(moves)
        tokens = tokenize_game(move_str, color, self.token_to_idx)
        
        # Calculate required padding
        block_size = 128
        pad_token = self.token_to_idx.get('[PAD]', 0)
        bog_token = self.token_to_idx.get('[BOG]', 0)
        
        # Add padding before BOG token
        content_length = len(tokens) + 1  # +1 for BOG token
        pad_length = block_size - content_length
        if pad_length > 0:
            tokens = [pad_token] * pad_length + [bog_token] + tokens
        else:
            # If sequence is too long, truncate and ensure BOG is first non-pad token
            tokens = tokens[-(block_size-1):]  # Leave space for BOG
            tokens = [bog_token] + tokens

        print(f"Sequence length after padding: {len(tokens)}")
        print("Token sequence starts with: [PAD, PAD, ..., BOG, first_move, ...]")
        
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)


        with torch.no_grad():
            logits, _ = self.model(tokens)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Get top k predictions
            top_probs, top_indices = torch.topk(probs[0], k=top_k)
            candidate_tokens = [self.idx_to_token[idx.item()] for idx in top_indices]
            
            print(f"\nTop {top_k} predictions and their probabilities:")
            for token, prob in zip(candidate_tokens, top_probs):
                print(f"{token}: {prob.item():.4f}")

            # First check for game ending tokens in top predictions
            for token in candidate_tokens:
                if token in ['[WWIN]', '[BWIN]', '[DRAW]']:
                    print(f"Game ending token found: {token}")
                    if token == '[WWIN]':
                        return 'white_wins'
                    elif token == '[BWIN]':
                        return 'black_wins'
                    else:
                        return 'draw'

            # Try each predicted move in order of probability
            for token, prob in zip(candidate_tokens, top_probs):
                try:
                    print(f"\nTrying move: {token} (probability: {prob.item():.4f})")
                    move = board.parse_san(token)
                    if move in board.legal_moves:
                        print(f"Legal move found: {token}")
                        return move
                    else:
                        print(f"Move {token} is not legal in current position")
                except ValueError as e:
                    print(f"Error parsing move {token}: {e}")
                    continue  # Try next candidate

            # If no legal moves found in top k, fall back to random
            print("\nNo legal moves found in top predictions, falling back to random move")
            return np.random.choice(list(board.legal_moves))

    def play_game(self):
        board = chess.Board()
        game_moves = []
        game_outcome = None

        while not board.is_game_over():
            color = 'white' if board.turn == chess.WHITE else 'black'
            result = self.generate_move(board, color)
            
            # Check if game ending token was generated
            if isinstance(result, str):  # Game ending token
                if result == 'white_wins':
                    print("Game ended: White wins")
                    game_outcome = (chess.WHITE, "White wins by [WWIN]")
                elif result == 'black_wins':
                    print("Game ended: Black wins")
                    game_outcome = (chess.BLACK, "Black wins by [BWIN]")
                else:  # draw
                    print("Game ended: Draw")
                    game_outcome = (None, "Draw by [DRAW]")
                break
            
            # Normal move
            board.push(result)
            game_moves.append(result)

        return board, game_moves, game_outcome

    def train_step(self, optimizer, games=100, gamma=0.99):
        total_reward = 0
        total_loss = 0

        for _ in range(games):
            board, moves, game_outcome = self.play_game()  # Now unpacking all three values

            # Skip if no moves were made
            if not moves:
                continue

            rewards = []
            for i in range(len(moves)):
                temp_board = chess.Board()
                for move in moves[:i + 1]:
                    temp_board.push(move)
                reward = self.reward_function.calculate_reward(temp_board)
                rewards.append(reward)

            # Add final reward based on game outcome
            if game_outcome:
                winner, reason = game_outcome
                if winner == chess.WHITE:
                    rewards[-1] = max(15.0, rewards[-1])  # Ensure positive reward for white win
                elif winner == chess.BLACK:
                    rewards[-1] = min(-15.0, rewards[-1])  # Ensure negative reward for black win
                else:  # Draw
                    rewards[-1] = 0.0

            # Discount rewards
            discounted_rewards = []
            running_reward = 0
            for r in reversed(rewards):
                running_reward = r + gamma * running_reward
                discounted_rewards.insert(0, running_reward)

            rewards = torch.tensor(discounted_rewards).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Split moves into white and black turns
            white_moves = moves[::2]
            black_moves = moves[1::2]

            # Train from white perspective
            if white_moves:
                temp_board = chess.Board()
                white_pgn_moves = []
                for move in white_moves:
                    white_pgn_moves.append(temp_board.san(move))
                temp_board.push(move)
                white_str = ' '.join(white_pgn_moves)
                white_tokens = tokenize_game(white_str, 'white', self.token_to_idx)
                white_tokens = torch.tensor(white_tokens, dtype=torch.long).unsqueeze(0).to(self.device)

                white_logits, white_loss = self.model(white_tokens)
                white_loss = white_loss * rewards[::2].mean()

                optimizer.zero_grad()
                white_loss.backward()
                optimizer.step()

                total_loss += white_loss.item()
                total_reward += rewards[::2].mean().item()

            # Train from black perspective
            if black_moves:
                temp_board = chess.Board()
                black_pgn_moves = []
                for move in black_moves:
                    black_pgn_moves.append(temp_board.san(move))
                    temp_board.push(move)
                black_str = ' '.join(black_pgn_moves)
                black_tokens = tokenize_game(black_str, 'black', self.token_to_idx)
                black_tokens = torch.tensor(black_tokens, dtype=torch.long).unsqueeze(0).to(self.device)

                black_logits, black_loss = self.model(black_tokens)
                black_loss = black_loss * rewards[1::2].mean()

                optimizer.zero_grad()
                black_loss.backward()
                optimizer.step()

                total_loss += black_loss.item()
                total_reward += rewards[1::2].mean().item()

        # Normalize over games * 2 (since we train on both sides)
        return total_loss / (games * 2), total_reward / (games * 2)

def train_rl():
    token_to_idx, idx_to_token, _ = load_tokenizer()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')

    config = ChessGPTConfig(
        vocab_size=len(token_to_idx),
        block_size=128,
        n_embd=384,
        n_head=6,
        n_layer=6,
        dropout=0.1
    )
    
    model = ChessGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Try to load latest checkpoint
    checkpoint, model, optimizer = load_latest_checkpoint('.', model, optimizer, device=device)
    model = model.to(device)

    trainer = SelfPlayTrainer(model, device, token_to_idx, idx_to_token)
    
    num_epochs = 100
    games_per_epoch = 10

    for epoch in range(num_epochs):
        loss, reward = trainer.train_step(optimizer, games=games_per_epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Reward: {reward:.4f}')

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'reward': reward
            }, f'checkpoint_epoch_{epoch+1}.pt')

if __name__ == '__main__':
    train_rl()
