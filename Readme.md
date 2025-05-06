# Chess GPT

A transformer-based model trained to play chess using supervised learning and reinforcement learning. The model learns from a database of 20,000 Lichess games and improves through self-play.

## Overview

This project implements a GPT (Generative Pre-trained Transformer) architecture specifically designed for chess. The model:
- Learns chess moves in PGN (Portable Game Notation) format
- Understands game state and makes legal moves
- Can predict game outcomes ([WWIN], [BWIN], [DRAW])
- Improves through reinforcement learning and self-play

## Architecture

- **Model**: GPT architecture with the following configuration:
  - Vocabulary Size: Based on unique chess moves + special tokens
  - Context Length: 128 tokens
  - Embedding Dimension: 384
  - Attention Heads: 6
  - Transformer Layers: 6
  - Dropout: 0.1

## Training Process

### Phase 1: Supervised Learning
- Initial training on 20,000 Lichess games
- Uses PGN format for move representation
- Learns patterns from human games
- Implemented in `train.py`

### Phase 2: Reinforcement Learning
- Self-play with reward optimization
- Reward function considers:
  - Material advantage
  - Position control
  - King safety
  - Game outcome
- Implemented in `train_rl.py`

## Special Tokens
- `[BOG]`: Beginning of Game
- `[PAD]`: Padding token
- `[WWIN]`: White wins
- `[BWIN]`: Black wins
- `[DRAW]`: Draw

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- python-chess
- numpy
- pandas

## Usage

1. Initial training:
```bash
python train.py
```

2. Reinforcement learning:
```bash
python train_rl.py
```

## Project Structure

```
ChessAI/
├── train.py           # Supervised learning training
├── train_rl.py        # Reinforcement learning training
├── chess_gpt.py       # Model architecture
├── chess_tokenizer.py # Tokenization utilities
├── helpers.py         # Helper functions
└── games.csv         # Lichess games database
```

## Training Output

The model saves checkpoints during training:
- Regular checkpoints every 10 epochs
- Best model saved separately
- Includes model state, optimizer state, and training metrics

## Model Features

- Context-aware move generation
- Probability-based move selection
- Game ending prediction
- Self-improving through RL
- Handles both white and black perspectives

## License

MIT License

Copyright (c) 2024 Swetang Sharma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- Lichess for the games database
- OpenAI GPT architecture
- Python Chess library