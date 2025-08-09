# Chess AI Training Notebook
# Multiple approaches for creating a chess AI with adjustable difficulty levels

import numpy as np
import chess
import chess.engine
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# For VideoChess environment (optional)
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    print("Gymnasium not available. Install with: pip install gymnasium[atari]")
    GYMNASIUM_AVAILABLE = False

print("Chess AI Training Environment Setup Complete!")

# ============================================================================
# APPROACH 1: Traditional Chess AI with Neural Networks
# ============================================================================

class ChessBoard:
    """Enhanced chess board representation for neural networks"""
    
    def __init__(self):
        self.board = chess.Board()
    
    def board_to_array(self) -> np.ndarray:
        """Convert chess board to neural network input (8x8x12 representation)"""
        # 12 channels: 6 piece types × 2 colors
        board_array = np.zeros((8, 8, 12), dtype=np.float32)
        
        piece_idx = {
            chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
            chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                color_offset = 0 if piece.color == chess.WHITE else 6
                channel = piece_idx[piece.piece_type] + color_offset
                board_array[row, col, channel] = 1.0
        
        return board_array
    
    def get_legal_moves_mask(self) -> np.ndarray:
        """Get a mask of legal moves for current position"""
        moves_mask = np.zeros(4096, dtype=np.float32)  # 64*64 possible moves
        
        for move in self.board.legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            move_idx = from_square * 64 + to_square
            moves_mask[move_idx] = 1.0
        
        return moves_mask
    
    def make_move(self, move_str: str):
        """Make a move on the board"""
        try:
            move = chess.Move.from_uci(move_str)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
        except:
            pass
        return False

class ChessNet(nn.Module):
    """Neural network for chess position evaluation and move prediction"""
    
    def __init__(self, difficulty_levels=5):
        super(ChessNet, self).__init__()
        
        # Convolutional layers for board understanding
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Output heads
        self.value_head = nn.Linear(256, 1)  # Position evaluation
        self.policy_head = nn.Linear(256, 4096)  # Move probabilities
        self.difficulty_head = nn.Linear(256, difficulty_levels)  # Difficulty adjustment
        
    def forward(self, x, difficulty_level=None):
        # x shape: (batch, 12, 8, 8)
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        value = torch.tanh(self.value_head(x))
        policy = torch.softmax(self.policy_head(x), dim=1)
        
        # Adjust policy based on difficulty
        if difficulty_level is not None:
            difficulty_weights = torch.softmax(self.difficulty_head(x), dim=1)
            # Apply difficulty adjustment (simplified)
            noise_factor = (5 - difficulty_level) / 5.0  # More noise for lower difficulty
            if noise_factor > 0:
                noise = torch.randn_like(policy) * noise_factor * 0.1
                policy = torch.softmax(torch.log(policy + 1e-8) + noise, dim=1)
        
        return value, policy

class ChessDataset(Dataset):
    """Dataset for training chess neural network"""
    
    def __init__(self, positions: List[Tuple[np.ndarray, float, np.ndarray]]):
        self.positions = positions
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        board_array, value, move_probs = self.positions[idx]
        return (
            torch.FloatTensor(board_array).permute(2, 0, 1),  # CHW format
            torch.FloatTensor([value]),
            torch.FloatTensor(move_probs)
        )

class ChessAI:
    """Main Chess AI class with adjustable difficulty"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = ChessNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        if model_path:
            self.load_model(model_path)
    
    def generate_training_data(self, num_games: int = 1000) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        """Generate training data from random games"""
        training_data = []
        
        print(f"Generating {num_games} games for training...")
        for game_idx in tqdm(range(num_games)):
            board = chess.Board()
            game_positions = []
            
            # Play random game
            while not board.is_game_over() and len(game_positions) < 100:
                chess_board = ChessBoard()
                chess_board.board = board.copy()
                
                # Get board representation
                board_array = chess_board.board_to_array()
                moves_mask = chess_board.get_legal_moves_mask()
                
                game_positions.append((board_array, moves_mask))
                
                # Make random move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    board.push(move)
                else:
                    break
            
            # Assign values based on game outcome
            result = board.result()
            if result == "1-0":  # White wins
                values = [1.0 if i % 2 == 0 else -1.0 for i in range(len(game_positions))]
            elif result == "0-1":  # Black wins
                values = [-1.0 if i % 2 == 0 else 1.0 for i in range(len(game_positions))]
            else:  # Draw
                values = [0.0] * len(game_positions)
            
            # Add to training data
            for (board_array, moves_mask), value in zip(game_positions, values):
                training_data.append((board_array, value, moves_mask))
        
        return training_data
    
    def train(self, training_data: List[Tuple[np.ndarray, float, np.ndarray]], 
              epochs: int = 50, batch_size: int = 32):
        """Train the chess neural network"""
        dataset = ChessDataset(training_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        value_criterion = nn.MSELoss()
        policy_criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        losses = []
        
        print("Training chess neural network...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_boards, batch_values, batch_policies in dataloader:
                batch_boards = batch_boards.to(self.device)
                batch_values = batch_values.to(self.device)
                batch_policies = batch_policies.to(self.device)
                
                optimizer.zero_grad()
                
                pred_values, pred_policies = self.model(batch_boards)
                
                value_loss = value_criterion(pred_values, batch_values)
                policy_loss = value_criterion(pred_policies, batch_policies)  # Using MSE for simplicity
                
                total_loss = value_loss + policy_loss
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def get_best_move(self, board: chess.Board, difficulty: int = 5) -> Optional[chess.Move]:
        """Get the best move for current position with specified difficulty"""
        if not list(board.legal_moves):
            return None
        
        chess_board = ChessBoard()
        chess_board.board = board.copy()
        
        board_array = chess_board.board_to_array()
        board_tensor = torch.FloatTensor(board_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            value, policy = self.model(board_tensor, difficulty_level=difficulty)
            
        # Convert policy to move probabilities
        legal_moves = list(board.legal_moves)
        move_probs = []
        
        for move in legal_moves:
            move_idx = move.from_square * 64 + move.to_square
            prob = policy[0][move_idx].item()
            move_probs.append((move, prob))
        
        # Sort by probability and add randomness based on difficulty
        move_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Adjust selection based on difficulty
        if difficulty >= 4:  # Expert level
            return move_probs[0][0]
        elif difficulty >= 3:  # Advanced level
            top_moves = move_probs[:3]
            weights = [0.7, 0.2, 0.1]
        elif difficulty >= 2:  # Intermediate level
            top_moves = move_probs[:5]
            weights = [0.4, 0.25, 0.2, 0.1, 0.05]
        else:  # Beginner level
            top_moves = move_probs[:min(len(move_probs), 10)]
            weights = [1.0/len(top_moves)] * len(top_moves)
        
        # Weighted random selection
        selected_move = random.choices(top_moves, weights=weights[:len(top_moves)])[0][0]
        return selected_move
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'difficulty_levels': 5
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

# ============================================================================
# APPROACH 2: Stockfish-based AI with Adjustable Strength
# ============================================================================

class StockfishAI:
    """Chess AI using Stockfish engine with adjustable strength"""
    
    def __init__(self, stockfish_path: str = "/usr/local/bin/stockfish"):
        self.stockfish_path = stockfish_path
        self.difficulty_settings = {
            1: {"depth": 1, "time": 0.1, "skill_level": 0},
            2: {"depth": 3, "time": 0.3, "skill_level": 3},
            3: {"depth": 5, "time": 0.5, "skill_level": 7},
            4: {"depth": 8, "time": 1.0, "skill_level": 12},
            5: {"depth": 15, "time": 3.0, "skill_level": 20}
        }
    
    def get_best_move(self, board: chess.Board, difficulty: int = 3) -> Optional[chess.Move]:
        """Get best move using Stockfish with specified difficulty"""
        try:
            settings = self.difficulty_settings.get(difficulty, self.difficulty_settings[3])
            
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                # Configure engine strength
                engine.configure({"Skill Level": settings["skill_level"]})
                
                # Get best move
                result = engine.play(
                    board, 
                    chess.engine.Limit(
                        depth=settings["depth"], 
                        time=settings["time"]
                    )
                )
                return result.move
        except Exception as e:
            print(f"Stockfish error: {e}")
            # Fallback to random move
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves) if legal_moves else None

# ============================================================================
# APPROACH 3: VideoChess Environment (Atari-style)
# ============================================================================

class VideoChessAI:
    """AI for VideoChess Atari environment"""
    
    def __init__(self):
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("Gymnasium not available for VideoChess")
        
        self.env = gym.make("ALE/VideoChess-v5")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Simple CNN for processing video frames
        self.model = self._build_cnn_model()
    
    def _build_cnn_model(self):
        """Build CNN model for processing VideoChess frames"""
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 16, 512),  # Adjust based on actual output size
            nn.ReLU(),
            nn.Linear(512, self.action_space.n)
        )
        return model
    
    def preprocess_observation(self, obs):
        """Preprocess VideoChess observation"""
        # Convert to tensor and normalize
        obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1) / 255.0
        return obs_tensor.unsqueeze(0)
    
    def train_videochess(self, episodes: int = 1000):
        """Train on VideoChess environment using DQN-style approach"""
        print("Training on VideoChess environment...")
        print("Note: This is a complex task and may require extensive training")
        
        # This would need a full DQN implementation
        # Placeholder for training loop
        for episode in range(episodes):
            obs, info = self.env.reset()
            done = False
            
            while not done:
                # Process observation
                obs_tensor = self.preprocess_observation(obs)
                
                # Get action (random for now)
                action = self.env.action_space.sample()
                
                # Take step
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            
            if episode % 100 == 0:
                print(f"Episode {episode}/{episodes} completed")

# ============================================================================
# FLUTTER INTEGRATION UTILITIES
# ============================================================================

class FlutterExporter:
    """Utilities for exporting models for Flutter integration"""
    
    @staticmethod
    def export_pytorch_model(model: ChessNet, export_path: str):
        """Export PyTorch model for Flutter (via ONNX or TorchScript)"""
        try:
            # Export as TorchScript
            model.eval()
            dummy_input = torch.randn(1, 12, 8, 8)
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(f"{export_path}.pt")
            print(f"TorchScript model saved to {export_path}.pt")
            
            # Also save as ONNX if available
            try:
                from torch import onnx as torch_onnx
                torch_onnx.export(
                    model, dummy_input, f"{export_path}.onnx",
                    export_params=True, opset_version=11,
                    do_constant_folding=True,
                    input_names=['board'], output_names=['value', 'policy']
                )
                print(f"ONNX model saved to {export_path}.onnx")
            except ImportError:
                print("ONNX not available")
                
        except Exception as e:
            print(f"Export error: {e}")
    
    @staticmethod
    def create_flutter_config(difficulty_levels: List[int], model_info: dict):
        """Create configuration file for Flutter app"""
        config = {
            "model_info": model_info,
            "difficulty_levels": difficulty_levels,
            "difficulty_descriptions": {
                1: "Beginner - Makes obvious mistakes",
                2: "Easy - Casual play level", 
                3: "Medium - Club player level",
                4: "Hard - Tournament level",
                5: "Expert - Master level"
            },
            "model_format": "torchscript",
            "input_shape": [1, 12, 8, 8],
            "output_shapes": {
                "value": [1, 1],
                "policy": [1, 4096]
            }
        }
        
        with open("flutter_chess_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("Flutter configuration saved to flutter_chess_config.json")

# ============================================================================
# TRAINING AND DEMONSTRATION SCRIPT
# ============================================================================

def main_training_pipeline():
    """Main training pipeline demonstrating all approaches"""
    
    print("=" * 60)
    print("CHESS AI TRAINING PIPELINE")
    print("=" * 60)
    
    # Choose approach
    approach = input("Choose approach (1: Neural Network, 2: Stockfish, 3: VideoChess): ")
    
    if approach == "1":
        print("\n--- Training Neural Network Chess AI ---")
        
        # Initialize AI
        ai = ChessAI()
        
        # Generate training data
        training_data = ai.generate_training_data(num_games=100)  # Start small
        print(f"Generated {len(training_data)} training positions")
        
        # Train model
        losses = ai.train(training_data, epochs=20, batch_size=16)
        
        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        
        # Save model
        ai.save_model("chess_ai_model.pth")
        
        # Export for Flutter
        FlutterExporter.export_pytorch_model(ai.model, "chess_model_export")
        FlutterExporter.create_flutter_config(
            difficulty_levels=[1, 2, 3, 4, 5],
            model_info={"type": "neural_network", "trained_games": 100}
        )
        
        # Test the AI
        print("\n--- Testing AI ---")
        test_board = chess.Board()
        for difficulty in [1, 3, 5]:
            move = ai.get_best_move(test_board, difficulty=difficulty)
            print(f"Difficulty {difficulty}: {move}")
    
    elif approach == "2":
        print("\n--- Setting up Stockfish AI ---")
        
        # Note: Requires Stockfish installation
        stockfish_path = input("Enter Stockfish path (or press Enter for default): ").strip()
        if not stockfish_path:
            stockfish_path = "/usr/local/bin/stockfish"
        
        try:
            ai = StockfishAI(stockfish_path)
            
            # Test different difficulties
            test_board = chess.Board()
            print("\n--- Testing Stockfish at different levels ---")
            for difficulty in [1, 2, 3, 4, 5]:
                move = ai.get_best_move(test_board, difficulty=difficulty)
                print(f"Difficulty {difficulty}: {move}")
            
            # Create Flutter config for Stockfish
            FlutterExporter.create_flutter_config(
                difficulty_levels=[1, 2, 3, 4, 5],
                model_info={"type": "stockfish", "engine_path": stockfish_path}
            )
            
        except Exception as e:
            print(f"Stockfish setup failed: {e}")
            print("Install Stockfish: https://stockfishchess.org/download/")
    
    elif approach == "3":
        print("\n--- VideoChess Environment Training ---")
        
        if GYMNASIUM_AVAILABLE:
            try:
                ai = VideoChessAI()
                print("VideoChess environment initialized")
                print("Warning: This approach requires extensive training and may not be practical")
                
                train = input("Start training? (y/n): ").lower() == 'y'
                if train:
                    ai.train_videochess(episodes=100)  # Short demo
                    
            except Exception as e:
                print(f"VideoChess setup failed: {e}")
                print("Install Atari environments: pip install gymnasium[atari]")
        else:
            print("Gymnasium not available. Install with: pip install gymnasium[atari]")
    
    else:
        print("Invalid choice!")
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)

# Run the training pipeline
if __name__ == "__main__":
    main_training_pipeline()

# ============================================================================
# QUICK USAGE EXAMPLES
# ============================================================================

def quick_examples():
    """Quick examples for different use cases"""
    
    print("\n--- Quick Usage Examples ---")
    
    # Example 1: Simple game with neural network AI
    print("\n1. Neural Network AI Game:")
    ai = ChessAI()
    board = chess.Board()
    
    # Generate some quick training data and train
    training_data = ai.generate_training_data(num_games=10)
    ai.train(training_data, epochs=5)
    
    # Play a few moves
    for i in range(3):
        move = ai.get_best_move(board, difficulty=3)
        if move:
            board.push(move)
            print(f"AI move {i+1}: {move}")
        else:
            break
    
    # Example 2: Stockfish AI (if available)
    print("\n2. Stockfish AI Example:")
    try:
        stockfish_ai = StockfishAI()
        board = chess.Board()
        move = stockfish_ai.get_best_move(board, difficulty=2)
        print(f"Stockfish move: {move}")
    except:
        print("Stockfish not available")
    
    # Example 3: Model export
    print("\n3. Model Export Example:")
    FlutterExporter.create_flutter_config(
        difficulty_levels=[1, 2, 3, 4, 5],
        model_info={"type": "demo", "version": "1.0"}
    )

# Uncomment to run examples
# quick_examples()

print("\n🚀 Chess AI Notebook Ready!")
print("📝 Run main_training_pipeline() to start training")
print("🔧 Use quick_examples() for simple demos")
print("📱 Export models for Flutter with FlutterExporter")