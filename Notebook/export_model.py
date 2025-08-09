import os
import importlib.util

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(THIS_DIR, 'Env for playing!.py')

ASSETS_DIR = os.path.join(os.path.dirname(THIS_DIR), 'App', 'ChessMaster3D', 'assets', 'models')
OUT_BASENAME = 'chess_model'

os.makedirs(ASSETS_DIR, exist_ok=True)

# Load the module with spaces/special chars in filename
spec = importlib.util.spec_from_file_location("env_chess_ai", ENV_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore

# Build a fresh model and export TorchScript/ONNX if available
model = mod.ChessNet()  # untrained stub
export_path = os.path.join(ASSETS_DIR, OUT_BASENAME)
mod.FlutterExporter.export_pytorch_model(model, export_path)

print(f"Exported model to: {export_path}.pt")
