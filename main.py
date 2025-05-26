import sys
import os
import re
import json
import random
import logging
from datetime import datetime
from cryptography.fernet import Fernet, InvalidToken
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.logger import Logger

# Konfigurasi Aplikasi
MEMORY_FILE = "memory.json.enc"
BACKUP_DIR = "backups"
ENCRYPTION_KEY = b'p606UkA3qNf7b8hffYUDDd5e-y0IqN7cL4D66p24Rk4='
cipher = Fernet(ENCRYPTION_KEY)

# === PORTABLE SYSTEM SETUP ===
def get_app_path():
    """Dapatkan path aplikasi yang benar untuk mode portable"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

APP_PATH = get_app_path()
os.chdir(APP_PATH)

# === OPTIMIZED MODEL LOADING ===
class ModelLoader:
    @classmethod
    def load_spacy(cls):
        try:
            import spacy
            model_path = os.path.join(APP_PATH, "id_core_news_sm")
            if os.path.exists(model_path):
                return spacy.load(model_path)
            Logger.warning("SPACY: Model ID tidak ditemukan, coba model EN...")
            return spacy.load("en_core_web_sm")
        except Exception as e:
            Logger.error(f"SPACY Error: {e}")
            return None

    @classmethod
    def load_torch_model(cls):
        try:
            import torch
            model_path = os.path.join(APP_PATH, "search_model.pth")
            if not os.path.exists(model_path):
                Logger.warning("TORCH: Model tidak ditemukan!")
                return None
            model = torch.jit.load(model_path)
            model.eval()
            if torch.cuda.is_available():
                model = model.to('cuda')
                Logger.info("TORCH: Menggunakan CUDA")
            else:
                model = model.to('cpu')
                torch.set_num_threads(2)
                Logger.info("TORCH: Menggunakan CPU dengan 2 thread")
            if not torch.cuda.is_available():
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            return model
        except Exception as e:
            Logger.error(f"TORCH Error: {e}")
            return None

# === CORE AI SYSTEM ===
class BaseAI:
    def __init__(self):
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "rb") as f:
                    return json.loads(decrypt_data(f.read()))
            except Exception as e:
                Logger.error(f"Memory Error: {e}")
        return {}

    def handle_command(self, user_input: str) -> str:
        return "Perintah diterima: " + user_input

class EnhancedAI(BaseAI):
    def __init__(self):
        super().__init__()
        self.nlp = ModelLoader.load_spacy()
        self.model = ModelLoader.load_torch_model()
        self._init_vector_db()

    def _init_vector_db(self):
        self.vector_db = {}
        if self.nlp:
            for topic, info in self.memory.items():
                self.vector_db[topic] = {
                    'topic_vec': self.text_to_vector(topic),
                    'info_vec': self.text_to_vector(info)
                }

    def text_to_vector(self, text: str) -> list:
        if not self.nlp:
            return []
        max_length = 50
        doc = self.nlp(text[:max_length])
        return doc.vector.tolist()

    def semantic_search(self, query: str) -> list:
        return []

class AdaptiveChatUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self._init_ui()
        self._init_ai()

    def _init_ai(self):
        try:
            import spacy
            import torch
            self.ai = EnhancedAI()
            Logger.info("SYSTEM: Mode Enhanced diaktifkan")
        except ImportError:
            self.ai = BaseAI()
            Logger.warning("SYSTEM: Fallback ke mode dasar")

    def _init_ui(self):
        self.text_input = TextInput(size_hint_y=None, height=40, multiline=False)
        self.text_input.bind(on_text_validate=lambda instance: self.process_input())
        self.output = TextInput(readonly=True)
        scroll = ScrollView()
        scroll.add_widget(self.output)
        self.add_widget(scroll)
        self.add_widget(self.text_input)

    def process_input(self):
        query = self.text_input.text.strip()
        if query:
            response = self.ai.handle_command(query)
            self.output.text += f"User: {query}
AI: {response}

"
            self.text_input.text = ""

class PortableApp(App):
    def build(self):
        self.title = "AI Portable Pro"
        self._check_dependencies()
        return AdaptiveChatUI()

    def _check_dependencies(self):
        try:
            import cryptography
            import spacy
            import torch
        except ImportError:
            Logger.warning("DEPENDENCY: Missing packages, running portable setup...")
            self._portable_setup()

    def _portable_setup(self):
        try:
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--target", os.path.join(APP_PATH, "deps"),
                "cryptography", "spacy", "torch"
            ])
            sys.path.append(os.path.join(APP_PATH, "deps"))
        except Exception as e:
            Logger.error(f"SETUP Error: {e}")

def encrypt_data(data: str) -> bytes:
    return cipher.encrypt(data.encode('utf-8'))

def decrypt_data(encrypted_data: bytes) -> str:
    return cipher.decrypt(encrypted_data).decode('utf-8')

if __name__ == "__main__":
    logging.basicConfig(
        filename=os.path.join(APP_PATH, 'ai_portable.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    PortableApp().run()
