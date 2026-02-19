
import os
from cryptography.fernet import Fernet

class CryptoManager:
    def __init__(self, key: str = None):
        # In Docker, this will come from env. For local dev, we might generate one temporarily.
        self.key = key or os.getenv("BOT_MASTER_KEY")
        if not self.key:
             # Just a warning or raise, but for now allow init for testing if passed manually
             pass

    def get_cipher(self):
        if not self.key:
             raise ValueError("BOT_MASTER_KEY environment variable is not set")
        return Fernet(self.key.encode())

    def encrypt(self, text: str) -> str:
        if not text: return None
        return self.get_cipher().encrypt(text.encode()).decode()

    def decrypt(self, token: str) -> str:
        if not token: return None
        return self.get_cipher().decrypt(token.encode()).decode()

    @staticmethod
    def generate_key() -> str:
        return Fernet.generate_key().decode()
