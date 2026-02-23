import os
from cryptography.fernet import Fernet


class CryptoManager:
    def __init__(self, key: str | None = None):
        self.key = key or os.getenv("BOT_MASTER_KEY")
        if not self.key:
            raise ValueError("BOT_MASTER_KEY environment variable is not set")

        # Fail fast on malformed key values.
        Fernet(self.key.encode())

    def get_cipher(self) -> Fernet:
        return Fernet(self.key.encode())

    def encrypt(self, text: str) -> str | None:
        if not text:
            return None
        return self.get_cipher().encrypt(text.encode()).decode()

    def decrypt(self, token: str) -> str | None:
        if not token:
            return None
        return self.get_cipher().decrypt(token.encode()).decode()

    @staticmethod
    def generate_key() -> str:
        return Fernet.generate_key().decode()
