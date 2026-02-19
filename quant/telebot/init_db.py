
import os
from sqlalchemy import create_engine
from quant.telebot.models import Base

def init_db(db_path="quant_bot.db"):
    # Ensure correct path relative to cwd if needed, or absolute
    # For now, let's put it in the root or nicely
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    print(f"Database initialized at {os.path.abspath(db_path)}")

if __name__ == "__main__":
    init_db()
