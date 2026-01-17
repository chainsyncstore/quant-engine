"""Promote crypto hypotheses for competition mode."""
import sqlite3

conn = sqlite3.connect('results/research.db')
cur = conn.cursor()

hypotheses = [
    ('crypto_momentum_breakout', 'PROMOTED', 'WF_V1', 'CRYPTO_BOOTSTRAP', '["manual promotion for crypto competition mode"]'),
    ('rsi_extreme_reversal', 'PROMOTED', 'WF_V1', 'CRYPTO_BOOTSTRAP', '["manual promotion for crypto competition mode"]'),
    ('volatility_expansion_assault', 'PROMOTED', 'WF_V1', 'CRYPTO_BOOTSTRAP', '["manual promotion for crypto competition mode"]'),
]

for h in hypotheses:
    cur.execute(
        """
        INSERT INTO hypothesis_status_history (
            hypothesis_id,
            status,
            policy_id,
            batch_id,
            decision_timestamp,
            rationale_json
        ) VALUES (?, ?, ?, ?, datetime('now'), ?)
        """,
        h
    )
    print(f"Promoted: {h[0]}")

conn.commit()
conn.close()
print("\nAll crypto hypotheses promoted for WF_V1!")
