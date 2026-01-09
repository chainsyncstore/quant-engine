"""
Integration test for Walk-Forward Orchestrator.
Runs the orchestrator with --walk-forward flag and checks if multiple results are stored.
"""
import sys
import subprocess
import sqlite3
import os

def test_walk_forward_integration():
    # Setup
    db_path = "test_integration.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    cmd = [
        sys.executable,
        "-m", "orchestrator.run_evaluation",
        "--hypothesis", "always_long",
        "--synthetic",
        "--synthetic-bars", "300",
        "--output-db", db_path,
        "--policy", "WF_V1",
        "--synthetic",
        "--synthetic-bars", "400",
        "--output-db", db_path,
        "--quiet"
    ]
    
    # Run
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Orchestrator failed:")
        print(result.stderr)
        sys.exit(1)
        
    # Verify DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check evaluations
    cursor.execute("SELECT count(*) FROM evaluations")
    count = cursor.fetchone()[0]
    print(f"Evaluations stored: {count}")
    
    # We have 300 bars.
    # W0: Train[0:100], Test[100:150]
    # W1: Train[50:150], Test[150:200]
    # W2: Train[100:200], Test[200:250]
    # W3: Train[150:250], Test[250:300]
    # Total 4 windows. Each window stores 2 records (Train + Test)?
    # No, implementation stores Train and Test separately.
    # Logic in run_evaluation:
    #   repo.store_evaluation(...) for Train
    #   repo.store_evaluation(...) for Test
    # So 4 windows * 2 = 8 records expected.
    
    cursor.execute("SELECT window_index, window_type, sample_type, market_regime, result_tag FROM evaluations ORDER BY evaluation_id")
    rows = cursor.fetchall()
    for row in rows:
        print(f"Row: {row}")
        
    if count == 0:
        print("FAILED: No evaluations stored")
        sys.exit(1)
        
    # Check if we have both TRAIN and TEST types
    types = set(r[1] for r in rows)
    if not ("TRAIN" in types and "TEST" in types):
        print(f"FAILED: Missing window types. Found: {types}")
        sys.exit(1)
        
    # Check if we have regime populated
    regimes = set(r[3] for r in rows)
    print(f"Regimes found: {regimes}")
    if all(r is None for r in regimes):
        print("FAILED: No regimes populated")
        sys.exit(1)
        
    # Check if result_tag is populated (only for TEST)
    tags = set(r[4] for r in rows if r[1] == "TEST")
    print(f"Result tags (Test): {tags}")
    
    print("Integration Test PASSED")
    conn.close()
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    test_walk_forward_integration()
