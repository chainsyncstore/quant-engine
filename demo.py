"""
Quick demo script to test the system without pytest.

Run this after installing dependencies.
"""

from orchestrator.run_evaluation import run_evaluation

def main():
    print("Running demonstration evaluation...")
    print("=" * 70)
    
    # Run a simple evaluation with synthetic data
    result = run_evaluation(
        hypothesis_id="always_long",
        use_synthetic=True,
        synthetic_bars=252,  # One year of daily bars
        symbol="DEMO",
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("DEMO RESULTS SUMMARY")
    print("=" * 70)
    print(f"Evaluation ID: {result['evaluation_id']}")
    print(f"Database: {result['db_path']}")
    print("\nMetrics:")
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nBenchmark:")
    print(f"  Buy & Hold Return: {result['benchmark_metrics']['benchmark_return_pct']:.2f}%")
    print(f"  Alpha: {result['metrics']['total_return'] - result['benchmark_metrics']['benchmark_return_pct']:.2f}%")


if __name__ == "__main__":
    main()
