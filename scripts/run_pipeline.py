"""
End-to-End Demonstration Pipeline Automation.
Executes dataset generation, training, threshold calibration, and zero-day inference.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_environment import check_env
from scripts.create_sample_data import main as generate_data
from scripts.train_model import train_cli
from scripts.evaluate_model import evaluate_cli

def run_pipeline():
    print("\n" + "#" * 70)
    print("   STARTING COMPLETE ZERO-DAY GRAPHIDS END-TO-END PIPELINE")
    print("#" * 70 + "\n")

    # Step 1: Environment Check
    print(">>> [STEP 1/4] Running Environment Diagnostics...")
    check_env()

    # Step 2: Sample Data Generation
    print("\n>>> [STEP 2/4] Generating Demonstration Datasets...")
    generate_data()

    # Step 3: Self-Supervised Training
    print("\n>>> [STEP 3/4] Training GraphSAGE Model on Normal Flow Baselines...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'sample_data', 'CICIDS2017_sample.csv')
    model_output = os.path.join(base_dir, 'model_artifacts', 'gnn_pipeline_demo.pt')

    train_cli(
        csv_path=dataset_path,
        output_model_path=model_output,
        epochs=10,
        lr=0.005,
        hidden_dim=64,
        embedding_dim=32
    )

    # Step 4: Zero-Day Evaluation & Baseline Comparison
    print("\n>>> [STEP 4/4] Evaluating Zero-Day Detection on Uninspected Traffic...")
    uninspected_traffic = os.path.join(base_dir, 'sample_data', 'uninspected_traffic.csv')
    evaluate_cli(model_path=model_output, test_csv_path=uninspected_traffic)

    print("\n" + "#" * 70)
    print("   END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    print("#" * 70 + "\n")

if __name__ == '__main__':
    run_pipeline()
