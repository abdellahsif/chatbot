#!/usr/bin/env python
"""
Quick-Start Workflow: Top-1 Accuracy Improvement
Demonstrates the complete pipeline end-to-end
"""

import subprocess
import sys
import json
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*80}")
    logger.info(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed: {e}")
        return False


def verify_data_exists() -> bool:
    """Verify required data files exist"""
    required_files = [
        'combined_finetune_pairs.json',
        'combined_finetune_pairs_highquality.json'
    ]
    
    for file in required_files:
        if not Path(file).exists():
            logger.error(f"✗ Missing required file: {file}")
            logger.error(f"  Run: python merge_and_quality_filter.py && python rescore_quality.py")
            return False
    
    logger.info("✓ All required data files present")
    return True


def run_workflow(args=None):
    """Run complete workflow"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Quick-start workflow")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 queries, 1 epoch)")
    parser.add_argument("--full", action="store_true", help="Full evaluation (100 queries, 5 epochs)")
    parser.add_argument("--blend-only", action="store_true", help="Only tune blend weight")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only evaluate")
    
    args = parser.parse_args(args or [])
    
    # Configuration
    config = {
        'quick': {
            'epochs': 1,
            'batch_size': 8,
            'eval_samples': 5,
            'output': 'checkpoints/model_test'
        },
        'standard': {
            'epochs': 3,
            'batch_size': 16,
            'eval_samples': 50,
            'output': 'checkpoints/model_v2'
        },
        'full': {
            'epochs': 5,
            'batch_size': 16,
            'eval_samples': 100,
            'output': 'checkpoints/model_v3'
        }
    }
    
    mode = 'quick' if args.quick else 'full' if args.full else 'standard'
    cfg = config[mode]
    
    logger.info(f"\n{'*'*80}")
    logger.info(f"WORKFLOW: Top-1 Accuracy Improvement ({mode.upper()})")
    logger.info(f"{'*'*80}\n")
    
    # Step 1: Verify data
    logger.info("PHASE 0: Verification")
    if not verify_data_exists():
        return False
    
    # Step 2: Train improved model
    if not args.blend_only and not args.skip_training:
        logger.info("\nPHASE 1: Advanced Fine-tuning")
        
        train_cmd = [
            'python', 'finetune_advanced.py',
            '--high-quality', 'combined_finetune_pairs_highquality.json',
            '--full', 'combined_finetune_pairs.json',
            '--output', cfg['output'],
            '--epochs', str(cfg['epochs']),
            '--batch-size', str(cfg['batch_size']),
            '--loss-type', 'cosine'
        ]
        
        if not run_command(train_cmd, f"Train improved model (mode={mode})"):
            logger.error("Training failed. Aborting workflow.")
            return False
    else:
        logger.info("\n[*] Skipping training (--skip-training or --blend-only)")
        cfg['output'] = 'checkpoints/model'  # Use existing
    
    # Step 3: Evaluate
    if not args.blend_only:
        logger.info("\nPHASE 2: Comprehensive Evaluation")
        
        eval_cmd = [
            'python', 'evaluation_dashboard.py',
            '--dataset', 'combined_finetune_pairs.json',
            '--base-model', 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
            '--ft-model', cfg['output'],
            '--max-samples', str(cfg['eval_samples']),
            '--top-k', '10',
            '--output', 'evaluation_results.json'
        ]
        
        if not run_command(eval_cmd, f"Evaluate models (samples={cfg['eval_samples']})"):
            logger.warning("Evaluation had issues, but continuing...")
    
    # Step 4: Blend tuning
    logger.info("\nPHASE 3: Blend Weight Optimization")
    
    blend_cmd = [
        'python', 'model_optimization.py',
        '--tune-blend',
        '--dataset', 'combined_finetune_pairs.json',
        '--model', cfg['output'],
        '--max-samples', str(min(50, cfg['eval_samples']))
    ]
    
    if not run_command(blend_cmd, "Optimize blend weight"):
        logger.warning("Blend tuning failed, skipping...")
    
    # Step 5: View results
    logger.info("\nPHASE 4: Results Visualization")
    
    dashboard_cmd = ['python', 'metrics_dashboard.py']
    
    if not run_command(dashboard_cmd, "Display metrics dashboard"):
        logger.warning("Could not display dashboard")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("WORKFLOW COMPLETE")
    logger.info(f"{'='*80}\n")
    
    logger.info("Next steps:")
    logger.info(f"  1. Review metrics: python metrics_dashboard.py")
    logger.info(f"  2. Deploy model: cp {cfg['output']}/* checkpoints/model/")
    logger.info(f"  3. Update .env: CROSS_ENCODER_MODEL=checkpoints/model")
    logger.info(f"  4. Restart app: python -m app.main")
    
    return True


if __name__ == "__main__":
    success = run_workflow()
    sys.exit(0 if success else 1)
