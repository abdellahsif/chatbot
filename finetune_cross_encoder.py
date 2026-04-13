"""
Cross-Encoder Fine-tuning Script
Fine-tunes cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 on Moroccan education Q&A datasets
"""

import json
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import CrossEncoder
from sentence_transformers.losses import CosineSimilarityLoss
from torch.utils.data import DataLoader
from torch import nn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingExample:
    """Training example object with texts and label attributes"""
    def __init__(self, texts, label):
        self.texts = texts
        self.label = label

def prepare_dataset(json_file, use_high_quality=True):
    """Load and prepare dataset"""
    logger.info(f"Loading dataset from {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pairs = []
    
    # Add positive pairs (label = 1)
    for item in data['positives']:
        query = item.get('query', '')
        response = item.get('positive', '')
        if query and response:
            pairs.append({
                'texts': [query, response],
                'label': 1.0,
                'score': item.get('score', 5.0)
            })
    
    # Add negative pairs (label = 0)
    for item in data['negatives']:
        query = item.get('query', '')
        response = item.get('negative', '')
        if query and response:
            pairs.append({
                'texts': [query, response],
                'label': 0.0,
                'score': item.get('score', 5.0)
            })
    
    logger.info(f"Loaded {len(pairs)} pairs")
    logger.info(f"  Positives: {len(data['positives'])}")
    logger.info(f"  Negatives: {len(data['negatives'])}")
    
    return pairs

def train_cross_encoder(
    high_quality_file='combined_finetune_pairs_highquality.json',
    full_file='combined_finetune_pairs.json',
    output_dir='./models/cross-encoder-finetuned',
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    warmup_steps=100,
    use_high_quality_first=True
):
    """
    Fine-tune cross-encoder model
    
    Args:
        high_quality_file: Path to high-quality dataset
        full_file: Path to full dataset
        output_dir: Where to save the model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        use_high_quality_first: Train on high-quality dataset first
    """
    
    print("\n" + "="*70)
    print("[*] CROSS-ENCODER FINE-TUNING")
    print("="*70)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load base model
    logger.info("[*] Loading base model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device=device)
    
    # Prepare high-quality dataset
    if use_high_quality_first and Path(high_quality_file).exists():
        logger.info("[*] Phase 1: Training on high-quality dataset...")
        high_quality_pairs = prepare_dataset(high_quality_file)
        
        # Convert to training format for sentence-transformers
        train_data = [
            TrainingExample(pair['texts'], pair['label']) 
            for pair in high_quality_pairs
        ]
        
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Batch size: {batch_size}, Epochs: 2 (high-quality)")
        
        # Train on high-quality data (fewer epochs)
        model.fit(
            train_dataloader=DataLoader(train_data, shuffle=True, batch_size=batch_size),
            epochs=min(2, epochs),
            warmup_steps=warmup_steps,
            output_path=None,  # Don't save intermediates
            show_progress_bar=True,
            optimizer_params={'lr': learning_rate}
        )
        logger.info("[OK] Phase 1 complete")
    
    # Prepare and train on full dataset
    if Path(full_file).exists():
        logger.info(f"[*] Phase 2: Training on full dataset...")
        full_pairs = prepare_dataset(full_file)
        
        # Convert to training format for sentence-transformers
        train_data = [
            TrainingExample(pair['texts'], pair['label']) 
            for pair in full_pairs
        ]
        
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")
        
        # Train on full data
        model.fit(
            train_dataloader=DataLoader(train_data, shuffle=True, batch_size=batch_size),
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_dir,
            show_progress_bar=True,
            optimizer_params={'lr': learning_rate}
        )
        logger.info(f"[OK] Model saved to {output_dir}")
    
    print("\n" + "="*70)
    print("[OK] FINE-TUNING COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Base model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    print(f"  Fine-tuned on: {len(full_pairs) if 'full_pairs' in locals() else len(high_quality_pairs)} pairs")
    print(f"  Training epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Output model: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Update .env: CROSS_ENCODER_MODEL={output_dir}")
    print(f"  2. Restart the app to use the fine-tuned model")
    print(f"  3. Test with: curl http://localhost:3001/health")
    print("="*70)
    
    return model

def test_model(model_path):
    """Simple test of the fine-tuned model"""
    logger.info(f"Testing model: {model_path}")
    model = CrossEncoder(model_path)
    
    # Test queries
    test_cases = [
        ("Quels sont les débouchés après une école d'ingénieur ?", 
         "Les ingénieurs trouvent du travail dans l'industrie, l'informatique, le secteur public, etc."),
        ("Quel est le meilleur école au Maroc ?",
         "C'est un spam qui n'a rien à voir avec la question"),
    ]
    
    print("\n[*] Test Results:")
    for query, response in test_cases:
        score = model.predict([[query, response]])[0]
        print(f"  Query: {query[:50]}...")
        print(f"  Response: {response[:60]}...")
        print(f"  Relevance Score: {score:.3f}")
        print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune Cross-Encoder model')
    parser.add_argument('--high-quality-only', action='store_true', help='Train on high-quality dataset only')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--test', action='store_true', help='Test the fine-tuned model after training')
    
    args = parser.parse_args()
    
    try:
        # Train
        model = train_cross_encoder(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_high_quality_first=not args.high_quality_only
        )
        
        # Optional: test
        if args.test:
            test_model('./models/cross-encoder-finetuned')
            
    except Exception as e:
        logger.error(f"❌ Error during fine-tuning: {e}", exc_info=True)
        raise
