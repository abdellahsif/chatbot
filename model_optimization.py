"""
Model Optimization Utilities
Blend tuning, hyperparameter optimization, and metrics tracking
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from dotenv import load_dotenv
import argparse

load_dotenv()
os.environ.setdefault("SUPABASE_STRICT_MODE", "0")

from app.retriever import retrieve
from app.data_loader import load_bundle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BlendTuner:
    """Optimize cross-encoder blend weight for better ranking"""
    
    @staticmethod
    def tune_blend_weights(
        dataset_file: str,
        model_name: str,
        blend_values: List[float] = None,
        max_samples: int = 50
    ) -> Dict:
        """
        Test different blend weights and find optimal configuration
        
        Args:
            dataset_file: Path to evaluation dataset
            model_name: Cross-encoder model name
            blend_values: List of blend values to test (default: [0.2, 0.4, 0.6, 0.8, 1.0])
            max_samples: Number of queries to test
        
        Returns:
            Dictionary with results for each blend value
        """
        
        if blend_values is None:
            blend_values = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        logger.info(f"Loading test dataset: {dataset_file}")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_pairs = data.get('positives', [])[:max_samples]
        
        results = {
            'model': model_name,
            'test_size': len(test_pairs),
            'blend_results': {}
        }
        
        # Test each blend value
        for blend in blend_values:
            logger.info(f"\n>>> Testing blend weight: {blend}")
            
            os.environ['CROSS_ENCODER_BLEND'] = str(blend)
            
            # Reset model cache to pick up new blend
            from app import retriever as retriever_module
            retriever_module.CROSS_ENCODER_RERANKER._blend = blend
            
            hits_at_1 = 0
            hits_at_5 = 0
            hits_at_10 = 0
            
            for i, pair in enumerate(test_pairs[:max_samples]):
                query = pair.get('query', '')
                gold_school = pair.get('school', '')
                
                if not query or not gold_school:
                    continue
                
                try:
                    results_top10 = retrieve(query, top_k=10, use_cross_encoder=True)
                    
                    # Check if gold school in results
                    for rank, result in enumerate(results_top10, start=1):
                        school_name = result.get('school', {}).get('name', '')
                        
                        if gold_school.lower() in school_name.lower() or \
                           school_name.lower() in gold_school.lower():
                            if rank == 1:
                                hits_at_1 += 1
                            if rank <= 5:
                                hits_at_5 += 1
                            if rank <= 10:
                                hits_at_10 += 1
                            break
                
                except Exception as e:
                    logger.debug(f"Query {i} failed: {e}")
                    continue
            
            num_evaluated = min(max_samples, len(test_pairs))
            
            results['blend_results'][blend] = {
                'hit@1': hits_at_1 / num_evaluated if num_evaluated > 0 else 0,
                'hit@5': hits_at_5 / num_evaluated if num_evaluated > 0 else 0,
                'hit@10': hits_at_10 / num_evaluated if num_evaluated > 0 else 0,
            }
            
            logger.info(f"  Hit@1: {results['blend_results'][blend]['hit@1']:.3f}")
            logger.info(f"  Hit@5: {results['blend_results'][blend]['hit@5']:.3f}")
            logger.info(f"  Hit@10: {results['blend_results'][blend]['hit@10']:.3f}")
        
        # Find optimal blend for different metrics
        best_at_1 = max(results['blend_results'].items(), 
                       key=lambda x: x[1]['hit@1'])
        best_at_10 = max(results['blend_results'].items(),
                        key=lambda x: x[1]['hit@10'])
        
        results['recommendations'] = {
            'best_for_top1': {
                'blend': best_at_1[0],
                'hit@1': best_at_1[1]['hit@1']
            },
            'best_for_top10': {
                'blend': best_at_10[0],
                'hit@10': best_at_10[1]['hit@10']
            }
        }
        
        return results


class ModelEvaluation:
    """Track and store model evaluation metrics"""
    
    def __init__(self, metrics_dir: str = "./model_metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
    
    def save_eval_metrics(self, model_name: str, metrics: Dict):
        """Save evaluation metrics for a model"""
        
        metric_file = self.metrics_dir / f"{model_name.replace('/', '_')}_metrics.json"
        
        with open(metric_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to: {metric_file}")
    
    def load_eval_metrics(self, model_name: str) -> Dict:
        """Load evaluation metrics for a model"""
        
        metric_file = self.metrics_dir / f"{model_name.replace('/', '_')}_metrics.json"
        
        if not metric_file.exists():
            return {}
        
        with open(metric_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compare_models(self, model1: str, model2: str) -> Dict:
        """Compare metrics between two models"""
        
        metrics1 = self.load_eval_metrics(model1)
        metrics2 = self.load_eval_metrics(model2)
        
        if not metrics1 or not metrics2:
            logger.warning("Could not load metrics for comparison")
            return {}
        
        comparison = {
            'model1': model1,
            'model2': model2,
            'improvements': {}
        }
        
        # Compare key metrics
        for metric in ['recall@1', 'recall@10', 'mrr', 'ndcg@10']:
            m1_val = metrics1.get(metric, 0)
            m2_val = metrics2.get(metric, 0)
            
            if m1_val > 0:
                improvement = ((m2_val - m1_val) / m1_val) * 100
                comparison['improvements'][metric] = {
                    'base': m1_val,
                    'finetuned': m2_val,
                    'improvement_pct': improvement
                }
        
        return comparison


class MetricsServer:
    """Metrics storage and retrieval for chatbot integration"""
    
    def __init__(self, db_path: str = "./metrics.db"):
        self.db_path = db_path
        self._ensure_db()
    
    def _ensure_db(self):
        """Initialize metrics database"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                sample_size INTEGER,
                query_sample TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ranking_queries (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                gold_school TEXT,
                base_rank INTEGER,
                ft_rank INTEGER,
                recall_improvement REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_metric(self, model_name: str, metric_name: str, value: float, 
                     sample_size: int = None, query_sample: str = None):
        """Record a model metric"""
        
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics 
            (model_name, metric_name, metric_value, sample_size, query_sample)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_name, metric_name, value, sample_size, query_sample))
        
        conn.commit()
        conn.close()
    
    def get_model_history(self, model_name: str, metric_name: str, limit: int = 10):
        """Get historical metrics for a model"""
        
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, metric_value, sample_size
            FROM model_metrics
            WHERE model_name = ? AND metric_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (model_name, metric_name, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {'timestamp': r[0], 'value': r[1], 'sample_size': r[2]}
            for r in results
        ]
    
    def get_ranking_stats(self, limit: int = 100):
        """Get ranking statistics"""
        
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                AVG(CASE WHEN base_rank IS NOT NULL THEN 1 ELSE 0 END) as base_hit_rate,
                AVG(CASE WHEN ft_rank IS NOT NULL THEN 1 ELSE 0 END) as ft_hit_rate,
                AVG(recall_improvement) as avg_improvement
            FROM ranking_queries
            LIMIT ?
        ''', (limit,))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'base_hit_rate': result[0] if result[0] else 0,
            'ft_hit_rate': result[1] if result[1] else 0,
            'avg_improvement': result[2] if result[2] else 0
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model optimization utilities")
    parser.add_argument("--tune-blend", action="store_true", help="Tune blend weights")
    parser.add_argument("--dataset", default="combined_finetune_pairs.json")
    parser.add_argument("--model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--max-samples", type=int, default=50)
    
    args = parser.parse_args()
    
    if args.tune_blend:
        logger.info("Starting blend weight tuning...")
        results = BlendTuner.tune_blend_weights(
            dataset_file=args.dataset,
            model_name=args.model,
            max_samples=args.max_samples
        )
        
        print("\n" + "="*60)
        print("BLEND TUNING RESULTS")
        print("="*60)
        
        for blend, metrics in results['blend_results'].items():
            print(f"\nBlend={blend}:")
            print(f"  Hit@1:  {metrics['hit@1']:.3f}")
            print(f"  Hit@5:  {metrics['hit@5']:.3f}")
            print(f"  Hit@10: {metrics['hit@10']:.3f}")
        
        print(f"\nRECOMMENDATIONS:")
        print(f"  Best for Top-1: blend={results['recommendations']['best_for_top1']['blend']}")
        print(f"                  hit@1={results['recommendations']['best_for_top1']['hit@1']:.3f}")
        print(f"  Best for Top-10: blend={results['recommendations']['best_for_top10']['blend']}")
        print(f"                   hit@10={results['recommendations']['best_for_top10']['hit@10']:.3f}")
