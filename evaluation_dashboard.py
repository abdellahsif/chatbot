"""
Comprehensive Evaluation Dashboard
Metrics: Recall@k, MRR, NDCG, MAP, Precision@k, Hit Rates
"""

import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sentence_transformers import CrossEncoder
import difflib
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
import os
os.environ.setdefault("SUPABASE_STRICT_MODE", "0")

from app.retriever import retrieve
from app.data_loader import load_bundle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RankingMetrics:
    """Compute ranking evaluation metrics"""
    
    @staticmethod
    def recall_at_k(gold_rank: int | None, k: int) -> float:
        """Recall@k: did we find the gold item within top-k?"""
        if gold_rank is None:
            return 0.0
        return 1.0 if gold_rank <= k else 0.0
    
    @staticmethod
    def precision_at_k(gold_rank: int | None, k: int) -> float:
        """Precision@k: what fraction of top-k contains gold?"""
        if gold_rank is None:
            return 0.0
        return 1.0 / k if gold_rank <= k else 0.0
    
    @staticmethod
    def mrr(gold_rank: int | None) -> float:
        """Mean Reciprocal Rank: 1 / rank of first relevant item"""
        if gold_rank is None:
            return 0.0
        return 1.0 / gold_rank
    
    @staticmethod
    def ndcg_at_k(gold_rank: int | None, k: int) -> float:
        """Normalized Discounted Cumulative Gain@k"""
        if gold_rank is None:
            return 0.0
        
        # DCG: relevance / log2(rank + 1)
        # Ideal DCG (iDCG) for single relevant item: 1 / log2(2) = 1
        if gold_rank <= k:
            dcg = 1.0 / np.log2(gold_rank + 1)
            idcg = 1.0  # Perfect ranking has gold at rank 1
            return dcg / idcg
        return 0.0
    
    @staticmethod
    def average_precision(gold_rank: int | None, k: int) -> float:
        """Average Precision@k"""
        if gold_rank is None:
            return 0.0
        if gold_rank <= k:
            # For single relevant item: AP = P@rank / num_relevant
            return (1.0 / gold_rank) / 1.0
        return 0.0


def normalize_school_name(name: str) -> str:
    """Normalize school name for matching"""
    if not name:
        return ""
    
    name = name.strip().lower()
    # Remove common prefixes/suffixes
    name = name.replace("école ", "").replace("école d'", "")
    name = name.replace("- maroc", "").replace("(maroc)", "")
    name = difflib.SequenceMatcher(None, name, name).ratio()
    
    return name


def find_gold_rank(gold_school: str, ranked_results: List[Dict]) -> int | None:
    """Find rank of gold school in results (1-indexed, None if not found)"""
    gold_norm = normalize_school_name(gold_school)
    
    for rank, item in enumerate(ranked_results, start=1):
        school = item.get('school', {})
        school_name = school.get('name', '')
        
        # Try exact match first
        if gold_norm.lower() == normalize_school_name(school_name).lower():
            return rank
        
        # Try acronym/code match
        school_code = school.get('code', '').upper()
        if gold_school.upper() == school_code:
            return rank
        
        # Try fuzzy match
        ratio = difflib.SequenceMatcher(None, gold_norm.lower(), 
                                       normalize_school_name(school_name).lower()).ratio()
        if ratio > 0.85:
            return rank
    
    return None


def evaluate_ranking_quality(
    dataset_file: str,
    base_model: str,
    ft_model: str | None = None,
    max_samples: int = 100,
    top_k: int = 10,
    per_school: int = 2
) -> Dict:
    """
    Comprehensive ranking evaluation
    
    Returns: Dictionary with all metrics
    """
    
    logger.info(f"Loading dataset: {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare test queries
    test_pairs = data.get('positives', [])[:max_samples]
    
    # Load bundle for school data
    try:
        bundle = load_bundle()
        schools = bundle.schools if bundle else []
    except:
        logger.warning("Could not load Supabase data, using fallback")
        schools = []
    
    results = {
        'base_model': {
            'model_name': base_model,
            'metrics_by_k': defaultdict(list),
            'all_metrics': []
        }
    }
    
    if ft_model:
        results['ft_model'] = {
            'model_name': ft_model,
            'metrics_by_k': defaultdict(list),
            'all_metrics': []
        }
    
    # Evaluate each query
    query_count = 0
    for pair in test_pairs:
        if query_count >= max_samples:
            break
        
        query = pair.get('query', '')
        gold_school = pair.get('school', '')  # Expected/correct school
        
        if not query or not gold_school:
            continue
        
        logger.info(f"\n[{query_count + 1}] Query: {query[:60]}...")
        logger.info(f"    Gold school: {gold_school}")
        
        # Retrieve with base model
        logger.info(f"    Retrieving with base model...")
        try:
            base_results = retrieve(query, top_k=top_k, use_cross_encoder=False)
            base_rank = find_gold_rank(gold_school, base_results[:top_k])
        except Exception as e:
            logger.warning(f"    Base model retrieval failed: {e}")
            base_rank = None
            base_results = []
        
        # Retrieve with fine-tuned model
        ft_rank = None
        if ft_model:
            logger.info(f"    Retrieving with fine-tuned model...")
            try:
                # Temporarily switch model
                os.environ['CROSS_ENCODER_MODEL'] = ft_model
                from app import retriever as retriever_module
                retriever_module.CROSS_ENCODER_RERANKER._model = None  # Reset cache
                
                ft_results = retrieve(query, top_k=top_k, use_cross_encoder=True)
                ft_rank = find_gold_rank(gold_school, ft_results[:top_k])
                
                # Restore base model
                os.environ['CROSS_ENCODER_MODEL'] = base_model
                retriever_module.CROSS_ENCODER_RERANKER._model = None
            except Exception as e:
                logger.warning(f"    Fine-tuned model retrieval failed: {e}")
                ft_rank = None
        
        # Compute metrics
        ks = [1, 2, 5, 10]
        
        base_metrics = {
            'query': query[:50],
            'gold_school': gold_school,
            'rank': base_rank,
        }
        
        for k in ks:
            base_metrics[f'recall@{k}'] = RankingMetrics.recall_at_k(base_rank, k)
            base_metrics[f'precision@{k}'] = RankingMetrics.precision_at_k(base_rank, k)
            results['base_model']['metrics_by_k'][f'recall@{k}'].append(base_metrics[f'recall@{k}'])
            results['base_model']['metrics_by_k'][f'precision@{k}'].append(base_metrics[f'precision@{k}'])
        
        base_metrics['mrr'] = RankingMetrics.mrr(base_rank)
        base_metrics['ndcg@10'] = RankingMetrics.ndcg_at_k(base_rank, 10)
        base_metrics['map@10'] = RankingMetrics.average_precision(base_rank, 10)
        
        results['base_model']['all_metrics'].append(base_metrics)
        results['base_model']['metrics_by_k']['mrr'].append(base_metrics['mrr'])
        results['base_model']['metrics_by_k']['ndcg@10'].append(base_metrics['ndcg@10'])
        results['base_model']['metrics_by_k']['map@10'].append(base_metrics['map@10'])
        
        if ft_model:
            ft_metrics = {
                'query': query[:50],
                'gold_school': gold_school,
                'rank': ft_rank,
            }
            
            for k in ks:
                ft_metrics[f'recall@{k}'] = RankingMetrics.recall_at_k(ft_rank, k)
                ft_metrics[f'precision@{k}'] = RankingMetrics.precision_at_k(ft_rank, k)
                results['ft_model']['metrics_by_k'][f'recall@{k}'].append(ft_metrics[f'recall@{k}'])
                results['ft_model']['metrics_by_k'][f'precision@{k}'].append(ft_metrics[f'precision@{k}'])
            
            ft_metrics['mrr'] = RankingMetrics.mrr(ft_rank)
            ft_metrics['ndcg@10'] = RankingMetrics.ndcg_at_k(ft_rank, 10)
            ft_metrics['map@10'] = RankingMetrics.average_precision(ft_rank, 10)
            
            results['ft_model']['all_metrics'].append(ft_metrics)
            results['ft_model']['metrics_by_k']['mrr'].append(ft_metrics['mrr'])
            results['ft_model']['metrics_by_k']['ndcg@10'].append(ft_metrics['ndcg@10'])
            results['ft_model']['metrics_by_k']['map@10'].append(ft_metrics['map@10'])
        
        query_count += 1
    
    return results


def print_dashboard(results: Dict):
    """Print formatted evaluation dashboard"""
    
    print("\n" + "="*80)
    print("EVALUATION DASHBOARD - RANKING METRICS")
    print("="*80)
    
    for model_key in ['base_model', 'ft_model']:
        if model_key not in results:
            continue
        
        model_data = results[model_key]
        print(f"\n{'─'*80}")
        print(f"MODEL: {model_data['model_name']}")
        print(f"{'─'*80}")
        
        metrics_by_k = model_data['metrics_by_k']
        
        if not metrics_by_k:
            print("No metrics available")
            continue
        
        # Compute averages
        print("\nPERFORMANCE SUMMARY:")
        print(f"{'Metric':<20} {'Value':>12} {'Samples':>10}")
        print(f"{'-'*42}")
        
        for metric in ['recall@1', 'recall@2', 'recall@5', 'recall@10', 
                       'precision@1', 'precision@2', 'precision@5', 'precision@10',
                       'mrr', 'ndcg@10', 'map@10']:
            if metric in metrics_by_k:
                values = metrics_by_k[metric]
                avg = np.mean(values)
                print(f"{metric:<20} {avg:>11.4f} {len(values):>10}")
        
        # Detailed results
        print(f"\nDETAILED RESULTS (showing first 20):")
        print(f"{'Query':<40} {'Gold':<15} {'Rank':<6} {'R@10':<6} {'MRR':<6} {'NDCG':<6}")
        print(f"{'-'*80}")
        
        for i, metrics in enumerate(model_data['all_metrics'][:20]):
            query = metrics['query'][:35]
            gold = metrics['gold_school'][:12]
            rank = str(metrics['rank']) if metrics['rank'] else 'X'
            r10 = '✓' if metrics.get('recall@10') else '✗'
            mrr = f"{metrics.get('mrr', 0):.2f}"
            ndcg = f"{metrics.get('ndcg@10', 0):.2f}"
            print(f"{query:<40} {gold:<15} {rank:<6} {r10:<6} {mrr:<6} {ndcg:<6}")
    
    # Comparison
    if 'base_model' in results and 'ft_model' in results:
        print(f"\n{'='*80}")
        print("COMPARISON: Base vs Fine-Tuned")
        print(f"{'='*80}")
        
        base_metrics = results['base_model']['metrics_by_k']
        ft_metrics = results['ft_model']['metrics_by_k']
        
        print(f"\n{'Metric':<20} {'Base':>12} {'Fine-Tuned':>12} {'Delta':>12}")
        print(f"{'-'*56}")
        
        for metric in ['recall@1', 'recall@10', 'mrr', 'ndcg@10']:
            if metric in base_metrics and metric in ft_metrics:
                base_val = np.mean(base_metrics[metric])
                ft_val = np.mean(ft_metrics[metric])
                delta = ft_val - base_val
                delta_str = f"{delta:+.4f}"
                print(f"{metric:<20} {base_val:>11.4f} {ft_val:>12.4f} {delta_str:>12}")
    
    print("\n" + "="*80)


def save_results(results: Dict, output_file: str = "evaluation_results.json"):
    """Save detailed results to file"""
    
    # Convert metrics to serializable format
    serializable_results = {}
    
    for model_key, model_data in results.items():
        serializable_results[model_key] = {
            'model_name': model_data['model_name'],
            'metrics_summary': {},
            'detailed_results': model_data['all_metrics']
        }
        
        # Compute summary metrics
        for metric, values in model_data['metrics_by_k'].items():
            if values:
                serializable_results[model_key]['metrics_summary'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive evaluation dashboard")
    parser.add_argument("--dataset", default="combined_finetune_pairs.json")
    parser.add_argument("--base-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--ft-model", default=None, help="Fine-tuned model path (optional)")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--per-school", type=int, default=2)
    parser.add_argument("--output", default="evaluation_results.json")
    
    args = parser.parse_args()
    
    try:
        results = evaluate_ranking_quality(
            dataset_file=args.dataset,
            base_model=args.base_model,
            ft_model=args.ft_model,
            max_samples=args.max_samples,
            top_k=args.top_k,
            per_school=args.per_school
        )
        
        print_dashboard(results)
        save_results(results, args.output)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise
