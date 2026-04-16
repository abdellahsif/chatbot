#!/usr/bin/env python
"""
Metrics Dashboard - CLI utility for viewing model performance
Live dashboard showing Top-1 accuracy, Recall@k, MRR, NDCG
"""

import json
import time
import os
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsDashboard:
    """CLI dashboard for model metrics"""
    
    def __init__(self, metrics_file: str = "./evaluation_results.json"):
        self.metrics_file = Path(metrics_file)
        self.baseline_metrics = None
        self.latest_metrics = None
    
    def load_metrics(self) -> bool:
        """Load metrics from file"""
        if not self.metrics_file.exists():
            print(f"[!] Metrics file not found: {self.metrics_file}")
            return False
        
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            self.latest_metrics = data
            return True
        except Exception as e:
            print(f"[!] Failed to load metrics: {e}")
            return False
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print dashboard header"""
        print("\n" + "="*90)
        print("  MODEL PERFORMANCE DASHBOARD - MOROCCAN EDUCATIONAL INSTITUTIONS".center(90))
        print("="*90)
        print(f"  Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(90))
        print("="*90 + "\n")
    
    def print_model_metrics(self, model_name: str, metrics_summary: Dict):
        """Print metrics for a single model"""
        
        print(f"\n{model_name}:")
        print(f"{'-'*90}")
        print(f"{'Metric':<25} {'Mean':>15} {'Std':>15} {'Min':>15} {'Max':>15}")
        print(f"{'-'*90}")
        
        # Key metrics to display
        key_metrics = [
            'recall@1', 'recall@2', 'recall@5', 'recall@10',
            'precision@1', 'precision@2', 'precision@5', 'precision@10',
            'mrr', 'ndcg@10', 'map@10'
        ]
        
        for metric in key_metrics:
            if metric in metrics_summary:
                m = metrics_summary[metric]
                print(f"{metric:<25} {m['mean']:>14.4f} {m['std']:>15.4f} {m['min']:>15.4f} {m['max']:>15.4f}")
    
    def print_comparison(self):
        """Print comparison between base and fine-tuned models"""
        
        if 'base_model' not in self.latest_metrics or 'ft_model' not in self.latest_metrics:
            return
        
        base = self.latest_metrics['base_model']['metrics_summary']
        ft = self.latest_metrics['ft_model']['metrics_summary']
        
        print(f"\n\nCOMPARISON: Base vs Fine-Tuned Model")
        print(f"{'-'*90}")
        print(f"{'Metric':<20} {'Base Model':>20} {'Fine-Tuned':>20} {'Improvement':>20}")
        print(f"{'-'*90}")
        
        comparison_metrics = ['recall@1', 'recall@10', 'mrr', 'ndcg@10', 'map@10']
        
        for metric in comparison_metrics:
            if metric in base and metric in ft:
                base_val = base[metric]['mean']
                ft_val = ft[metric]['mean']
                improvement = ((ft_val - base_val) / base_val * 100) if base_val != 0 else 0
                
                # Color coding
                indicator = "▲" if improvement > 0 else "▼" if improvement < 0 else "→"
                
                print(f"{metric:<20} {base_val:>19.4f} {ft_val:>20.4f} {indicator} {improvement:>18.2f}%")
    
    def print_recommendations(self):
        """Print recommendations based on metrics"""
        
        if 'base_model' not in self.latest_metrics or 'ft_model' not in self.latest_metrics:
            return
        
        base = self.latest_metrics['base_model']['metrics_summary']
        ft = self.latest_metrics['ft_model']['metrics_summary']
        
        print(f"\n\nRECOMMENDATIONS")
        print(f"{'-'*90}")
        
        # Analyze key metrics
        if 'recall@1' in ft and 'recall@1' in base:
            ft_r1 = ft['recall@1']['mean']
            base_r1 = base['recall@1']['mean']
            
            if ft_r1 < base_r1 - 0.1:
                print(f"⚠  Top-1 accuracy degraded: {base_r1:.2%} → {ft_r1:.2%}")
                print(f"   → Consider:")
                print(f"     • Reducing blend weight (currently 0.6)")
                print(f"     • Using hard negative mining in training")
                print(f"     • Adjusting learning rate")
            elif ft_r1 > base_r1:
                print(f"✓  Top-1 accuracy improved: {base_r1:.2%} → {ft_r1:.2%}")
        
        if 'recall@10' in ft and 'recall@10' in base:
            ft_r10 = ft['recall@10']['mean']
            base_r10 = base['recall@10']['mean']
            
            if ft_r10 > base_r10:
                print(f"✓  Top-10 recall improved: {base_r10:.2%} → {ft_r10:.2%}")
        
        if 'ndcg@10' in ft and 'ndcg@10' in base:
            ft_ndcg = ft['ndcg@10']['mean']
            base_ndcg = base['ndcg@10']['mean']
            
            if ft_ndcg > base_ndcg:
                delta = ((ft_ndcg - base_ndcg) / base_ndcg * 100)
                print(f"✓  NDCG@10 improved by {delta:.1f}%: {base_ndcg:.4f} → {ft_ndcg:.4f}")
    
    def print_top_queries(self, limit: int = 5):
        """Print top performing queries"""
        
        if 'base_model' not in self.latest_metrics:
            return
        
        detailed = self.latest_metrics['base_model'].get('detailed_results', [])
        
        if not detailed:
            return
        
        print(f"\n\nTOP PERFORMING QUERIES (Base Model)")
        print(f"{'-'*90}")
        print(f"{'Query':<50} {'School':<20} {'Rank':<10} {'MRR':<10}")
        print(f"{'-'*90}")
        
        # Sort by MRR descending
        sorted_queries = sorted(detailed, key=lambda x: x.get('mrr', 0), reverse=True)
        
        for query_data in sorted_queries[:limit]:
            query = query_data['query'][:45]
            school = query_data['gold_school'][:15]
            rank = str(query_data.get('rank', 'N/A'))
            mrr = f"{query_data.get('mrr', 0):.3f}"
            print(f"{query:<50} {school:<20} {rank:<10} {mrr:<10}")
    
    def print_failed_queries(self, limit: int = 5):
        """Print queries where gold school wasn't found"""
        
        if 'base_model' not in self.latest_metrics:
            return
        
        detailed = self.latest_metrics['base_model'].get('detailed_results', [])
        
        failed = [q for q in detailed if q.get('rank') is None]
        
        if not failed:
            print(f"\n\nNO FAILED QUERIES - All gold schools found in top-10!")
            return
        
        print(f"\n\nFAILED QUERIES (Gold school not in top-10)")
        print(f"{'-'*90}")
        print(f"{'Query':<50} {'School':<20} {'Count':<10}")
        print(f"{'-'*90}")
        
        for query_data in failed[:limit]:
            query = query_data['query'][:45]
            school = query_data['gold_school'][:15]
            print(f"{query:<50} {school:<20}")
        
        print(f"\nTotal failed: {len(failed)}/{len(detailed)} ({len(failed)/len(detailed)*100:.1f}%)")
    
    def display(self, refresh_interval: int = None):
        """
        Display the dashboard
        
        Args:
            refresh_interval: Seconds between refreshes (None = single display)
        """
        
        try:
            while True:
                self.clear_screen()
                
                if not self.load_metrics():
                    print("Cannot load metrics. Run evaluation first:")
                    print("  python evaluation_dashboard.py --max-samples 100 --ft-model checkpoints/model")
                    break
                
                self.print_header()
                
                if self.latest_metrics:
                    # Print metrics for each model
                    for model_key in ['base_model', 'ft_model']:
                        if model_key in self.latest_metrics:
                            model_data = self.latest_metrics[model_key]
                            self.print_model_metrics(
                                model_key.replace('_', ' ').title(),
                                model_data.get('metrics_summary', {})
                            )
                    
                    # Print comparison and recommendations
                    self.print_comparison()
                    self.print_recommendations()
                    self.print_top_queries()
                    self.print_failed_queries()
                
                if refresh_interval:
                    print(f"\n[!] Refreshing in {refresh_interval}s... (Ctrl+C to exit)")
                    time.sleep(refresh_interval)
                else:
                    break
        
        except KeyboardInterrupt:
            print("\n\n[*] Dashboard closed")
            sys.exit(0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model metrics dashboard")
    parser.add_argument("--metrics-file", default="evaluation_results.json")
    parser.add_argument("--refresh", type=int, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    dashboard = MetricsDashboard(args.metrics_file)
    dashboard.display(refresh_interval=args.refresh)
