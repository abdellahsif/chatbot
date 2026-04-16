"""
Chatbot Metrics Integration Module
Seamlessly track and serve evaluation metrics from the retrieval pipeline
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)


class ChatbotMetricsTracker:
    """Tracks retrieval metrics during chatbot operation"""
    
    def __init__(self, metrics_file: str = "./chatbot_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.session_metrics = defaultdict(list)
        self.load_metrics()
    
    def load_metrics(self):
        """Load existing metrics from disk"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    # Load into session
                    for model, metrics in data.items():
                        self.session_metrics[model] = metrics
                logger.info(f"Loaded {len(self.session_metrics)} models metrics")
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
    
    def track_retrieval(self, query: str, results: List[Dict], model_name: str = "default"):
        """
        Track a retrieval operation
        
        Args:
            query: The query string
            results: List of ranked results
            model_name: Name of the model used
        """
        
        if model_name not in self.session_metrics:
            self.session_metrics[model_name] = {
                'total_queries': 0,
                'avg_results': 0,
                'score_distribution': [],
                'timestamps': [],
                'sample_queries': []
            }
        
        metrics = self.session_metrics[model_name]
        metrics['total_queries'] += 1
        metrics['timestamps'].append(datetime.now().isoformat())
        
        # Track result count and scores
        if results:
            metrics['avg_results'] = (
                (metrics['avg_results'] * (metrics['total_queries'] - 1) + len(results)) 
                / metrics['total_queries']
            )
            
            for result in results[:5]:  # Track top-5 scores
                score = result.get('score', 0.0)
                metrics['score_distribution'].append(float(score))
            
            # Keep sample queries
            if len(metrics['sample_queries']) < 10:
                metrics['sample_queries'].append({
                    'query': query[:100],
                    'top_result': results[0].get('school', {}).get('name', 'N/A'),
                    'top_score': float(results[0].get('score', 0.0))
                })
    
    def track_ranking_comparison(self, query: str, gold_school: str, 
                                base_rank: int | None, ft_rank: int | None):
        """
        Track ranking comparison between models
        
        Args:
            query: The query string
            gold_school: Expected/correct school
            base_rank: Rank from base model (None if not found)
            ft_rank: Rank from fine-tuned model (None if not found)
        """
        
        if 'ranking_stats' not in self.session_metrics:
            self.session_metrics['ranking_stats'] = {
                'total_queries': 0,
                'base_hits': 0,
                'ft_hits': 0,
                'ft_improvements': 0,
                'ft_regressions': 0,
                'rank_deltas': []
            }
        
        stats = self.session_metrics['ranking_stats']
        stats['total_queries'] += 1
        
        if base_rank is not None:
            stats['base_hits'] += 1
        if ft_rank is not None:
            stats['ft_hits'] += 1
        
        if base_rank is not None and ft_rank is not None:
            delta = base_rank - ft_rank
            stats['rank_deltas'].append(delta)
            
            if ft_rank < base_rank:
                stats['ft_improvements'] += 1
            elif ft_rank > base_rank:
                stats['ft_regressions'] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics across all models"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, metrics in self.session_metrics.items():
            if model_name == 'ranking_stats':
                summary['ranking_comparison'] = metrics
            else:
                summary['models'][model_name] = {
                    'total_queries': metrics.get('total_queries', 0),
                    'avg_results': round(metrics.get('avg_results', 0), 2),
                    'score_stats': self._compute_score_stats(metrics.get('score_distribution', [])),
                    'samples': metrics.get('sample_queries', [])[:3]
                }
        
        return summary
    
    @staticmethod
    def _compute_score_stats(scores: List[float]) -> Dict:
        """Compute statistics on score distribution"""
        
        if not scores:
            return {}
        
        return {
            'mean': round(statistics.mean(scores), 4),
            'median': round(statistics.median(scores), 4),
            'stdev': round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
            'min': round(min(scores), 4),
            'max': round(max(scores), 4),
            'count': len(scores)
        }
    
    def save_metrics(self):
        """Save metrics to disk"""
        
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(dict(self.session_metrics), f, indent=2, default=str)
            logger.info(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
    
    def reset_session(self):
        """Reset session metrics"""
        self.session_metrics.clear()
        logger.info("Session metrics reset")


# Global metrics tracker instance
_metrics_tracker = None


def get_metrics_tracker() -> ChatbotMetricsTracker:
    """Get or initialize the global metrics tracker"""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = ChatbotMetricsTracker()
    return _metrics_tracker


def track_retrieval_operation(query: str, results: List[Dict], model_name: str = "default"):
    """Convenient wrapper to track retrieval operations"""
    tracker = get_metrics_tracker()
    tracker.track_retrieval(query, results, model_name)


def track_ranking_comparison(query: str, gold_school: str, 
                            base_rank: int | None, ft_rank: int | None):
    """Convenient wrapper to track ranking comparisons"""
    tracker = get_metrics_tracker()
    tracker.track_ranking_comparison(query, gold_school, base_rank, ft_rank)


def get_metrics_summary() -> Dict[str, Any]:
    """Get current metrics summary"""
    tracker = get_metrics_tracker()
    return tracker.get_summary()


def save_metrics_state():
    """Persist current metrics to disk"""
    tracker = get_metrics_tracker()
    tracker.save_metrics()


# Metrics endpoints for fastapi integration
def create_metrics_endpoints(app):
    """
    Create FastAPI endpoints for metrics
    
    Usage in main.py:
        from app.metrics_integration import create_metrics_endpoints
        create_metrics_endpoints(app)
    """
    
    @app.get("/metrics/summary")
    async def get_summary():
        """Get current metrics summary"""
        return get_metrics_summary()
    
    @app.get("/metrics/models")
    async def get_models_metrics():
        """Get metrics for each model"""
        tracker = get_metrics_tracker()
        return {
            'models': {k: v for k, v in tracker.session_metrics.items() 
                      if k != 'ranking_stats'}
        }
    
    @app.get("/metrics/ranking")
    async def get_ranking_metrics():
        """Get ranking comparison metrics"""
        tracker = get_metrics_tracker()
        return tracker.session_metrics.get('ranking_stats', {})
    
    @app.post("/metrics/reset")
    async def reset_metrics():
        """Reset all metrics (admin only)"""
        tracker = get_metrics_tracker()
        tracker.reset_session()
        return {"status": "metrics reset"}
    
    return app


if __name__ == "__main__":
    # Test the metrics tracker
    tracker = ChatbotMetricsTracker()
    
    # Simulate some operations
    tracker.track_retrieval(
        query="Best engineering schools in Maroc?",
        results=[
            {'school': {'name': 'ENSEM'}, 'score': 0.85},
            {'school': {'name': 'EMI'}, 'score': 0.82},
            {'school': {'name': 'ENSA'}, 'score': 0.79}
        ],
        model_name="base_model"
    )
    
    tracker.track_ranking_comparison(
        query="Best engineering schools?",
        gold_school="EMI",
        base_rank=None,
        ft_rank=1
    )
    
    # Print summary
    summary = tracker.get_summary()
    print(json.dumps(summary, indent=2, default=str))
    
    # Save
    tracker.save_metrics()
