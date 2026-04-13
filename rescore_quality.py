import json
import numpy as np

def calculate_pair_score(item, is_positive=True):
    """Calculate a quality score for each pair (0-10)"""
    score = 0.0
    
    query = item.get('query', '')
    response = item.get('positive' if is_positive else 'negative', '')
    
    # Length score (longer = generally better, but not too long)
    response_len = len(response)
    if 50 <= response_len <= 2000:
        score += 2
    elif 20 <= response_len < 50:
        score += 1
    elif response_len > 2000:
        score += 1.5  # Long but maybe verbose
    
    # Explicit quality score
    quality = item.get('quality', 0.8)
    score += quality * 3
    
    # Diversity in words (not too repetitive)
    words = response.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        score += unique_ratio * 2
    
    # Source priority (orion > reddit, but reddit has value)
    source_id = item.get('source_id', '')
    if source_id.startswith('213'):  # ORION pattern
        score += 1
    
    # Theme/category diversity bonus
    if item.get('theme') in ['débouchés', 'formation', 'admission']:
        score += 0.5
    
    return min(score, 10.0)

def rank_and_rescore():
    """Add quality scores and create ranked subsets"""
    print("📊 Loading combined dataset...")
    with open('combined_finetune_pairs.json', 'r', encoding='utf-8') as f:
        combined = json.load(f)
    
    print(f"   Loaded: {len(combined['positives'])} positives, {len(combined['negatives'])} negatives")
    
    # Calculate scores for all pairs
    print("\n🎯 Calculating quality scores...")
    for i, item in enumerate(combined['positives']):
        item['score'] = calculate_pair_score(item, is_positive=True)
    
    for i, item in enumerate(combined['negatives']):
        item['score'] = calculate_pair_score(item, is_positive=False)
    
    # Get statistics
    pos_scores = [item['score'] for item in combined['positives']]
    neg_scores = [item['score'] for item in combined['negatives']]
    
    print(f"   Positives - Mean: {np.mean(pos_scores):.2f}, Min: {np.min(pos_scores):.2f}, Max: {np.max(pos_scores):.2f}")
    print(f"   Negatives - Mean: {np.mean(neg_scores):.2f}, Min: {np.min(neg_scores):.2f}, Max: {np.max(neg_scores):.2f}")
    
    # Sort by score
    combined['positives'] = sorted(combined['positives'], key=lambda x: x['score'], reverse=True)
    combined['negatives'] = sorted(combined['negatives'], key=lambda x: x['score'], reverse=True)
    
    # Update metadata
    combined['stats']['quality_metrics'] = {
        'positives_mean_score': float(np.mean(pos_scores)),
        'positives_median_score': float(np.median(pos_scores)),
        'negatives_mean_score': float(np.mean(neg_scores)),
        'negatives_median_score': float(np.median(neg_scores)),
    }
    
    # Save scored dataset
    with open('combined_finetune_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    
    # Create high-quality subset
    high_quality_threshold = 7.0
    high_quality_positives = [item for item in combined['positives'] if item['score'] >= high_quality_threshold]
    high_quality_negatives = [item for item in combined['negatives'] if item['score'] >= high_quality_threshold]
    
    high_quality = {
        "generated_at": "2026-04-13T high-quality-filtered",
        "stats": {
            "positives": len(high_quality_positives),
            "negatives": len(high_quality_negatives),
            "total": len(high_quality_positives) + len(high_quality_negatives),
            "description": "Top-quality pairs (score >= 7.0) for premium fine-tuning",
            "quality_threshold": high_quality_threshold
        },
        "positives": high_quality_positives,
        "negatives": high_quality_negatives
    }
    
    with open('combined_finetune_pairs_highquality.json', 'w', encoding='utf-8') as f:
        json.dump(high_quality, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("✅ QUALITY SCORING COMPLETE")
    print("="*60)
    print(f"\n📈 Quality Tiers:")
    print(f"   All pairs:        {combined['stats']['total']:,} pairs")
    print(f"   High-quality:     {len(high_quality_positives) + len(high_quality_negatives):,} pairs (score ≥ 7.0)")
    print(f"   High-quality %:   {((len(high_quality_positives) + len(high_quality_negatives)) / combined['stats']['total'] * 100):.1f}%")
    print(f"\n💾 Files Created:")
    print(f"   combined_finetune_pairs.json              - All {combined['stats']['total']:,} pairs (ranked)")
    print(f"   combined_finetune_pairs_highquality.json - {len(high_quality_positives) + len(high_quality_negatives):,} premium pairs")
    print("="*60)

if __name__ == "__main__":
    rank_and_rescore()
