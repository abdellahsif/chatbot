import json
from pathlib import Path
from collections import defaultdict
import re

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove console/code artifacts
    text = re.sub(r'\r\n|\n', ' ', text)
    text = text.strip()
    return text

def is_relevant(query, response, min_similarity=0.05):
    """Check if response is relevant to query (basic heuristic)"""
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    
    # Remove common stop words
    stop_words = {'le', 'la', 'de', 'et', 'or', 'un', 'une', 'des', 'for', 'is', 'the', 'a', 'an', 'you', 'votre', 'ton', 'ma', 'mon', 'on', 'c', 'ça', 'que', 'qui', 'quoi'}
    query_words = query_words - stop_words
    response_words = response_words - stop_words
    
    if not query_words:
        return True
    
    # Calculate overlap
    overlap = len(query_words & response_words) / len(query_words) if query_words else 1
    
    # Accept if: has keyword overlap OR response is substantial (usually relevant if detailed)
    return overlap >= min_similarity or len(response) > 300

def is_low_quality(item):
    """Flag obviously low-quality items"""
    text = item.get('positive', '') or item.get('negative', '')
    
    # Check if response is too short (< 10 chars, very short)
    if len(text) < 10:
        return True
    
    # Check for extremely repetitive content (very poor quality)
    words = text.split()
    if len(words) > 50 and len(set(words)) / len(words) < 0.15:  # < 15% unique words = repetitive spam
        return True
    
    # Check quality score only if explicitly marked as very low
    quality = item.get('quality', 1.0)
    if quality < 0.5:  # Only reject really low quality
        return True
    
    return False

def merge_and_filter():
    """Merge datasets and improve quality"""
    print("📂 Loading datasets...")
    with open('reddit_finetune_pairs.json', 'r', encoding='utf-8') as f:
        reddit = json.load(f)
    
    with open('orion_finetune_pairs.json', 'r', encoding='utf-8') as f:
        orion = json.load(f)
    
    print(f"   Reddit: {len(reddit['positives'])} positives, {len(reddit['negatives'])} negatives")
    print(f"   Orion:  {len(orion['positives'])} positives, {len(orion['negatives'])} negatives")
    
    # Deduplicate and filter
    seen_pairs = set()
    combined_positives = []
    combined_negatives = []
    
    stats = {
        'total_reddit': len(reddit['positives']) + len(reddit['negatives']),
        'total_orion': len(orion['positives']) + len(orion['negatives']),
        'duplicates_removed': 0,
        'low_quality_removed': 0,
        'irrelevant_removed': 0,
    }
    
    print("\n🔄 Processing ORION dataset (higher priority)...")
    for item in orion['positives']:
        query = clean_text(item.get('query', ''))
        response = clean_text(item.get('positive', ''))
        
        if not query or not response:
            continue
        
        pair_key = (query[:80], response[:100])  # First 80/100 chars
        
        if pair_key in seen_pairs:
            stats['duplicates_removed'] += 1
            continue
        
        if is_low_quality(item):
            stats['low_quality_removed'] += 1
            continue
        
        if not is_relevant(query, response):
            stats['irrelevant_removed'] += 1
            continue
        
        seen_pairs.add(pair_key)
        item['positive'] = response
        item['query'] = query
        combined_positives.append(item)
    
    for item in orion['negatives']:
        query = clean_text(item.get('query', ''))
        response = clean_text(item.get('negative', ''))
        
        if not query or not response:
            continue
        
        pair_key = (query[:80], response[:100])
        
        if pair_key in seen_pairs:
            stats['duplicates_removed'] += 1
            continue
        
        if is_low_quality(item):
            stats['low_quality_removed'] += 1
            continue
        
        if not is_relevant(query, response):
            stats['irrelevant_removed'] += 1
            continue
        
        seen_pairs.add(pair_key)
        item['negative'] = response
        item['query'] = query
        combined_negatives.append(item)
    
    print(f"   ✅ ORION: {len([i for i in combined_positives if i.get('source_id', '').startswith('213')])} positives added")
    
    print("\n🔄 Processing Reddit dataset (fill gaps)...")
    reddit_added = 0
    for item in reddit['positives']:
        query = clean_text(item.get('query', ''))
        response = clean_text(item.get('positive', ''))
        
        if not query or not response:
            continue
        
        pair_key = (query[:80], response[:100])
        
        if pair_key in seen_pairs:
            stats['duplicates_removed'] += 1
            continue
        
        if is_low_quality(item):
            stats['low_quality_removed'] += 1
            continue
        
        if not is_relevant(query, response):
            stats['irrelevant_removed'] += 1
            continue
        
        seen_pairs.add(pair_key)
        item['positive'] = response
        item['query'] = query
        combined_positives.append(item)
        reddit_added += 1
    
    for item in reddit['negatives']:
        query = clean_text(item.get('query', ''))
        response = clean_text(item.get('negative', ''))
        
        if not query or not response:
            continue
        
        pair_key = (query[:80], response[:100])
        
        if pair_key in seen_pairs:
            stats['duplicates_removed'] += 1
            continue
        
        if is_low_quality(item):
            stats['low_quality_removed'] += 1
            continue
        
        if not is_relevant(query, response):
            stats['irrelevant_removed'] += 1
            continue
        
        seen_pairs.add(pair_key)
        item['negative'] = response
        item['query'] = query
        combined_negatives.append(item)
        reddit_added += 1
    
    print(f"   ✅ Reddit: {reddit_added} pairs added (no duplicates)")
    
    # Create final merged dataset
    merged = {
        "generated_at": "2026-04-13T combined-filtered",
        "stats": {
            "positives": len(combined_positives),
            "negatives": len(combined_negatives),
            "total": len(combined_positives) + len(combined_negatives),
            "original_total": stats['total_reddit'] + stats['total_orion'],
            "sources": ["orion_finetune_pairs.json", "reddit_finetune_pairs.json"],
            "quality_improvements": {
                "duplicates_removed": stats['duplicates_removed'],
                "low_quality_removed": stats['low_quality_removed'],
                "irrelevant_removed": stats['irrelevant_removed'],
            }
        },
        "positives": combined_positives,
        "negatives": combined_negatives
    }
    
    # Save merged dataset
    with open('combined_finetune_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ MERGE & QUALITY FILTER COMPLETE")
    print("="*60)
    print(f"\n📊 Final Statistics:")
    print(f"   Original total: {stats['total_reddit'] + stats['total_orion']}")
    print(f"   Final total:   {merged['stats']['total']}")
    print(f"   Positives:     {len(combined_positives)}")
    print(f"   Negatives:     {len(combined_negatives)}")
    print(f"\n🔧 Quality Improvements:")
    print(f"   Duplicates removed:  {stats['duplicates_removed']}")
    print(f"   Low-quality removed: {stats['low_quality_removed']}")
    print(f"   Irrelevant removed:  {stats['irrelevant_removed']}")
    print(f"   Retention rate:      {(merged['stats']['total'] / (stats['total_reddit'] + stats['total_orion']) * 100):.1f}%")
    print(f"\n💾 Saved to: combined_finetune_pairs.json")
    print("="*60)

if __name__ == "__main__":
    merge_and_filter()
