#!/usr/bin/env python3
"""Detailed test to verify evidence is used in LLM generation."""

import json
import requests
from time import sleep

API_URL = "http://127.0.0.1:3001"
QUERY_ENDPOINT = f"{API_URL}/chat/query"

def detailed_test():
    """Run detailed test to show evidence usage."""
    
    print("Waiting for server to be ready...")
    sleep(2)
    
    query = {
        "question": "I want an IT school in Fes with affordable fees",
        "profile": {
            "bac_stream": "sm",
            "expected_grade_band": "bien",
            "motivation": "employability",
            "budget_band": "tight_25k",
            "city": "fes",
            "country": "MA"
        },
        "top_k": 5,
        "chat_history": []
    }
    
    print("\nSending query to /chat/query endpoint...")
    print(f"Question: {query['question']}")
    
    resp = requests.post(QUERY_ENDPOINT, json=query, timeout=120)
    result = resp.json()
    
    print("\n" + "="*70)
    print("FULL RESPONSE")
    print("="*70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "="*70)
    print("EVIDENCE ANALYSIS")
    print("="*70)
    
    evidence = result.get('evidence', [])
    print(f"\nRetrieved {len(evidence)} evidence items:")
    for i, ev in enumerate(evidence[:3], 1):
        print(f"\n  {i}. {ev.get('school_name')} - {ev.get('program')}")
        print(f"     Score: {ev.get('score')}")
        print(f"     Text: {ev.get('text')[:100]}...")
    
    print("\n" + "="*70)
    print("GENERATION ANALYSIS")
    print("="*70)
    
    print(f"\nShort Answer (should mention retrieved schools):")
    print(f"  {result.get('short_answer')}")
    
    print(f"\nWhy It Fits (should reference evidence):")
    print(f"  {result.get('why_it_fits')}")
    
    print(f"\nAlternative Options:")
    print(f"  {result.get('alternative')}")
    
    print(f"\nMessage Paragraph (comprehensive summary):")
    print(f"  {result.get('message_paragraph')}")

if __name__ == "__main__":
    detailed_test()
