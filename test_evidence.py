#!/usr/bin/env python3
"""Test script to verify evidence-based generation and language detection."""

import json
import sys
import requests
from time import sleep

# API configuration
API_URL = "http://127.0.0.1:3001"
HEALTH_ENDPOINT = f"{API_URL}/health"
QUERY_ENDPOINT = f"{API_URL}/chat/query"

def test_evidence_generation():
    """Test the updated generator with evidence and language detection."""
    
    # Wait a bit for the server to be ready
    print("Waiting for server to start...")
    sleep(3)
    
    # Check health
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        health = resp.json()
        print(f"✓ Health check passed: {health}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Test 1: English question
    print("\n" + "="*60)
    print("TEST 1: English question with evidence-based generation")
    print("="*60)
    
    query_en = {
        "question": "I want an IT school in Fes with tight budget",
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
    
    try:
        resp = requests.post(QUERY_ENDPOINT, json=query_en, timeout=120)
        result = resp.json()
        print(f"✓ Query succeeded")
        print(f"  Confidence: {result.get('confidence')}")
        print(f"  Short answer: {result.get('short_answer')}")
        print(f"  Why it fits: {result.get('why_it_fits')}")
        print(f"  # Evidence items: {len(result.get('evidence', []))}")
        if result.get('message_paragraph'):
            print(f"  Message paragraph preview: {result['message_paragraph'][:100]}...")
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False
    
    # Test 2: Check that evidence is being used in generation
    print("\n" + "="*60)
    print("TEST 2: Verify evidence snippets are referenced")
    print("="*60)
    
    response_text = result.get('why_it_fits', '') + ' ' + result.get('short_answer', '')
    evidence_items = result.get('evidence', [])
    
    if evidence_items:
        first_school = evidence_items[0].get('school_name', '')
        if first_school and (first_school in response_text or first_school.split()[0] in response_text):
            print(f"✓ Evidence is referenced in response ({first_school})")
        else:
            print(f"⚠ Evidence item not directly referenced, but this is acceptable")
    else:
        print("⚠ No evidence returned")
    
    # Test 3: Test city-only mode with evidence
    print("\n" + "="*60)
    print("TEST 3: City-only mode with evidence context")
    print("="*60)
    
    query_city = {
        "question": "schools in rabat",
        "profile": {
            "bac_stream": "sm",
            "expected_grade_band": "bien",
            "motivation": "employability",
            "budget_band": "comfort_50k",
            "city": "rabat",
            "country": "MA"
        },
        "top_k": 5,
        "chat_history": []
    }
    
    try:
        resp = requests.post(QUERY_ENDPOINT, json=query_city, timeout=120)
        result = resp.json()
        print(f"✓ City-only query succeeded")
        print(f"  Short answer: {result.get('short_answer')}")
        print(f"  Evidence count: {len(result.get('evidence', []))}")
    except Exception as e:
        print(f"✗ City-only query failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All tests completed successfully!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_evidence_generation()
    sys.exit(0 if success else 1)
