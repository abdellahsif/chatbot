import requests
import json
import sys

BASE_URL = "http://localhost:8000"

test_cases = [
    {
        "name": "✅ Test 1: Engineering + Budget 50k MAD",
        "profile": {
            "name": "Ahmed",
            "bac_stream": "sm",
            "budget_mad": 50000,
            "preferred_cities": ["Rabat", "Casablanca"],
            "preferred_type": "any"
        },
        "query": "génie informatique cloud computing"
    },
    {
        "name": "✅ Test 2: Free Public School",
        "profile": {
            "name": "Fatima",
            "bac_stream": "sm",
            "budget_mad": 1000,
            "preferred_cities": ["Fès", "Meknes"],
            "preferred_type": "public"
        },
        "query": "écoles publiques gratuites"
    },
    {
        "name": "✅ Test 3: Commerce Programs",
        "profile": {
            "name": "Hassan",
            "bac_stream": "eco",
            "budget_mad": 80000,
            "preferred_cities": ["Casablanca", "Rabat", "Agadir"],
            "preferred_type": "any"
        },
        "query": "commerce gestion business"
    },
    {
        "name": "✅ Test 4: Environment/Forestry",
        "profile": {
            "name": "Layla",
            "bac_stream": "sm_a",
            "budget_mad": 5000,
            "preferred_cities": ["Salé", "Rabat"],
            "preferred_type": "public"
        },
        "query": "environnement eaux forêts"
    },
    {
        "name": "✅ Test 5: Premium Private Engineering",
        "profile": {
            "name": "Mohammed",
            "bac_stream": "sm",
            "budget_mad": 150000,
            "preferred_cities": ["Casablanca", "Marrakech"],
            "preferred_type": "private"
        },
        "query": "école ingénieur excellence"
    }
]

print("\n" + "="*80)
print("RECOMMENDATION SYSTEM TEST SUITE")
print("="*80 + "\n")

passed = 0
failed = 0

for test in test_cases:
    print(f"\n{test['name']}")
    print("-" * 80)
    print(f"Profile: {test['profile']['name']} | BAC: {test['profile']['bac_stream']} | Budget: {test['profile']['budget_mad']} MAD")
    print(f"Cities: {', '.join(test['profile']['preferred_cities'])}")
    print(f"Query: {test['query']}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommendations/query",
            json={
                "user_profile": test["profile"],
                "query": test["query"],
                "top_k": 5
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            schools = data.get('schools', [])
            count = len(schools)
            
            print(f"✅ Response received: {count} schools found\n")
            
            for idx, school in enumerate(schools, 1):
                print(f"  {idx}. {school.get('name', 'N/A')}")
                print(f"     City: {school.get('city', 'N/A')}")
                print(f"     Type: {school.get('type', 'N/A')}")
                print(f"     Tuition: {school.get('tuition_min_mad', 0)} - {school.get('tuition_max_mad', 0)} MAD")
                print(f"     Score: {school.get('semantic_score', 0):.4f}")
                if school.get('programs'):
                    print(f"     Programs: {', '.join(school.get('programs', [])[:3])}")
                print()
            
            passed += 1
        else:
            print(f"❌ Error (Status {response.status_code}): {response.text}\n")
            failed += 1
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error - API not running at {BASE_URL}\n")
        failed += 1
    except Exception as e:
        print(f"❌ Exception: {str(e)}\n")
        failed += 1

print("\n" + "="*80)
print(f"TEST RESULTS: {passed} passed, {failed} failed")
print("="*80 + "\n")
