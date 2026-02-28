import requests
import uuid

BASE_URL = "http://localhost:8000"

def test_isolation():
    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())
    
    print(f"User A: {user_a}")
    print(f"User B: {user_b}")
    
    # 1. Submit score for User A
    print("\nSubmitting score for User A...")
    res = requests.post(f"{BASE_URL}/submit-score", 
                        headers={"X-User-ID": user_a},
                        json={"filename": "test.pdf", "score": 8, "total_questions": 10})
    if res.status_code != 200:
        print(f"Error {res.status_code}: {res.text}")
        return
    print(res.json())
    
    # 2. Check stats for User A
    print("\nChecking stats for User A...")
    res = requests.get(f"{BASE_URL}/stats", headers={"X-User-ID": user_a})
    stats_a = res.json()
    print(f"User A total flashcards (should be 0 since no upload): {stats_a['total_flashcards']}")
    print(f"User A recent quizzes count: {len(stats_a['recent_quizzes'])}")
    
    # 3. Check stats for User B (should be empty)
    print("\nChecking stats for User B...")
    res = requests.get(f"{BASE_URL}/stats", headers={"X-User-ID": user_b})
    stats_b = res.json()
    print(f"User B total flashcards: {stats_b['total_flashcards']}")
    print(f"User B recent quizzes count: {len(stats_b['recent_quizzes'])}")
    
    if len(stats_b['recent_quizzes']) > 0:
        print("\n[FAILURE] User B can see User A's quizzes!")
    else:
        print("\n[SUCCESS] User B stats are empty.")

if __name__ == "__main__":
    test_isolation()
