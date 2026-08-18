import requests
import json
import os

def fetch_issues(repo="huggingface/transformers", n_pages=5):
    issues = []
    headers = {}
    
    # Optional token from environment if user has it configured to avoid rate-limiting
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
        
    for page in range(1, n_pages + 1):
        r = requests.get(
            f"https://api.github.com/repos/{repo}/issues",
            params={"state": "all", "per_page": 50, "page": page},
            headers=headers
        )
        if r.status_code != 200:
            raise Exception(f"Failed to fetch page {page}: {r.status_code} - {r.text}")
        issues.extend(r.json())
    
    os.makedirs("data/tickets", exist_ok=True)
    with open("data/tickets/github_issues.json", "w") as f:
        json.dump(issues, f, indent=2)
    return len(issues)

if __name__ == "__main__":
    print(f"Fetched {fetch_issues()} issues")
