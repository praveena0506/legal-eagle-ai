import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE_DIR / "raw_data" / "unlabeled_daily_dump"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def get_latest_cases():
    # 👇 CHANGED URL: Now we look at the "Most Recent Supreme Court" search results
    url = "https://indiankanoon.org/search/?formInput=suit%20for%20specific%20performance%20sortby:%20mostrecent+doctypes:supremecourt"
    print(f"🌍 Connecting to {url}...")

    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')

        links = []
        # Indian Kanoon search results usually put the title link inside a div with class 'result_title'
        # But looking for any raw /doc/ link is safer and easier.
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # We want /doc/ links, but NOT the ones that are just fragments (contain #)
            if href.startswith("/doc/") and "#" not in href:
                full_link = "https://indiankanoon.org" + href
                if full_link not in links:  # Avoid duplicates
                    links.append(full_link)

        print(f"   found {len(links)} raw links")
        return links[:5]  # Return top 5
    except Exception as e:
        print(f"❌ Error fetching links: {e}")
        return []


def download_case(link):
    print(f"   ⬇️ Downloading: {link}")
    try:
        response = requests.get(link, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to get cleaner text (usually in 'judgments' div), else fallback to all text
        text_div = soup.find('div', {'class': 'judgments'})
        if text_div:
            text = text_div.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        return text
    except Exception as e:
        print(f"   ⚠️ Failed to download: {e}")
        return None


def run_scraper():
    print(f"🚀 Starting Daily Scrape: {datetime.now()}")
    links = get_latest_cases()
    print(f"🔎 Found {len(links)} cases to process.")

    count = 0
    for link in links:
        text = download_case(link)
        if text:
            # Create a safe filename (replace / with _ to avoid path errors)
            case_id = link.split("/")[-2]
            filename = f"{datetime.now().strftime('%Y-%m-%d')}_{case_id}.txt"

            save_path = SAVE_DIR / filename
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"   💾 Saved: {filename}")
            count += 1

            # Be polite
            time.sleep(3)

    print(f"🎉 Job Done. Downloaded {count} cases.")


if __name__ == "__main__":
    run_scraper()