import sys
import os
import requests
from bs4 import BeautifulSoup
import time

# Fix path
sys.path.append(os.getcwd())
from data_pipeline.scripts.db_utils import save_case_to_db


class LegalNewsScraper:
    def __init__(self):
        # 📡 Google News RSS Feed (Targeting "Supreme Court Verdict")
        # This is XML data, designed for robots! 🤖
        self.rss_url = "https://news.google.com/rss/search?q=Supreme+Court+of+India+verdict+judgment&hl=en-IN&gl=IN&ceid=IN:en"

    def fetch_latest_cases(self):
        print(f"📡 Connecting to Google Legal News Feed...")
        try:
            response = requests.get(self.rss_url)
            # Use 'xml' parser because RSS is XML
            soup = BeautifulSoup(response.content, features="xml")

            items = soup.find_all("item")
            print(f"📄 Found {len(items)} recent legal updates.")

            cases_data = []
            for item in items[:10]:  # Get top 10
                title = item.title.text
                description = item.description.text
                pub_date = item.pubDate.text

                # Combine title and description for our AI to read
                full_text = f"{title}. {description}"

                print(f"   🗞️ Found: {title[:50]}...")
                cases_data.append(full_text)

            return cases_data

        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            return []


def auto_label_verdict(text):
    text = text.lower()

    # 🟢 WIN Keywords (The "Vacuum Cleaner" list for Wins)
    # If ANY of these words appear, we assume it's a WIN.
    win_triggers = [
        "allowed", "allow", "granted", "grant", "accept", "accepted",
        "relief", "acquitted", "acquit", "set aside", "quashed", "quash",
        "stayed", "stay", "bail", "release", "released", "approved", "approve",
        "favour", "favor", "wins", "won", "clear", "cleared", "suspended",
        "suspend", "not guilty", "free", "freed", "closure"
    ]

    # 🔴 LOSS Keywords (The "Vacuum Cleaner" list for Losses)
    # If ANY of these words appear, we assume it's a LOSS.
    loss_triggers = [
        "dismissed", "dismiss", "rejected", "reject", "denied", "deny",
        "refused", "refuse", "declined", "decline", "upheld", "uphold",
        "guilty", "convicted", "conviction", "fine", "fined", "sentence",
        "sentenced", "loses", "lost", "rejects", "dismisses", "jail",
        "prison", "custody", "surrender"
    ]

    # Priority Check: Look for "Dismissed" first (it's usually clearer)
    if any(word in text for word in loss_triggers):
        return "Dismissed"

    # Then look for "Allowed"
    if any(word in text for word in win_triggers):
        return "Allowed"

    # ⚠️ FINAL RESORT: If it mentions "Court" but we still don't know,
    # let's just GUESS "Dismissed" (Status Quo) to save the data.
    # This prevents "Data Starvation".
    if "court" in text or "sc" in text or "bench" in text:
        return "Dismissed"  # Defaulting to Loss if unclear (safe guess)

    return "Unknown"

def run_daily_scraping():
    print("🚀 Starting News Scraper...")
    scraper = LegalNewsScraper()

    raw_texts = scraper.fetch_latest_cases()

    if not raw_texts:
        print("⚠️ No news found.")
        return

    saved_count = 0
    for text in raw_texts:
        verdict = auto_label_verdict(text)

        # We only save if we can figure out the verdict (Labeled Data)
        if verdict != "Unknown":
            case_record = {
                "text": text,
                "verdict": verdict,
                "source": "GoogleNews_RSS",
                "timestamp": time.time()
            }

            save_case_to_db(case_record)
            saved_count += 1
        else:
            print(f"   (Skipping unclear verdict: {text[:30]}...)")

    print(f"✅ Successfully saved {saved_count} labeled cases to MongoDB Cloud.")


if __name__ == "__main__":
    run_daily_scraping()