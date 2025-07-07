import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.mosdac.gov.in/"
OUTPUT_DIR = "./mosdac_data/processed"

def save_text(content, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Saved: {out_path}")

def extract_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "header", "footer", "nav", "form"]):
        tag.decompose()

    # Get visible text
    text = soup.get_text(separator="\n", strip=True)
    # Remove empty lines
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def scrape_satellites():
    print(f"[i] Fetching homepage: {BASE_URL}")
    resp = requests.get(BASE_URL, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    missions = {}

    # Find the Missions dropdown
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if text in [
            "INSAT-3DR", "INSAT-3D", "KALPANA-1", "INSAT-3A",
            "MeghaTropiques", "SARAL-Altika", "OCEANSAT-2",
            "OCEANSAT-3", "INSAT-3DS", "SCATSAT-1"
        ]:
            missions[text] = urljoin(BASE_URL, href)

    print(f"[i] Found {len(missions)} satellites")

    for name, link in missions.items():
        print(f"[i] Fetching {name} -> {link}")
        sat_resp = requests.get(link, timeout=10)
        sat_resp.raise_for_status()

        # Extract visible text
        text_content = extract_visible_text(sat_resp.text)

        # Save text
        filename = f"{name.replace('/', '_')}.txt"
        save_text(text_content, filename)

if __name__ == "__main__":
    scrape_satellites()
