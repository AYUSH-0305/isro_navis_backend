import requests
from bs4 import BeautifulSoup
import os

# Directory setup
output_dir = "./mosdac_data/processed"
os.makedirs(output_dir, exist_ok=True)

# --- Fetch Tool Descriptions + Download URLs ---
def fetch_tools_data():
    url = "https://www.mosdac.gov.in/tools"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    tools_data = []

    table_rows = soup.find_all("tr")[1:]  # Skip header row
    for row in table_rows:
        cols = row.find_all("td")
        if len(cols) >= 4:
            description = cols[1].get_text(strip=True)
            a_tag = cols[2].find("a")
            download_url = a_tag['href'] if a_tag and a_tag.has_attr('href') else "No download URL available"

            tools_data.append(f"Description: {description}\nDownload URL: {download_url}\n")

    with open(f"{output_dir}/tools_data.txt", "w", encoding="utf-8") as f:
        f.write("\n---\n".join(tools_data))

# --- Fetch FAQ Questions ---
def fetch_faqs():
    url = "https://www.mosdac.gov.in/faq-page"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    faqs = []

    links = soup.select("div.region.region-content li a, .content a")
    for link in links:
        question = link.get_text(strip=True)
        faqs.append(f"FAQ Question: {question}")

    with open(f"{output_dir}/faqs.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(faqs))

# --- Run All ---
fetch_tools_data()
fetch_faqs()
print("Data fetched and saved in ./mosdac_data/processed/")