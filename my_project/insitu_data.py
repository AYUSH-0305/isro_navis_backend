import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Output directory
output_dir = "./mosdac_data/processed"
os.makedirs(output_dir, exist_ok=True)

# Setup Selenium
options = Options()
options.add_argument("--headless")  # Comment this out if you want to see the browser
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

from selenium.webdriver.chrome.service import Service

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://www.mosdac.gov.in/catalog/insitu.php")

time.sleep(2)  # Let page load

# Scroll simulation (for JS-rendered rows if needed)
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.5)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Extract table data
table = driver.find_element(By.TAG_NAME, "table")
headers = [th.text.strip() for th in table.find_elements(By.TAG_NAME, "th")]
rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header

extracted = []

for row in rows:
    cols = row.find_elements(By.TAG_NAME, "td")
    if cols:
        values = [col.text.strip() for col in cols]
        block = "\n".join(f"{headers[i]}: {values[i]}" for i in range(len(values)))
        extracted.append(block)

# Save to txt
with open(os.path.join(output_dir, "insitu_aws_selenium_data.txt"), "w", encoding="utf-8") as f:
    f.write("\n\n---\n\n".join(extracted))

driver.quit()
print("✅ AWS data fetched using Selenium and saved.")
