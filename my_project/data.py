import os
import glob
from pathlib import Path
from bs4 import BeautifulSoup
import requests
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader



def read_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_excel(file_path):
    dfs = pd.read_excel(file_path, sheet_name=None)
    return "\n".join(df.to_csv(index=False) for df in dfs.values())

def read_html(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup.get_text()



def save_as_txt(content, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)



def process_files(input_dir="./mosdac_data/raw", output_dir="./mosdac_data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    supported = [".pdf", ".docx", ".xls", ".xlsx"]

    for file in glob.glob(f"{input_dir}/**/*.*", recursive=True):
        ext = Path(file).suffix.lower()
        name = Path(file).stem

        try:
            if ext == ".pdf":
                text = read_pdf(file)
            elif ext == ".docx":
                text = read_docx(file)
            elif ext in [".xls", ".xlsx"]:
                text = read_excel(file)
            else:
                continue

            out_path = os.path.join(output_dir, f"{name}.txt")
            save_as_txt(text, out_path)
            print(f"[✓] Saved: {out_path}")

        except Exception as e:
            print(f"[!] Error processing {file}: {e}")



def process_web_pages(url_list, output_dir="./mosdac_data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    for idx, url in enumerate(url_list):
        try:
            content = read_html(url)
            out_path = os.path.join(output_dir, f"web_{idx}.txt")
            save_as_txt(content, out_path)
            print(f"[✓] Scraped and saved: {url} → {out_path}")
        except Exception as e:
            print(f"[!] Error scraping {url}: {e}")


if __name__ == "__main__":
    
    process_files("./mosdac_data/raw", "./mosdac_data/processed")


    url_list = [
    
    "https://www.mosdac.gov.in/insat-3d",                     
    "https://www.mosdac.gov.in/insat-3dr",                    
    "https://www.mosdac.gov.in/kalpana-1",                    
    "https://www.mosdac.gov.in/oceansat-2",                  
    "https://www.mosdac.gov.in/oceansat-2-introduction",     
    "https://www.mosdac.gov.in/oceansat-2-objectives",      
    "https://www.mosdac.gov.in/oceansat-2-payloads",         
    "https://www.mosdac.gov.in/meghatropiques",              
    "https://www.mosdac.gov.in/faq-page",                   
    "https://www.mosdac.gov.in/what-mosdac",                
    "https://www.mosdac.gov.in/tools",                       
    "https://www.mosdac.gov.in/announcements",              
]


 
    
    process_web_pages(url_list, "./mosdac_data/processed")
