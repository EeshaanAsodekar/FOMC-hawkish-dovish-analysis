import json
import requests
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime

def download_speeches(json_file_path, start_year=2012, end_year=2024):
    base_url = 'https://www.federalreserve.gov'
    speeches_dir = 'data/raw/fed_speeches'
    if not os.path.exists(speeches_dir):
        os.makedirs(speeches_dir)
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8-sig') as f:
        speeches_data = json.load(f)
    
    for speech in speeches_data:
        # Extract the date and parse the year
        speech_date_str = speech.get('d', '')
        try:
            speech_date = datetime.strptime(speech_date_str, '%m/%d/%Y %I:%M:%S %p')
        except ValueError:
            print(f"Invalid date format for speech: {speech.get('t', 'Unknown Title')}")
            continue
        speech_year = speech_date.year
        if speech_year < start_year or speech_year > end_year:
            continue
        
        # Construct the full URL
        speech_url_path = speech.get('l', '')
        if not speech_url_path:
            print(f"No URL path for speech: {speech.get('t', 'Unknown Title')}")
            continue
        speech_url = base_url + speech_url_path
        
        title = speech.get('t', 'No Title')
        print(f"Processing speech: {title} ({speech_date_str})")
        
        # Fetch the speech page
        response = requests.get(speech_url)
        if response.status_code != 200:
            print(f"Failed to retrieve speech: {title}")
            continue
        
        # Parse the speech content
        soup = BeautifulSoup(response.content, 'html.parser')
        content_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8')
        speech_text = ''
        if content_div:
            # Remove scripts, styles, and footnotes
            for unwanted in content_div(['script', 'style', 'sup', 'img']):
                unwanted.decompose()
            speech_text = content_div.get_text(separator='\n', strip=True)
        else:
            # Check for PDF link
            pdf_link = None
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href.endswith('.pdf'):
                    pdf_link = href if href.startswith('http') else base_url + href
                    break
            if pdf_link:
                print(f"Downloading PDF for speech: {title}")
                # Download the PDF and extract text
                pdf_response = requests.get(pdf_link)
                if pdf_response.status_code != 200:
                    print(f"Failed to download PDF for speech: {title}")
                    continue
                pdf_filename = os.path.join(speeches_dir, f"{speech_date.strftime('%Y%m%d')}_{title}.pdf")
                with open(pdf_filename, 'wb') as f:
                    f.write(pdf_response.content)
                # Extract text from PDF
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(pdf_filename)
                    speech_text = ''
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            speech_text += text
                except ImportError:
                    print("PyPDF2 is not installed. Cannot extract text from PDF.")
                    print("Please install PyPDF2 using 'pip install PyPDF2'")
                    continue
                except Exception as e:
                    print(f"Error extracting text from PDF for speech: {title}")
                    print(e)
                    continue
            else:
                print(f"No content found for speech: {title}")
                continue
        
        if speech_text:
            # Clean up the filename
            safe_title = ''.join(c if c.isalnum() or c in ' _-.' else '_' for c in title)
            filename = f"{speech_date.strftime('%Y%m%d')}_{safe_title}.txt"
            file_path = os.path.join(speeches_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(speech_text)
        else:
            print(f"Could not extract speech content for {title}")
    
    print("Done.")

if __name__ == '__main__':
    # Update the path to your JSON file
    json_file_path = 'data/raw/fed_speeches.json'
    download_speeches(json_file_path)
