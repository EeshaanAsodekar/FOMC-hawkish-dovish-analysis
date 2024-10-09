import requests
import os
from PyPDF2 import PdfReader

def download_and_extract_fomc_press_conferences(dates, output_dir='data/raw/fomc_press_conf'):
    base_url = 'https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{}.pdf'
    pdf_dir = os.path.join(output_dir, 'pdfs')
    text_dir = os.path.join(output_dir, 'texts')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    
    for date in dates:
        url = base_url.format(date)
        print(f"Attempting to download: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            pdf_filename = f'FOMCpresconf{date}.pdf'
            pdf_path = os.path.join(pdf_dir, pdf_filename)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {pdf_filename}")
            
            # Extract text from PDF
            try:
                reader = PdfReader(pdf_path)
                text = ''
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
                # Save text to file
                text_filename = f'FOMCpresconf{date}.txt'
                text_path = os.path.join(text_dir, text_filename)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Extracted text to: {text_filename}")
            except Exception as e:
                print(f"Failed to extract text from {pdf_filename}: {e}")
        else:
            print(f"Failed to download: {url} (Status code: {response.status_code})")
    print("Done.")

if __name__ == '__main__':
    dates = [
        # 2012
        '20120125', '20120425', '20120620', '20120913', '20121212',
        # 2013
        '20130320', '20130619', '20130918', '20131218', # incorrec!!! more meetings happned
        # 2014
        '20140319', '20140618', '20140917', '20141217',# incorrec!!! more meetings happned
        # 2015
        '20150318', '20150617', '20150917', '20151216',# incorrec!!! more meetings happned
        # 2016
        '20160316', '20160615', '20160921', '20161214',# incorrec!!! more meetings happned
        # 2017# incorrec!!! more meetings happned
        '20170315', '20170614', '20170920', '20171213',# incorrec!!! more meetings happned
        # 2018# incorrec!!! more meetings happned
        '20180321', '20180613', '20180926', '20181219',# incorrec!!! more meetings happned
        # 2019
        '20190130', '20190320', '20190501', '20190619', '20190731', '20190918', '20191030', '20191211',
        # 2020
        '20200129', '20200303', '20200315', '20200429', '20200610', '20200729', '20200916', '20201105', '20201216',
        # 2021
        '20210127', '20210317', '20210428', '20210616', '20210728', '20210922', '20211103', '20211215',
        # 2022
        '20220126', '20220316', '20220504', '20220615', '20220727', '20220921', '20221102', '20221214',
        # 2023
        '20230201', '20230322', '20230503', '20230614', '20230726', '20230920', '20231101', '20231213',
        # 2024
        '20240131', '20240320', '20240501', '20240612', '20240731', '20240918', # Need to update

    ]

    
    download_and_extract_fomc_press_conferences(dates)
