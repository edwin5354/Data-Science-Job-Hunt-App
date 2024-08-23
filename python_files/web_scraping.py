from bs4 import BeautifulSoup
import requests
import pandas as pd

def getTitle(soup):
    try:
        title = soup.find('h1', attrs = {'data-automation': 'job-detail-title'}).get_text(strip=True)
    except AttributeError:
        title = ""
    return title

def companyName(soup):
    try:
        company = soup.find('span', attrs = {'data-automation': 'advertiser-name'}).get_text(strip=True)
    except AttributeError:
        company =''
    return company

def findDistrict(soup):
    try:
        district = soup.find('span', attrs = {'data-automation': 'job-detail-location'}).get_text(strip=True)
    except AttributeError:
        district = ''
    return district

def jobClassify(soup):
    try:
        classify = soup.find('span', attrs = {'data-automation': 'job-detail-classifications'}).get_text(strip=True)
    except AttributeError:
        classify = ''
    return classify

def workType(soup):
    try:
        work = soup.find('span', attrs = {'data-automation': 'job-detail-work-type'}).get_text(strip=True)
    except AttributeError:
        work = ''
    return work

# This column requires a lot of work for extracting the job requirements and qualifications --> need to clean it using NLP 
def jobDesc(soup):
    try:
        description = soup.find('div', attrs = {'class': '_4603vi0 pcx2080'}).get_text(strip=True)
    except AttributeError:
        description =''
    return description

if __name__ == '__main__':
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
    HEADERS = {'User-Agent': user_agent, 'Accept-Language': 'en-US, en;q=0.5'}
    jobs_dict = {'title':[], 'company':[], 'district': [], 'classification': [], 'work-type': [], 'description': []}
    for i in range(1,6):
        url = f"https://hk.jobsdb.com/junior-data-scientist-jobs?page={i}"
        webpage = requests.get(url, headers=HEADERS)
        main_soup = BeautifulSoup(webpage.content, 'html.parser')

        new_link = main_soup.find_all('a', attrs= {'data-automation': "job-list-item-link-overlay"})
        sub_web = []
        for each_link in new_link:
            sub_web.append(each_link.get('href'))

        for new_link in sub_web:
            new_page = requests.get('https://hk.jobsdb.com'+ new_link, headers=HEADERS)
            new_soup = BeautifulSoup(new_page.content, 'html.parser')

            jobs_dict['title'].append(getTitle(new_soup))
            jobs_dict['company'].append(companyName(new_soup))
            jobs_dict['district'].append(findDistrict(new_soup))
            jobs_dict['classification'].append(jobClassify(new_soup))
            jobs_dict['work-type'].append(workType(new_soup))
            jobs_dict['description'].append(jobDesc(new_soup))

    jobs_df = pd.DataFrame.from_dict(jobs_dict)
    jobs_df.to_csv('./csv/raw.csv', index = False)