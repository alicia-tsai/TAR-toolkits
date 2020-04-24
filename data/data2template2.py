import pandas as pd
from urllib.request import urlopen
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
from PIL import Image

def process(source_data, dest_file, index):
    """
    source_data: csv data file, with directory to the csv file.
    dest_file: txt template.
    index: the index of event to be processed (starting at 0).
    
    In order to inspect page elemtn, we need selenium, which could be downloaded at 
    https://selenium-python.readthedocs.io/installation.html#downloading-python-bindings-for-selenium
    We used Chrome version, which could automatically open Chrome
    After downloading, run it, and fill in below the executable_path, which is
    /Users/lichenglong/Downloads/chromedriver in my case
    If the code fails, try one more time.
    """

    df = pd.read_csv(source_data)
    url = df.iloc[index]['url']
    url_intensity = url+"/shakemap/intensity"
    url_tectonic = url+"/region-info"

    # intensity image
    url_image = ""

    while (url_image == ""):
        driver = webdriver.Chrome(executable_path="/Users/lichenglong/Downloads/chromedriver")
        driver.get(url_intensity)
        content_element = driver.find_element_by_class_name('ng-star-inserted')
        content_html = content_element.get_attribute("innerHTML")

        soup = BeautifulSoup(content_html, "html.parser")
        a_tags = soup.find_all("a", href=True)


        for a in a_tags:
            if (a['href'][-4:] == ".jpg"):
                url_image = a['href']
                break

        driver.close()

    img = Image.open(urlopen(url_image))
    img.save("intensity.jpg")

    # tectonic information
    driver = webdriver.Chrome(executable_path="/Users/lichenglong/Downloads/chromedriver")
    driver.get(url_tectonic)
    content_element = driver.find_element_by_class_name('ng-star-inserted')
    content_html = content_element.get_attribute("innerHTML")
    soup = BeautifulSoup(content_html, "html.parser")
    p_tags = soup.find_all("p")

    text = open(dest_file, "a");
    for p in p_tags:
        text.write(p.getText())
    text.close()

    driver.close()

process("earthquakes_log.csv", "record.txt", 1);