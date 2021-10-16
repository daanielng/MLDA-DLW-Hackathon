import requests
from bs4 import BeautifulSoup
import re
import datetime
import parsel
from parsel import Selector
import time
import numpy as np
import pandas as pd
import os
import glob
import keyboard
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



driver = webdriver.Chrome("D:\Daniel\PMD\scripts\data preprocessing scripts\chromedriver.exe")
driver.get('https://www.sgslsignbank.org.sg/signs')


src = driver.page_source
soup = BeautifulSoup(src, 'lxml')
video_links = soup.findAll('a', href=True)

# get the links to the videos
link_lst = []
for link in video_links:
    if 'https://www.sgslsignbank.org.sg/signs/word/' in link.get('href'):
        url = link.get('href')
        name = url[len('https://www.sgslsignbank.org.sg/signs/word/'):] 
        link_lst.append((url,name))

print(link_lst)
# get the links to the gif
gif_lst = []
for link, name in link_lst:
    driver.get(link)
    src = driver.page_source
    soup = BeautifulSoup(src, 'lxml')
    image = soup.find('div', {'class': 'col-lg-7'})
    gif = image.find('img')
    gif_link = gif.get('src')

    # save gif 
    driver.get(gif_link)
    gif_name = gif_link.split('_')[1]
    keyboard.press_and_release('ctrl+s')
    time.sleep(1)
    keyboard.write(f'{name}')
    keyboard.press_and_release('enter')
    time.sleep(1)


print(gif_lst)