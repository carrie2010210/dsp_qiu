# Movie Box Office Analysis and Prediction using Linear Regression

Linear Regression can be considered as the first pass in both exploratory and predictive modeling in that it is easy to interpret and provides valuable insights (if EDA is thoroughly performed). 
The motivation behind this project is to present a business case where linear regression can be used to predict revenue.

### Motivation 
A client for this project could be any stakeholder in the film industry, like movie producers, investment firms, or directors. 
Some interesting questions that can be explored includes:
* How does Domestic Grossing compare to Foreign Grossing by feature?
* Does seasonal release influence box office returns?
* What are the most important features contributing to box office returns?

This is a good exercise for Linear Regression because there are some very logical assumptions we can make about the linear relationship between independent and dependent variables. 
With these linear regression assumptions in mind, the modeling can begin. 


### Methodology
1. [Scrape the web using BeautifulSoup](#sub-heading)

2. [Clean the Dataframe](#sub-heading-2)

3. [Preliminary EDA and Regularization](#sub-heading-3)

4. [Feature Engineering](#sub-heading-4)

5. [Iterative Regression Modeling and Analysis](#sub-heading-5)


## Webscraping

For a static website, webscraping in python can be performed using BeautifulSoup. I scraped my data from MovieMojo, by first examining the HTML tags and selecting features that may influence Domestic Grossing such as Genre, Franchise, Year of Release, Distributor, etc. Here is an example code block. 

'''

im-port requests
requests.__path__
import numpy as np
import pandas as pd

url = 'https://www.boxofficemojo.com/alltime/adjusted.htm'
response = requests.get(url)
tables=pd.read_html(url)

'''

Now the HTML script is a dataframe format, and it becomes string manipulation from here. 

1. Grab all the url of the movie pages:

'
table_soup  = soup.find_all('table')[2]
def movies_list_url():
    urls=[]
    for tr in table_soup.find_all('tr')[2:]:
        link=tr.find('a')['href']
        url="https://www.boxofficemojo.com"+link
        urls.append(url)
    return urls
 '

2. Get Movie Title from table 2 of each movie page:

'
def get_title():
    titles=[]
    for i in movies_list_url():
        response = requests.get(i)
        page = response.text
        soup = BeautifulSoup(page, "lxml")
        try:
            title_summary=soup.find_all('table')[2] #Table with the title 
            titles.append(title_summary.find_all('td')[2].text.split("\n\n")[0]) #td contains all the features
        except:
            titles.append(np.nan)
    return titles
            
Titles=get_title()
'

3. Get all the features from a specific table (table 5) of each movie page:

'
def get_features():
    features=[]
    for i in movies_list_url():
        response = requests.get(i)
        page = response.text
        soup = BeautifulSoup(page, "lxml")
        table_summary=soup.find_all('table')[5] #Table with the 6 key features 
        for td in table_summary.find_all('td'): #td contains all the features
            features.append([td.text])
    return features

all_features=get_features()
'
4. 

## Put everything in a dataframe
The scraped data needs to be in a clean dataframe before preceeding to Exploratory Data Analysis. Some methods include converting release dates to datetime (for cases when we want to know revenue with respect to time), converting currency to....


## Exploratory Data Anlysis 

EDA is all about pandas manipulation, and may require domain knowledge or just having a keen eye for patterns and critical thinking. By using groupby and sort methods, ......


If all categories are converted with "onehotencoding", there are 19 genres and 33 categories from Distributor, Rating, and Part of a Franchise, totaling 52 categorical features. Some features must be dropped before proceeding by evaluating which are insignificant. This can be done by either intuition or lasso regression.


