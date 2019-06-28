# Movie Box Office Analysis and Prediction using Linear Regression

Linear Regression can be considered as the first pass in both exploratory and predictive modeling in that it is easy to interpret and provides valuable insights (if EDA is thoroughly performed). 
The motivation behind this project is to present a business case where linear regression can be used to predict revenue.

## Motivation 
A client for this project could be any stakeholder in the film industry, like movie producers, investment firms, or directors. 
Some interesting questions that can be explored includes:
* How does Domestic Grossing compare to Foreign Grossing?
* Does seasonal release influence box office returns?
* What are the most important features contributing to box office returns?

This is a good exercise for Linear Regression because there are some very logical assumptions we can make about the linear relationship between independent and dependent variables. 
With these linear regression assumptions in mind, the modeling can begin. 

See code for everything [here](link to notebook).

## Methodology
1. [Scrape the web using BeautifulSoup](#sub-heading)

2. [Clean the Dataframe](#sub-heading-2)

3. [Preliminary EDA](#sub-heading-3)

4. [Iterative Regression Modeling, Regularization, and Analysis](#sub-heading-4)


### Webscraping

For a static website, webscraping in python can be performed using BeautifulSoup. I scraped my data from BoxOfficeMojo, by first examining the HTML tags and selecting features that may influence Domestic Grossing such as Genre, Franchise, Year of Release, Distributor, etc. Here is an example code block. 

'''

    import requests
    requests.__path__
    import numpy as np
    import pandas as pd

    url = 'https://www.boxofficemojo.com/alltime/adjusted.htm'
    response = requests.get(url)
    page = response.text
    
'''

Now the HTML script is a dataframe format, and it becomes string manipulation from here. 

1. Grab all the url of the movie pages:

'''

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page, "lxml")
    
    table_soup  = soup.find_all('table')[2]
    def movies_list_url():
        urls=[]
        for tr in table_soup.find_all('tr')[2:]:
            link=tr.find('a')['href']
            url="https://www.boxofficemojo.com"+link
            urls.append(url)
        return urls
        
 '''

2. Get Movie Title from table 2 of each movie page:

'''

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
 '''


3. Get all the features from a specific table (table 5) of each movie page: 

''' 

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
    
'''

Additional features that were grabbed from each movie url were release theaters, franchise, and foreign grossing, which are found in other tables. 


### Put everything in a dataframe
The scraped data needs to be in a clean dataframe before preceeding to Exploratory Data Analysis. Some methods include converting release dates to datetime (for cases when we want to know revenue with respect to time), converting currency to int64, etc. 

''' 

    # Put all features into dictionary

    features_dict={"Title": Titles, 
                   "Domestic Grossing": Domestic_gross_int, 
                   "Foreign Grossing": foreign_converted,
                   "Release Date":Release_Date, 
                   "Genre": Genre, 
                   "Distributor": Distributor,
                   "Run Time": runtime_numeric,
                   "Rating": Ratings,
                   "Budget ($Mil)": budget,
                  "Release Theaters":theater_int_value,
                  "Part of a Franchise": franchise}

    movies=pd.DataFrame.from_dict(features_dict)

'''

### Exploratory Data Anlysis 

EDA is all about pandas manipulation, and may require domain knowledge or just having a keen eye for patterns and critical thinking. By using simple operations like groupby, count, and sort methods, we can gain some initial insights as to how the genre of the movies are distributed, and the number of titles or revenue by genre or distribution. Some categories should be combined such as "War" and "Historical", "Music" and "Musical", etc. Here is an examples below, plotted using seaborne. 

![Genre Revenues](https://github.com/carrie2010210/dsp_qiu/blob/master/LR_genre_rev.png)


### Linear Regression Models

1. Convert categorical variables 

If all categories are converted with "onehotencoding", there are 19 genres and 33 categories from Distributor, Rating, and Part of a Franchise, totaling 52 categorical features. Some feature engineering should be performed so insignificant features can be dropped or combined. 

'''

    cat_dummies=pd.get_dummies(movies_df[['Distributor','Rating','Part of a Franchise']])
    
    movies_final=pd.merge(movies_drop_cats, cat_dummies, left_index=True, right_index=True).merge(
                 genre_df,left_index=True, right_index=True)
    '''

2. Cross Validation and Regularization

Set up Linear Regression model by first normalizing the independent variables. 

'''

    from sklearn.linear_model import LinearRegression,Lasso, LassoCV, Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import train_test_split, KFold
    
    
    X = movies_final.loc[:, 'Year':'Western']
    y = movies_final['Domestic Grossing']
    
    X, X_test, y1, y1_test = train_test_split(X, y, test_size=0.2,random_state=42)
    
    ss=StandardScaler()

    X_scale=ss.fit_transform(X)
    X_scale_test=ss.transform(X_test)

 '''
 
 Set up Kfold Cross Validation for Linear Regression using the regular, polynomial, Lasso and Ridge:
 
'''

    kf= KFold(n_splits=5, shuffle=True, random_state=42)
    cv_lr, cv_poly, cv_ridge, cv_lasso = [],[],[],[]

    for train_ind, test_ind in kf.split(X_scale, y1):
         X_train, y_train = X_scale[train_ind], y1[train_ind]
         X_val, y_val = X_scale[test_ind], y1[test_ind] 
    
        # models
        lm = LinearRegression() #1
        poly = PolynomialRegression(degree=2) #2
        lm_reg = RidgeCV(cv=5) #3
        lm_las = LassoCV(cv=5) #4

        #1
        lm.fit(X_train, y_train)
        cv_lr.append(lm.score(X_val, y_val))

        # 3 
        lm_reg.fit(X_train, y_train)
        cv_ridge.append(lm_reg.score(X_val, y_val))

        #4
        lm_las.fit(X_train, y_train)
        cv_lasso.append(lm_las.score(X_val, y_val))

    print(f'Simple mean cv r^2: {np.mean(cv_lr):.3f} +- {np.std(cv_lr):.3f}')
    print(f'Ridge mean cv r^2: {np.mean(cv_ridge):.3f} +- {np.std(cv_ridge):.3f}')
    print(f'Lasso mean cv r^2: {np.mean(cv_lasso):.3f} +- {np.std(cv_lasso):.3f}')
'''


    
