''' Importing datetime module to setup the start and end dates
    of what we want to download from finance.yahoo.com/quote/ '''
# import datetime as dt

''' Note: pandas_datareader did not come with the installation of Anoconda '''
#import pandas_datareader.data as web 

''' dt.datetime(YEAR, MONTH, DAY)
    Start is the date '01-01-1998'''
# start = dt.datetime(1998, 1, 1)
# end = dt.datetime(2017, 12, 31)

''' web.DataReader('STOCK_TICKER', 'WEBSITE', START_DATE, END_DATE)

    START_DATE and END_DATE must be of type datetime '''
# data = web.DataReader('NVDA', 'yahoo', start , end)

''' to_csv('PATH/NAME_OF_FILE.csv')
 
    if you want to save this csv in the local directory then this 
    is enough to_csv('NAME_OF_FILE.csv').
    Note: '~/desktop' means 'Home_Directory/Desktop'
    In english'~/desktop/NDVA.csv': 'Go to this users home folder(~)
    and once in that folder go to the desktop folder and within the 
    desktop folder save this csv file and name it 'NDVA.csv' '''
# data.to_csv('~/desktop/NDVA.csv')



''' These next 3 lines are standard. Always import them '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' These next two lines are optional, it just to make the graphs look nice '''
from matplotlib import style
style.use('ggplot')


''' pd.read_csv('PATH') returns a pandas DataFrame from a csv file
    think of a DataFrame as pretty much a matrix '''
data = pd.read_csv('~/Desktop/NDVA.csv')

''' DataFrame.head() return the first 5 rows of every column in the matrix.
    Its very useful when getting comfortable with your data'''
print(data.head())
''' DataFrame.describe() returns summary statistics of every column, also
    very useful when exploring the dataset. '''
print(data.describe())

''' DataFrame.iloc[START_ROW_IDX : END_ROW_IDX, START_COL_IDX : END_COL_IDX] 
    iloc is short for 'integer location', it returns a matrix (DataFrame)
    for the specified indexes.
        Ex. if you want the first 4 rows and first 4 columns of a DataFrame:
            DataFrame.iloc[0:4, 0:4]
    Note: the rows '0:4' is referring  to the 0'th 1'st 2'nd and 3'rd rows, it
    does not include the 4'th index.
    
    The variable 'y' is being assigned as a DataFrame containing  all of the rows
    of our dataset's 5th column. Remember that python start counting at 0, so 
    our 5th column is the 4th index. Also remember that 4:5 does not include 
    the 5th index, to python the 5th index is our 6th column.'''
y = data.iloc[:,4:5]            # Closing prices from 01-22-1988 to present
y.plot()                        # Generates a graph
plt.show()                      # Shows the graph
''' y_12 is assigned as a DataFrame containing the rows 0-3258 of the 5th Column. 
    Our 5th column contains NVIDIA's closing prices from 01/22/1998 - 01/01/2012'''
y_12 = data.iloc[0:3258, 4:5]   # Closing prices from 01-22-1998 to 01-01-2012
y_14 = data.iloc[0:3760, 4:5]   # Closing prices from 01-22-1998 to 01-01-2014

''' Next we will create a DataFrame that looks like [0, 1, 2, 3,........., N],
    Why?
    Regressions require independent variables. We will represent 'time' as we 
    did before in lines ##=##, 0 == '01-22-1998', 1 == '01-23-1998' and so on..
    Why time? 
    Because  we are modeling NVDIA's closing prices through time
    
    pd.DataFrame(LIST_LIKE)
    pd.DataFrame takes a list like data structure as its input: lists, 
    dictionaries, sets.. We want a list that looks like [[0,1,2....]]
    
    list(SEQUENCE)
    list() is a python 'built-in' function (standard python), it takes a
    sequence of things as an input and returns a list from it. 
    
    range(INT)
    range() is also a built in function, it takes an integer as an input 
    and returns a list from 0 to that number. It can do a lot more but, for now,
    we will only use it for that.
    
    len(LIST_LIKE)
    len(), also a built in function, takes a list like structure: list, sets ...
    and returns the size of it. len([2, 4, 1]) will return a 3
    
    Putting that together we get X as a DataFrame = [0, 1, 2, .... N] '''
X = pd.DataFrame(list(range(len(y_12))))


# Part two of the lecture, No Comments


from sklearn.linear_model import LinearRegression
reg = LinearRegression()


reg.fit(X, y_12)

y_hat = pd.DataFrame(list(range(len(y_14))))
# not needed
y_hat_total = pd.DataFrame(list(range(len(y))))


def predF(df):
	for i, x in enumerate(df):
		df[i] = reg.intercept_+ reg.coef_* i

        
y_hat.apply(predF)
#Not needed
#y_hat_total.apply(predF)

print(reg.intercept_)
print(reg.coef_)


#plt.plot(y, color='Yellow')
plt.plot(y_14)
plt.plot(y_12, color='Blue')
#plt.plot(y_hat_total, color='Green')
plt.plot(y_hat)
plt.show()

import statsmodels.api as sm
model = sm.OLS(y_14, y_hat).fit()
predictions = model.predict(X) 
print(model.summary())

