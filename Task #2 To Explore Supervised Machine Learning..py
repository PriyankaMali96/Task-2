#!/usr/bin/env python
# coding: utf-8

# # 1) Import the Data

# In[1]:


#Important libraries to import for further analysis.
import pandas as pd


# In[2]:


D="http://bit.ly/w-data"
data=pd.read_csv(D)
print(data)
print("Data loaded successfully!")


# In[3]:


# Now, we will obtain a scatter plot to get the idea about the relationship between study hours and percentage scores.
import matplotlib.pyplot as plt  
data.plot(x='Hours',y='Scores',style="o")
plt.title('Hours Studied vs Percentage Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()    


# In[4]:


# We will check is there any outliers present or not. To do that we will draw a Boxplot.
box_plot_data=[data['Hours'],data['Scores']]
plt.boxplot(box_plot_data,patch_artist=True,labels=['Hours','Scores'])
plt.show()
print("Since, no data point present is out of the upper and lower Whisker's therefore no outlies outliers are present.")


# In[5]:


# Our next step is to divide the data into “attributes” and “labels”.
# Attributes are the independent variables while labels are dependent variables whose values are to be predicted. 
# In our dataset, we only have two columns. 
# We want to predict the Scores depending upon the Hours studied. 
# Therefore our attribute set will consist of the “Hours” column which is stored in the X variable, and the label will be the “Scores” column which is stored in y variable.
x = data['Hours'].values.reshape(-1,1)
y = data['Scores'].values.reshape(-1,1)

# Fit the regression
from sklearn.linear_model import LinearRegression  
Model=LinearRegression()
Model.fit(x,y)


# In[6]:


# To obtain the coefficient of determination (𝑅²) 
R_sq = Model.score(x, y)
print('coefficient of determination:', R_sq)
print("From coefficient of determination we can conclude that 'Studied Hours' and 'Percentage Scores' possessing strong relationship" )


# In[17]:


# Plotting the regression line
line = Model.coef_*x+Model.intercept_
# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# In[8]:


y_pred = Model.predict(x)
print('predicted response:', y_pred, sep='\n')


# In[27]:


# You can also test with your own data
import numpy as np
H=np.reshape(9.25,(1,1))
print(H)
own_pred = Model.predict(H)
own_pred


# In[ ]:




