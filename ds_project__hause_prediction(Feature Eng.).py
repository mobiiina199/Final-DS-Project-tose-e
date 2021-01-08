# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 19:00:43 2020

@author: mobiiina199

"""
"""# ***4.Feature engineering***

There are four assumptions associated with a linear regression model:

1-Linearity: The relationship between X and the mean of Y is linear.
2-Homoscedasticity: The variance of residual is the same for any value of X.
3-Independence: Observations are independent of each other.
4-Normality: For any fixed value of X, Y is normally distributed

and in this problem data target value (SalePrice ) not Normality , it is right Skewed
and to solve this apply log transformation on target variable when it has skewed distribution. That being said, you need to apply inverse function on top of the predicted values to get the actual predicted target value.
"""

#Transform target variable
f, ax = plt.subplots(1, figsize=(10,8))
sns.distplot(np.log(train.SalePrice), fit=norm, color='purple');
ax.set_title('SalePrice', fontsize=14)


sns.histplot(train.YearBuilt)

sns.histplot(train.SaleCondition);


#derived features
df['SaleCondition1'] = df['SaleCondition'].apply(lambda x: 1 if x=='Normal' else 0)
df['TotalSF']=df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
df['TotalPorchSF'] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
df['TotalBath'] = df["FullBath"] + df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"] + 0.5*df["HalfBath"]
df['AgeBlt']= df['YrSold'] - df['YearBuilt']
df['Numfloors']=df['1stFlrSF'].apply(lambda x: 0 if x==0 else 1) + df['2ndFlrSF'].apply(lambda x: 0 if x==0 else 1)

#
sns.distplot(df['TotalPorchSF'], color='green')
sns.distplot(df['AgeBlt'], color='red')

df.groupby('Numfloors').BedroomAbvGr.value_counts()
df['Numfloors'].value_counts().plot(kind='bar');

df['TotalBath'].value_counts().plot(kind='bar');
df['TotalSF'].plot(kind='bar');

df['SaleCondition'].value_counts().plot(kind='bar');
df['SaleCondition1'].value_counts().plot(kind='bar');
#get dummy features

both=df.loc[:,['Neighborhood','MSZoning']]
both=pd.get_dummies(both)

df=pd.get_dummies(df.drop(['Neighborhood','MSZoning'],axis=1), drop_first=True )
df=pd.concat((df,both),axis=1)
df.shape

#save Dataframe to file 
df.loc[df.SalePrice!= -1314].to_csv('E:/data/data science/data course/Final  DS Project/dataset/train-mobiiina199.csv')
columns = [column for column in df.columns if column != 'SalePrice']
df.loc[df.SalePrice == -1314 , columns].to_csv('E:/data/data science/data course/Final  DS Project/dataset/test-mobiiina199.csv')


