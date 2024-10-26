import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


conn = mysql.connector.connect(user = 'root', password = 'PASSW',
                                  host = 'localhost',database='forbes_db')
cursor = conn.cursor()
engine =create_engine('mysql+mysqldb://root:PASSW.@localhost/forbes_db')
df = pd.read_sql("Select * from forbes_table",con=engine)
cursor.close()
conn.close

df
df.describe()
print(df.dtypes)

df['Age'] = pd.to_numeric(df['Age'],errors='coerce',downcast='integer')

df.isna().sum()
np.unique(df['Age']) # We have nan values

freq = df['Net_Worth'].value_counts()
freq.max()
freq.idxmax()

# analyzing Net Worth for each industry and frequence
industry_worth=df.groupby(['industry'])['Net_Worth'].sum()

fig, axs = plt.subplots(ncols=2,figsize = (8,12))
sns.histplot(df['Net_Worth'], bins=30, kde=True,ax=axs[0])
industry_worth.plot.pie(ax=axs[1])
axs[1].set_title('industries')
axs[1].set_xlabel('')
axs[1].set_ylabel('')
plt.savefig('net_worth_pie.png')
plt.show()


# Is age correlated with Net Worth ?
Common_Age = df['Age'].value_counts().idxmax()

fig, axs = plt.subplots(ncols=2,figsize=(8,12))
df.plot(x='Age',y='Net_Worth',kind='scatter',ax=axs[0])
df['Age'].plot.kde(ax=axs[1]) # normal distribution
axs[1].set_title('Distribution of Billionaires Age')
axs[0].set_title('Age vs Net Worth')
print(f'the most common age for billionaires: {Common_Age}') # 61
plt.savefig('Age_vs_Net_Worth.png')
plt.show()


# Number of billionaires for each Country
grouped_df = df.groupby(['Country'])['Net_Worth'].sum()
top10_nw_c = grouped_df.sort_values(ascending=False).head(10)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
ax = top10_nw_c.plot.bar(color=colors)

for index,value in enumerate(top10_nw_c):
    ax.text(index,value,f'{value:,.0f}'[:-4]+'B',ha = 'center',va = 'bottom')
plt.xticks(rotation = 45)
plt.title('Number of Billionaires for Each Country')
plt.xlabel('Country')
plt.ylabel('Net Worth (in USD) M')
print('Total Net Worth for Top 10 Countries','\n',top10_nw_c)
plt.savefig('Top 10 Countries.png')
plt.show()


# Gender Distribution of Billionaires
gender_dist = df['gender'].value_counts()
explode = [0,0.2]
gender_dist.plot.pie(autopct='%.1f%%',colors=['royalblue','red'],explode=explode,shadow=True)
plt.title('Gender Distribution of Billionaires')
plt.xlabel=''
plt.ylabel=''
plt.savefig('Gender Distribution of Billionaires.png')
plt.show()



sns.pairplot(df, y_vars='Net_Worth', x_vars=['gender','Age','Country'], height=4, aspect=1.2)
plt.show()

df.columns
df = df[['gender','Age','industry','Country','Net_Worth']]
df['Country'].unique()

# Predicting 

df.replace('N/A',np.nan,inplace=True)
df.dropna(how='all',inplace=True,axis=0)


# Ensure 'gender' is mapped correctly to 0 and 1
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# Define the ColumnTransformer with the correct structure
transformer = ColumnTransformer(
    transformers=[
        # Impute missing values in 'Age'
        ('age_imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent'), ['Age']),
        
        # One-hot encode 'industry'
        ('industry_encoder', OneHotEncoder(sparse_output=False, drop='first'), ['industry']),
        
        # Ordinal encode 'Country'
        ('country_encoder', OrdinalEncoder(
            categories=[['France', 'United States', 'India', 'Spain', 'Mexico', 'Canada',
                         'China', 'Belgium', 'Indonesia', 'Singapore', 'Japan', 'Austria',
                         'Switzerland', 'Germany', 'Hong Kong', 'United Arab Emirates',
                         'United Kingdom', 'Australia', 'Russia', 'Chile', 'Monaco',
                         'Czech Republic', 'Sweden', 'N/A', 'Nigeria', 'Uzbekistan',
                         'Thailand', 'Denmark', 'South Africa', 'South Korea', 'Italy',
                         'Taiwan', 'Philippines', 'Brazil', 'New Zealand', 'Israel',
                         'Malaysia', 'Egypt', 'Norway', 'Colombia', 'Eswatini (Swaziland)',
                         'Poland', 'Greece', 'Argentina', 'Netherlands', 'Kazakhstan',
                         'Portugal', 'Cayman Islands', 'Turkey', 'Uruguay', 'Georgia',
                         'Luxembourg', 'Vietnam', 'Latvia', 'Ukraine', 'Finland', 'Bermuda',
                         'Cyprus', 'Lebanon', 'Ireland', 'Guernsey', 'Algeria', 'Bahamas',
                         'Oman', 'British Virgin Islands', 'Romania', 
                         'Turks and Caicos Islands', 'Qatar', 'Nepal', 'Tanzania',
                         'Slovakia', 'Morocco', 'Hungary', 'Bahrain', 'Andorra',
                         'Liechtenstein', 'Armenia']],
            handle_unknown='use_encoded_value', unknown_value=-1
        ), ['Country'])
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

transformed_data = transformer.fit_transform(df)
print(f'Transformed Data Shape: {transformed_data.shape}')

ohe_columns = transformer.named_transformers_['industry_encoder'].get_feature_names_out(['industry'])
print(len(ohe_columns))

columns = ['Age'] + list(ohe_columns) + ['Country','gender','Net_Worth']

df = pd.DataFrame(transformed_data,columns=columns)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
x = df.drop('Net_Worth',axis=1)
y = df['Net_Worth']
x
x

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)

df.describe()
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)


pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Scale the features
    ('model', LinearRegression())
])

pipeline.fit(x, y)
y_pred = pipeline.predict(x)
r2_score(y,y_pred)


model = LinearRegression()
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validated R²: {scores.mean()}")


# Conclusion: attributes not associated with Net Worth