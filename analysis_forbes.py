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
engine =create_engine('mysql+mysqldb://root:PASSW@localhost/forbes_db')
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
plt.figure(figsize=(6,4))
industry_worth=df.groupby(['industry'])['Net_Worth'].sum().sort_values()
colors = [
    "#FFF1E5", "#F3F9E5", "#E4F1E4", "#C2E6C4", "#A2C4A2", "#6B8E3A",
    "#4A8BCA", "#7FA9D5", "#A0C4D2", "#B5D3D1", "#A8B6B1", "#8E8E8E", 
    "#6D6D6D", "#5C5C5C", "#4B4B4B", "#2E2E2E", "#1C1C1C", "#0A0A0A"
]

industry_worth.plot.barh(color=colors)
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.savefig('net_worth_barh.png')
plt.show()


# Is age correlated with Net Worth ?
plt.figure(figsize=(10, 5))  # Adjust size for better layout
age_vs_net_worth=df.groupby('Age')['Net_Worth'].sum().replace('N/A',np.nan).dropna()
age_vs_net_worth.plot.bar(color='skyblue', edgecolor='black')
plt.title('Age vs Net Worth')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.savefig('Age_vs_Net_Worth.png')

# Distribution of Billionaires Age
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
Common_Age = df['Age'].value_counts().idxmax()

plt.figure(figsize=(6, 4)) 
df['Age'].plot.kde(color='blue', linewidth=2)
plt.title('Distribution of Billionaires Age', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.axvline(Common_Age, color='red', linestyle='--', linewidth=1.5, label=f'Most Common Age: {Common_Age}')
plt.legend()
plt.tight_layout()
plt.savefig('Distribution_of_Billionaires Age.png')
plt.show()


# Number of billionaires for each Country
plt.figure(figsize=(6,6))
grouped_df = df.groupby(['Country'])['Net_Worth'].sum()
top10_nw_c = grouped_df.sort_values(ascending=False).head(10)
colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1', 
          '#955251', '#B565A7', '#009B77', '#DD4124', '#45B8AC']
ax = top10_nw_c.plot.bar(color=colors)

for index,value in enumerate(top10_nw_c):
    ax.text(index,value,f'{value:,.0f}'[:-4]+'B',ha = 'center',va = 'bottom')
plt.xticks(rotation = 30)
plt.title('Number of Billionaires for Each Country')
plt.xlabel('Country')
plt.ylabel('Net Worth (in USD) M')
print('Total Net Worth for Top 10 Countries','\n',top10_nw_c)
plt.savefig('Top 10 Countries.png')
plt.show()


# Gender Distribution of Billionaires
plt.figure(figsize=(6,4))
gender_dist = df['gender'].value_counts()
explode = [0,0.2]
gender_dist.plot.pie(autopct='%.1f%%',colors=['royalblue','red'],explode=explode,shadow=True)
plt.title('Gender Distribution of Billionaires')
plt.xlabel=''
plt.ylabel=''
plt.savefig('Gender_Distribution_of_Billionaires.png')
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