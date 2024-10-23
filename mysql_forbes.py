from mysql.connector import (connection)
from forbes_analysis.scrape3 import ForbesScrape
from sqlalchemy import create_engine
import pandas as pd

sc = ForbesScrape('C:\\Users\\şerefcanmemiş\\Documents\\Projects\\forbes_analysis\\forbes_json.txt')
df=sc.forbes_df()
df
cnn = connection.MySQLConnection(user='root',password='Selincanım321.',
                                host='localhost',database='forbes_db' )
print('connection established')
cursor = cnn.cursor()
cursor.execute("use forbes_db")

engine = create_engine("mysql+mysqldb://root:Selincanım321.@localhost/forbes_db")
df.to_sql('forbes_table',engine,if_exists='replace',index=False )

cursor.execute("use forbes_db")
cursor.execute("""
SELECT * from forbes_table
""")
cnn.close()
cursor.close()
df_sql = pd.read_sql("select * from forbes_table", con = engine)
df_sql
