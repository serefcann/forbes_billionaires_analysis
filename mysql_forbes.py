import mysql.connector
from forbes_analysis.scrape3 import ForbesScrape
from sqlalchemy import create_engine
import pandas as pd

def save_db():
    sc = ForbesScrape('C:\\Users\\şerefcanmemiş\\Documents\\Projects\\forbes_analysis\\forbes_json.txt')
    df=sc.forbes_df()

    cnn = mysql.connector.connect(user='root',password='PASSW',
                                    host='localhost',database='forbes_db' )
    print('connection established')
    cursor = cnn.cursor()
    cursor.execute("use forbes_db")

    engine = create_engine("mysql+mysqldb://root:PASSW.@localhost/forbes_db")
    df.to_sql('forbes_table',engine,if_exists='replace',index=False )

    df_sql = pd.read_sql("select * from forbes_table", con = engine)
    df_sql
    cnn.close()
    cursor.close()
    print('successfully closed the connection')

