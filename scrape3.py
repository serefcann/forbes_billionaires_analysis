import requests
import pandas as pd
import json

class ForbesScrape():

    def __init__(self,forbes_json_path,url,cookie):
        self.forbes_json_path = forbes_json_path
        self.url = url
        self.cookie = cookie

    def json_scrape_dump(self):

        querystring = {"filter":"uri,finalWorth,age,country,source,qas,rank,category,person,personName,industries,organization,gender,firstName,lastName,squareImage,bios"}
        payload = ""
        headers = {
            "cookie":self.cookie,
            "^sec-ch-ua": "'Chromium';v='128', 'Not;A=Brand';v='24', 'Opera';v='114'",
            "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36",
            "sec-ch-ua-platform": "'Android'",
            "sec-ch-ua-mobile": "?1",
            "referer": "https://www.forbes.com/billionaires/"
        }
        response = requests.get(self.url, data=payload, headers=headers, params=querystring)
        data = response.text

        with open(self.forbes_json_path,'w+',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False)
            f.close()


    def load_json(self):
        with open(self.forbes_json_path,'r',encoding='utf-8') as f:
            content = f.read()
        return content

    def parse(self):
        content = self.load_json()
        try:
            first_parse = json.loads(content)
            if isinstance(first_parse,str):
                data = json.loads(first_parse)
            else:
                data = json.loads(content)
        except json.JSONDecodeError as e:
            print(e)
        return(data)

    def forbes_df(self):
        data = self.parse()

        person_list = data['personList']['personsLists']
        all_data = []
        for person in person_list:
            name = person.get('firstName','N/A')
            last_name = person.get('lastName','N/A')
            Country = person.get('country','N/A')
            age = person.get('age','N/A')
            gender = person.get('gender','N/A')
            finalWorth = person.get('finalWorth','N/A')
            industries = person.get('industries','N/A')

            save = {'Name':name,
                    'Last_Name':last_name,
                    'Country':Country,
                    'Age':age,
                    'Net_Worth':finalWorth,
                    'industry':industries,
                    'gender':gender
                    }
            all_data.append(save)

        df = pd.DataFrame(all_data)
        return df
if __name__ == '__main__':

    forbes_json_path = r'C:\Users\şerefcanmemiş\Documents\Projects\forbes_json.txt'
    url ="https://www.forbes.com/forbesapi/person/billionaires/2024/position/true.json"
    cookie ="datadome=K1yzUfmMbPiTisAZPhDFcaUd4ww_OBxJqtLmUA3X0yOwamOAPB4TGZSuMtJ2xVTSEDzJh5bSUX7n14Gt~dHpyQlE6bTcK7tv3hmzwUcbPlDiqVjJyOEfxmMge0b8W8JN; Max-Age=31536000; Domain=.forbes.com; Path=/; Secure; SameSite=Lax"

    s = ForbesScrape(forbes_json_path,url,cookie)
    s.forbes_df()


