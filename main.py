import os
import pandas as pd
import numpy as np
import requests
import time
import pickle
import shutil
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from statistics import median
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self,comp,year):
        self.comp = comp
        self.year = year
        self.teammap = {} # stores each teams wins, losses, and league
        #self.dfmain = pd.DataFrame()
        self.dfmain = pd.DataFrame(columns=['Team A','Team B','A Odds','B Odds',
                                            'A Wins','A Losses','B Wins','B Losses',
                                            'A LPL', 'A LCK', 'A LEC', 'A LCS',
                                            'B LPL', 'B LCK', 'B LEC', 'B LCS',
                                            'Result']) # 1 being A win        
        self.load_data()
        self.dfmain.fillna(0,inplace=True)


    def load_data(self):

        for team in self.get_teams_present():
            self.teammap[team] = Team(self.stats(team))
        # Remove VCS team(s)
        self.teammap = {k:v for k,v in self.teammap.items() if v.league != 'VCS'}
        # Add extra team spellings    
        extra = ['Jd Gaming','Edward Gaming','GenG Esports']
        og = ['JD Gaming','EDward Gaming','Gen.G Esports']
        for e,o in zip(extra,og):
            if o in self.teammap.keys():
                self.teammap[e] = self.teammap[o]
        # save to pickle
        # f = open('teammap.pckl', 'wb')
        # pickle.dump(self.teammap,f)
        # f.close()  
        # or open pickle:
        # f = open('teammap.pckl', 'rb')
        # self.teammap = pickle.load(f)
        # f.close()   

        root = str(self.year) + '/'
        filecount = 0

        for subdir, _, files in os.walk(root):
            for file in files:
                filecount += 1
                df = pd.read_json(subdir + '\\' + file,compression='bz2',lines=True)
                if self.relevant(df.loc[0,'mc'][0]):
                    shutil.copy(subdir + '/' + file, str(self.year) + 'rel\\' + file)
                    odds = self.get_odds(df)
                    self.dfmain = pd.concat([self.dfmain,self.populate_row(odds)],ignore_index=True)
            if filecount % 50 == 0:
                print(f'{filecount} files scanned')

    def relevant(self,cell):
        if 'marketDefinition' in cell:
            eventName = cell['marketDefinition']['eventName']
            if len([t for t in self.teammap.keys() if t in eventName]) == 2:
                return True
        return False

    def populate_row(self,odds):
        newrow = pd.DataFrame(data = 0, columns = self.dfmain.columns, index=[0])
        odds = [i if i != 'Jd Gaming' else 'JD Gaming' for i in odds]
        newrow.loc[0,['Team A','Team B']] = [odds[0],odds[2]]
        newrow.loc[0,['A Odds','B Odds']] = [odds[1],odds[3]]
        newrow.loc[0,['A Wins','A Losses']] = self.teammap[odds[0]].score
        newrow.loc[0,['B Wins','B Losses']] = self.teammap[odds[2]].score
        aleague = 'A ' + self.teammap[odds[0]].league
        bleague = 'B ' + self.teammap[odds[2]].league
        newrow.loc[0,[aleague,bleague]] = 1 # set a and b leagues to 1
        newrow.loc[0,'Result'] = odds[4]
        return newrow

    def get_teams_present(self):
        # Get teams present at comp year

        #URL = 'https://liquipedia.net/leagueoflegends/World_Championship/2022'
        URL = 'https://liquipedia.net/leagueoflegends/'

        if self.comp == 'worlds':
            URL += 'World_Championship/'
        elif self.comp == 'msi':
            URL += 'msi' #?
        URL += str(self.year)

        teams = []
        time.sleep(1)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content,"html.parser")
        section = soup.find('div',class_='prizepool-section-tables')
        for block in section.find_all("div", class_='block-team'):
            teams.append(block.find("span").get('data-highlightingclass'))
        # remove teams have dif league naming conventions
        toremove = ['LOUD','Isurus','Chiefs Esports Club']
        teams = [t for t in teams if t not in toremove]
        return teams

    def stats(self,team):
        # Get league of team
        league = self.get_league(team)

        # Now get domestic standings (URL example: 'https://liquipedia.net/leagueoflegends/LCS/2022/Summer')
        URL = 'https://liquipedia.net/leagueoflegends/'
        URL += league + '/' + str(self.year) + '/'

        # Non-standard leagues
        nonstd1 = {'LLA':'Opening','CBLOL':'Split 1'}
        nonstd2 = {'LLA':'Closing','CBLOL':'Split 2'}
        if self.comp == 'msi':
            URL += nonstd1.get(league,'Summer')
        elif self.comp == 'worlds':
            URL += nonstd2.get(league,'Spring')
        page = requests.get(URL)
        if page.status_code != 200: # if league does not have spring season, then change to winter
            time.sleep(1)
            URL = URL[:-6] + 'Winter'
            page = requests.get(URL)
        soup = BeautifulSoup(page.content,"html.parser")

        # Now get standings
        flag = 0
        tr = soup.find('table', class_="wikitable wikitable-bordered grouptable")
        for td in tr.find_all("td",class_=lambda x: str(x)[:3] == 'bg-'):
            if team in td.text: #if its a team name, process it
                flag = 1 # scan for next score
            elif flag == 1 and '<b>' in str(td): # if its actual standing
                hi = td.text.index('-') # hyphen index
                stats = [int(td.text[:hi]),int(td.text[hi+1:])]
            else:
                flag = 0
        stats.append(league)
        return stats

    def get_odds(self,df):
        # get odds ids from first line market def
        try: 
            oddsmap = {}
            idmap = {}
            for x in df.loc[0,'mc'][0]['marketDefinition']['runners']:
                oddsmap[x['id']] = [] # save ids and odds here
                idmap[x['id']] = x['name'] # save ids and names here
        except:
            return
        # gather all odds into a dictionary
        for row in df['mc']:
            if 'rc' in row[0]:
                for odds in row[0]['rc']:
                    oddsmap[odds['id']].append(odds['ltp'])

        # get winner
        if df.loc[len(df)-1,'mc'][0]['marketDefinition']['runners'][0]['status'] == 'WINNER':
            winner = 1
        else:
            winner = 0

        # then take median
        res = []
        for k,v in oddsmap.items():
            if idmap[k] == 'Mad Lions':
                res.append('MAD Lions')
            else:
                res.append(idmap[k])
            res.append(median(v))
        res.append(winner)
        return res # team A name, team A odds, team B name, team B odds, winner

    def get_league(self,team):
        URL = 'https://liquipedia.net/leagueoflegends/' + team + '/Results'
        page = requests.get(URL)
        time.sleep(.5)
        soup = BeautifulSoup(page.content,"html.parser")
        table = soup.find("div",class_="table-responsive")
        last = []
        if self.comp == 'worlds':
            title = str(self.year) + ' World Championship'
        elif self.comp == 'msi':
            title = 'Mid-Season Invitational ' + str(self.year)

        for tr in table.find_all("tr"):
            for td in tr.find_all("td",{'style':'width:30px'}):
                if last == title:
                    return td.get('data-sort-value')[:3]
                else:
                    last = td.get('data-sort-value')


class Team:
    def __init__(self,stats):
        self.score = stats[:2]
        self.league = stats[2]


data = Model('worlds',2021).dfmain
# f = open('data.pckl', 'rb')
# data = pickle.load(f)
# f.close()
# data = data.fillna(0)
X = data[[c for c in data.columns if c not in ['Team A', 'Team B', 'Result']]]
y = data['Result']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_probas = clf.predict_proba(X_test)

print(y_pred)
print(y_probas)
