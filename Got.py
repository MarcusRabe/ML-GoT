#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:05:43 2019

@author: mariadolgusheva
assignment 

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Got = pd.read_excel('GOT_character_predictions.xlsx')

#firstly we should check do we have null data or not

Got.info()

# I found that three object columns have enough values. But we should group them for better prediction.

Got['title'].value_counts()

Got['title'] = Got['title'].fillna('none')

#I renamed titles, when first world adjective

Got['title'] = Got['title'].replace({'Grand Maester':'Maester',
                                     'Hand of the King':'King',
                                     'Last Hearth':'Hearth',
                                     'First Ranger':'Ranger',
                                     'Good Master': 'Master'})

# I split titles and keep first value, because it is the most valuable part of the title, for example, King

Got['title'] = Got['title'].apply(lambda x: x.split()[0])

# # I grouped titles which similar or closely related. For example, Bloodrider serves Khal.

Got['title'] = Got['title'].replace({'Septa':'Lady',
                                     'King in the North':'King',
                                     "Lord Commander of the Night's Watch":'Lord',
                                     'Lord of Harrenhal':'Lord',
                                     'Lord Paramount of the Trident': 'Lord',
                                     'Prince of Dorne':'Prince',
                                     'Prince of Dragonstone':'Prince',
                                     'Lord Reaper of Pyke':'Lord',
                                     'Archmaester':'Master',
                                     'Septon':'Master',
                                     'Lordsport':'Lord',
                                     'Magister':'Master',
                                     'master':'Master',
                                     'PrincessQueenDowager':'Princes',
                                     'LadyQueenDowager':'Lady',
                                     'Queen':'Prince',
                                     'Bloodrider':'Khal',
                                     'King-Beyond-the-Wall':'King',
                                     'Princess':'Prince',
                                     'Prince':'King'})

# # Next step is to keep title with significant amount of values.

Shortlist = ['none', 
             'Ser',
             'Lord',
             'King', 
             'Master',
             'Maester',
             'Lady',
             'Winterfell',
             'Khal']

Got['title'] = Got['title'].apply(lambda x: x if x in Shortlist else 'other')


Got['house'].value_counts()

# Additionally I do the same algorithm for house and culture columns

Got['house'] = Got['house'].fillna('none')

ShortlistHouse = ['none', 
             "Night's Watch",
             'House Frey ',
             'House Stark', 
             'House Targaryen',
             'House Lannister',
             'House Greyjoy',
             'House Tyrell',
             'House Martell',
             'House Osgrey',
             'Faith of the Seven',
             'House Hightower',
             'House Arryn',
             'House Bracken',
             'House Florent',
             'House Botley',
             'House Baratheon',
             'House Bolton',
             'Brave Companions',
             'House Tully',
             'Brotherhood without banners',
             'House Velaryon',
             'House Whent',
             'House Crakehall']
             
Got['house'] = Got['house'].apply(lambda x: x if x in ShortlistHouse else 'other')

Got['culture'].value_counts()
             
Got['culture'] = Got['culture'].fillna('none')

ShortlistCulture = ['none',
                    'Northmen',
                    'Ironborn',
                    'Free Folk',
                    'Valyrian',
                    'Braavosi',
                    'Ghiscari',
                    'Dornish',
                    'Dothraki',
                    'Rivermen',
                    'Valemen',
                    'Reach',
                    'Vale mountain clans',
                    'Dornishmen',
                    'Westeros',
                    'Free folk']


Got['culture'] = Got['culture'].apply(lambda x: x if x in ShortlistCulture else 'other')

# I delete the column with a lot of missing values. They are not significant for the prediction model.

Got = Got.drop(['mother'],1)
Got = Got.drop(['father'],1)
Got = Got.drop(['heir'],1)
Got = Got.drop(['spouse'],1)
Got = Got.drop(['isAliveMother'],1)
Got = Got.drop(['isAliveFather'],1)
Got = Got.drop(['isAliveHeir'],1)
Got = Got.drop(['isAliveSpouse'],1)
Got = Got.drop(['isNoble'],1)
Got = Got.drop(['isMarried'],1)

# Next step replace objective data to numeric data for three columns: culture, title, house

dummies_Got = pd.get_dummies(list(Got['title']),drop_first = True)
dummies_Got = dummies_Got.add_prefix('tit_')
Got = pd.concat(
        [Got.loc[:,:],
         dummies_Got],
         axis = 1)

dummies_Got = pd.get_dummies(list(Got['culture']),drop_first = True)
dummies_Got = dummies_Got.add_prefix('cul_')
Got = pd.concat(
        [Got.loc[:,:],
         dummies_Got],
         axis = 1)

dummies_Got = pd.get_dummies(list(Got['house']),drop_first = True)
dummies_Got = dummies_Got.add_prefix('hou_')
Got = pd.concat(
        [Got.loc[:,:],
         dummies_Got],
         axis = 1)

# To use the age and date of Birth information, we should create a new column date of death. 

import numpy as np

Got['death'] = Got.apply(lambda x: np.nan if (pd.isna(x['age']) or pd.isna(x['dateOfBirth']))\
                         else x['dateOfBirth'] + x['age'], axis = 1) 

sns.distplot(Got['death'].dropna())
Got['death'].value_counts()

#Based on the 'death' chart we can see that the current year is 305.
#Because of most of values 305. Probably characters who have this value are alive

Got['stillaliveyear305'] = Got['death'] == 305
sns.distplot(Got['death'][Got['death']<305].dropna())

#If we see closer to death values before 305 values, we can see that there are several picks. 
#It can be valuable information for prediction because probably there were crises in these years.              

crisis = Got['death'].value_counts()[Got['death'].value_counts() > 5]
Got['crisis'] = Got['death'].apply(lambda x: 1 if x in crisis.index else 0)

Got['dateOfBirth'].describe().round(2)

sns.distplot(Got['dateOfBirth'].dropna());

#a lot of rows have 6 digit numbers. And first 3 numbers 0. And we should keep 3 last numbers

Got['dateOfBirth'] = Got['dateOfBirth'] = Got['dateOfBirth'].apply(lambda x: x if np.isnan(x) else (int(str(x)[:3]) if (len(str(x).split('.')[0]) > 3) else int(str(x).split('.')[0])))

Got['dateOfBirth'].describe().round(2)
sns.distplot(Got['dateOfBirth'].dropna());

# We have a date of death, and we have a cynical age, which is not correct. 
# But we can do a new column and calculate the age.

Got['age'] = Got['age'].where(Got['age'] > 0, Got['death'] - Got['dateOfBirth'])
Got['age'].describe().round(2)
sns.distplot(Got['age'].dropna());

# To analyze books' information. 

Got['Books'] = Got['book5_A_Dance_with_Dragons'] \
               + Got['book4_A_Feast_For_Crows'] \
               + Got['book3_A_Storm_Of_Swords'] \
               + Got['book2_A_Clash_Of_Kings'] \
               + Got['book1_A_Game_Of_Thrones']
              
# As we             
d = Got.groupby(["Books", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
p = d.div(d.sum(axis = 1),
          axis = 0).plot.barh(stacked = True,
                              rot = 0,
                              figsize = (15, 8),
                              width = .5)
_ = p.set(xticklabels = "",
          xlim = [0, 1],
          ylabel = "Books",
          xlabel = "Proportion of Dead vs. Alive"),
p.legend(["Dead", "Alive"],
         loc = "upper right",
         ncol = 2,
         borderpad = -.15)                                                                             
                    
Got['DeathRelations'] = Got['numDeadRelations'].apply(lambda x: np.nan if pd.isna(x) \
                                 else (1 if x > 0 else 0))               

sns.distplot(Got['popularity'].dropna());
sns.countplot(x='isAlive', data=Got)
                   
Got.info()

Got = Got.drop(['culture'],1)
Got = Got.drop(['title'],1)
Got = Got.drop(['name'],1)
Got = Got.drop(['house'],1)

Got.info()

# We have just numeric value

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

x = Got.drop(['isAlive'],1)
y = Got['isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size = 0.1,
            random_state = 508)

import xgboost as xgb

pred = xgb.XGBClassifier(objective = 'binary:logistic',
                          max_depth = 3,
                          n_estimators = 165,
                          learning_rate = 0.27,
                          min_child_weight = 1.0,
                          subsample = 1.0,
                          colsample_bytree = 0.9,
                          eval_metric = 'auc').fit(X_train, y_train)
score = xg_reg.score(X_test,
                     y_test)
train_score = xg_reg.score(X_train,
                           y_train)
diff = train_score - score

print("Testing score: %f" % (score))
print("Training score: %f" % (train_score))
print("Diff: %f" % (diff))

