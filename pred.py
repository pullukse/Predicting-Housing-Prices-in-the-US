import pandas as pd

#cleaning datasets

fed_files = ["MORTGAGE30US.csv","RRVRUSQ156N.csv","CPIAUCSL.csv"]

dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in fed_files]

fed_data = pd.concat(dfs, axis=1)

#putting all 3 into one table

fed_data = fed_data.ffill()

#fills in empty values with quarterly values (ie. gets first row value to fill subsequent rows)

fed_data = fed_data.dropna()

#no missing values

zillow_files = ["Metro_median_sale_price_uc_sfrcondo_week.csv","Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"]

dfs = [pd.read_csv(f) for f in zillow_files]

#importing zillow datasets

dfs = [pd.DataFrame(df.iloc[0,5:])for df in dfs]

#selecting the first row on table, remove 5 columns

for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")
    
#turning date string into datetime object + joining both tables(dfs) by creating a month column



price_data = dfs[0].merge(dfs[1], on="month")

#joining tables using month



price_data.index = dfs[0].index

#organizing index


del price_data["month"]
price_data.columns = ["price", "value"]

#removing month column, renaming the remaining two columns

from datetime import timedelta

#combine both fed & zillow dfs 
#data on both come on different days (e.g. tues vs wed)


fed_data.index = fed_data.index + timedelta(days=2)

#adding days to fed days to be able to merge with zillow

price_data = fed_data.merge(price_data, left_index=True, right_index=True)

#full dataset created


price_data.columns = ["interest","vacancy", "cpi", "price", "value"]


price_data.plot.line(y="price", use_index=True)

#adjust house prices to inflation

price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100

price_data.plot.line(y="price", use_index=True)

#goal: predict the adjusted price vis a vis inflation

price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100

#predict house price for the next quarter

price_data["next_quarter"] = price_data["adj_price"].shift(-13)

#13 is num of weeks in a quarter

price_data.dropna(inplace=True)

#NaN values are not useful for training

price_data["change"] = (price_data["next_quarter"] > price_data["adj_price"]).astype(int)

#starting ML

price_data["change"].value_counts()

#checking how often price goes up vs down

predictors = ["interest", "vacancy", "adj_price", "adj_value"]

target = "change"

#predicting target (ie. change) using predictors

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

#START 260 WEEKS (IE PREV YEARS), STEP 52 WEEK (IE 1 YEAR)
START = 260
STEP = 52

def predict(train, test, predictors, target):
    rf = RandomForestClassifier(min_samples_split=10, random_state=1)
    rf.fit(train[predictors], train[target])
    preds = rf.predict(test[predictors])
    return preds

#backtest: make sure only using data from past to predict future

def backtest(data, predictors, target):
    all_preds = []
    for i in range(START, data.shape[0], STEP):
        train = price_data.iloc[:i]
        test = price_data.iloc[i:(i+STEP)]
        all_preds.append(predict(train, test, predictors, target))
        
    preds = np.concatenate(all_preds)
    return preds, accuracy_score(data.iloc[START:][target], preds)
    
    preds, accuracy = backtest(price_data, predictors, target)
    
    accuracy
    #right 58% of the time
    
    yearly = price_data.rolling(52, min_periods=1).mean()
    #ratio current price vs last year
    yearly_ratios = [p + "_year" for p in predictors]
    price_data[yearly_ratios] = price_data[predictors] / yearly[predictors]
    
    preds, accuracy = backtest(price_data, predictors + yearly_ratios, target)
    accuracy
    #improved to 64%




