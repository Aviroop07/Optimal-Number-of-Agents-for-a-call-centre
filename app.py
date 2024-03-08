# import subprocess
# subprocess.check_call(["pip", "install", "-r", "requirements.txt","--verbose"])
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import MO
from prophet import Prophet
from fancyimpute import KNN
SHIFT = 12
# import subprocess
# subprocess.check_call(["/home/adminuser/venv/bin/python","-m","pip", "install", "--upgrade","pip"])
# subprocess.check_call(["pip", "install", "fbprophet"])
# header 
st.header("Call Center Prediction",divider="rainbow")

# ---------------------------------------------------------------------------------------------

# input data preprocessing
invoice_aggregate = pd.read_excel('INVOICES_AGG.xlsx')
invoice_aggregate['TRUNC(CURRENT_ISSUE_DATE)'] = pd.to_datetime(invoice_aggregate['TRUNC(CURRENT_ISSUE_DATE)'])
refined_aggregated_data = pd.read_excel('CS_CALLS_VW_OFFERED.xlsx')
refined_aggregated_data.fillna(0,inplace=True)
refined_aggregated_data['SUM(TOTALCALLS)'] = refined_aggregated_data['SUM(CALLSOFFERED)'] - refined_aggregated_data['SUM(CALLSDEQUEUED)']
refined_aggregated_data.rename(columns={'TRUNC(DATETIME)':'ds'},inplace=True)
refined_aggregated_data = pd.merge(left=refined_aggregated_data,right=invoice_aggregate,left_on='ds',right_on='TRUNC(CURRENT_ISSUE_DATE)',how='left')
refined_aggregated_data.drop(columns=['TRUNC(CURRENT_ISSUE_DATE)','SUM(CALLSOFFERED)','SUM(CALLSDEQUEUED)'],inplace=True)
min_date = refined_aggregated_data['ds'].min()
max_date = refined_aggregated_data['ds'].max()
all_dates = {}
all_dates['ds'] = pd.date_range(start=min_date,end=max_date,freq='D')
all_dates = pd.DataFrame(all_dates)
refined_aggregated_data = pd.merge(left=all_dates,right=refined_aggregated_data,on='ds',how='left')
refined_aggregated_data.sort_values(by='ds',inplace=True)
dates = refined_aggregated_data['ds']
data = refined_aggregated_data.drop(columns=['ds'])
# st.dataframe(data)
knn_imputer = KNN(k=100)
# st.dataframe(knn_imputer.fit_transform(data))
data_filled = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)
# st.dataframe(data_filled)
refined_aggregated_data = pd.concat([dates, data_filled], axis=1)
# st.dataframe(refined_aggregated_data_filled)
refined_aggregated_data['ds'] = pd.to_datetime(refined_aggregated_data['ds'])
data_copy = refined_aggregated_data.copy()
data_copy['ds'] = pd.to_datetime(data_copy['ds'])
# st.write("Initial data : ")
# st.dataframe(refined_aggregated_data,use_container_width=True,hide_index=True)
refined_aggregated_data['ds'] = refined_aggregated_data['ds'].shift(-SHIFT)
refined_aggregated_data['SUM(TOTALCALLS)'] = refined_aggregated_data['SUM(TOTALCALLS)'].shift(-SHIFT)
refined_aggregated_data = refined_aggregated_data.iloc[12:-12]
# st.dataframe(refined_aggregated_data.isna().sum())
# refined_aggregated_data['ds'] = pd.to_datetime(refined_aggregated_data['ds'])
# st.write("Our input data frame : ")
# st.dataframe(refined_aggregated_data,hide_index=True,use_container_width=True)

# ---------------------------------------------------------------------------------------------

# holiday dataframe creation

def thanksgiving_day(year):
    # Find the fourth Thursday of November
    november_1 = datetime(year, 11, 1)
    offset = (3 - november_1.weekday() + 7) % 7
    thanksgiving = november_1 + timedelta(days=offset)
    return thanksgiving

def veterans_day(year):
    veterans_day = datetime(year, 11, 11)
    if veterans_day.weekday() == 5:  # If it falls on a Saturday
        return veterans_day + timedelta(days=2)  # Move to Monday
    elif veterans_day.weekday() == 6:  # If it falls on a Sunday
        return veterans_day + timedelta(days=1)  # Move to Monday
    else:
        return veterans_day

def memorial_day(year):
    # Last Monday of May
    memorial_day = datetime(year, 5, 31)
    while memorial_day.weekday() != 0:  # Go back to Monday
        memorial_day -= timedelta(days=1)
    return memorial_day

def generate_holidays_df(start_year, end_year):
    holidays = {
        "Christmas": "12-25",
        "Thanksgiving Day": thanksgiving_day,
        "Veterans Day": veterans_day,
        "Columbus Day": lambda year: datetime(year, 10, 12) + relativedelta(weekday=MO(2)),
        "Labor Day": lambda year: datetime(year, 9, 1) + relativedelta(weekday=MO(1)),
        "Independence Day": "07-04",
        "Memorial Day": memorial_day,
        "Washington's Birthday": lambda year: datetime(year, 2, 1) + relativedelta(weekday=MO(3)),
        "Martin Luther King, Jr. Day": lambda year: datetime(year, 1, 1) + relativedelta(weekday=MO(3)),
        "New Year's Day": "01-01",
        "World Tennis Day": "03-04",
        "NASCAR Day": "05-17",
        "National Soccer Day": "07-28",
        "American Football Day": "11-05",
        "National Basketball Day": "11-06"
    }

    holiday_dates = []
    holiday_names = []
    for year in range(start_year, end_year + 1):
        for holiday, date_or_func in holidays.items():
            if callable(date_or_func):
                date = date_or_func(year).strftime("%m-%d")
                holiday_dates.append(f"{year}-{date}")
                holiday_names.append(holiday)
            else:
                holiday_dates.append(f"{year}-{date_or_func}")
                holiday_names.append(holiday)

    holidays_df = pd.DataFrame({
        "holiday": holiday_names,
        "ds": pd.to_datetime(holiday_dates)
    })

    return holidays_df

holidays = generate_holidays_df(2015, 2024)
sundays_with_zero = refined_aggregated_data[(refined_aggregated_data['ds'].dt.dayofweek == 6) & (refined_aggregated_data['SUM(TOTALCALLS)'] <= 100)]
sundays_holidays_df = pd.DataFrame({
    'holiday': ['Sunday'] * len(sundays_with_zero),
    'ds': sundays_with_zero['ds']
})
holidays = pd.concat([holidays,sundays_holidays_df])
# st.write("Our holiday dataframe : ")
# st.dataframe(holidays,use_container_width=True,hide_index=True)


# ---------------------------------------------------------------------------------------------

# taking date input from user 

st.date_input(value=max_date,label="Enter the date:",key="target_date",max_value=max_date,min_value=refined_aggregated_data.iloc[1015]['ds'])
target_date = pd.to_datetime(st.session_state.target_date)

st.number_input(value=SHIFT,label="Number of days of forecasting",key="number_of_days",max_value=SHIFT,min_value=1)  
number_of_days = st.session_state.number_of_days

# ---------------------------------------------------------------------------------------------


# creating future data frame
future_df = data_copy.drop(columns=['SUM(TOTALCALLS)'])
future_df = future_df[future_df['ds']<=target_date]
future_df.sort_values(by='ds',inplace=True,ascending=False)
future_df = future_df.iloc[:2*SHIFT]
future_df.sort_values(by='ds',inplace=True)
future_df = future_df.iloc[:number_of_days+SHIFT-1]
future_df['ds'] = future_df['ds'].apply(lambda x : x+pd.Timedelta(days=SHIFT))
# st.write("Our future dataframe : ")
# st.dataframe(future_df,hide_index=True,use_container_width=True)


# ---------------------------------------------------------------------------------------------

# training model on input data
refined_aggregated_data.rename(columns={'SUM(TOTALCALLS)':'y'},inplace=True)
model = Prophet(weekly_seasonality=True,yearly_seasonality=True,holidays=holidays,interval_width=0.8)
for col in [column for column in refined_aggregated_data.columns if column!='ds' and column!='y']:
    model.add_regressor(col)
model.fit(refined_aggregated_data)

# ---------------------------------------------------------------------------------------------

# predicting future values
prediction = model.predict(future_df)
future_df['yhat'] = prediction['yhat'].apply(lambda x : round(max(0,x),0)).values
future_df['yhat_lower'] = prediction['yhat_lower'].apply(lambda x : round(max(0,x),0)).values
future_df['yhat_upper'] = prediction['yhat_upper'].apply(lambda x : round(max(0,x),0)).values
cond1 = (future_df['yhat_lower'] <= 200)
cond2 = (future_df['yhat'] - future_df['yhat_lower']) >= 1500
cond3 = (future_df['yhat']<=1000)
cond4 = (future_df['yhat']>=4000)
cond5 = (future_df['yhat_upper']>=6000)
cond8 = (future_df['yhat_upper']<=3000)
cond6 = cond1 & cond2 & cond8 | cond3
cond7 = cond4 & cond5
cond9 = cond1 & cond2 & ~cond8
future_df['actual_pred'] = np.where(cond6,future_df['yhat_lower'],(0.25*future_df['yhat']+0.1*future_df['yhat_upper']+0.65*future_df['yhat_lower']))
future_df['actual_pred'] = np.where(cond7,(0.4*future_df['yhat']+0.6*future_df['yhat_upper']),future_df['actual_pred'])
future_df['actual_pred'] = np.where(cond9,future_df['yhat'],future_df['actual_pred'])
st.write("Predictions ")
cond10 = future_df['actual_pred'].shift(1) <= 200
cond11 = future_df['actual_pred'].shift(2) <= 200
cond12 = future_df['actual_pred'].shift(3) <= 200
cond13 = future_df['actual_pred'].shift(4) <= 200
cond14 = future_df['actual_pred'].shift(5) <= 200
cond15 = cond10 | cond11 | cond12 | cond13 | cond14
future_df['actual_pred'] = np.where(cond15,0.75*future_df['actual_pred'],future_df['actual_pred'])
cond11 = cond1 & cond8
future_df['actual_pred'] = np.where(cond11,future_df['yhat_lower'],future_df['actual_pred'])
future_df = future_df[future_df['ds']>=target_date]
future_df = pd.merge(future_df,refined_aggregated_data,on='ds',how='left')
st.dataframe(future_df[['ds','actual_pred','y']],hide_index=True,use_container_width=True)


# ---------------------------------------------------------------------------------------------