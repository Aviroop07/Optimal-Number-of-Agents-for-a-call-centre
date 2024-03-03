import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import MO
SHIFT = 12
import subprocess
def install_package(package_name):
    try:
        # Check if the package is already installed
        subprocess.check_call(["python", "-m", "pip", "show", package_name])
    except subprocess.CalledProcessError:
        # If the package is not installed, install i
        subprocess.check_call(["python", "-m", "pip", "install", package_name])

# Call the function with the package name you want to install
install_package("prophet")
from prophet import Prophet

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
refined_aggregated_data = pd.merge(left=refined_aggregated_data,right=invoice_aggregate,left_on='ds',right_on='TRUNC(CURRENT_ISSUE_DATE)',how='outer')
refined_aggregated_data.drop(columns=['TRUNC(CURRENT_ISSUE_DATE)','SUM(CALLSOFFERED)','SUM(CALLSDEQUEUED)'],inplace=True)
refined_aggregated_data.sort_values(by='ds',inplace=True)
# refined_aggregated_data.drop(refined_aggregated_data.tail(1),inplace=True)
# refined_aggregated_data = refined_aggregated_data[refined_aggregated_data['ds']>=pd.to_datetime("2016-03-01")]
# st.dataframe(refined_aggregated_data.isnull().sum()/len(refined_aggregated_data),use_container_width=True)
min_date = refined_aggregated_data['ds'].min()
max_date = refined_aggregated_data['ds'].max()
all_dates = {}
all_dates['ds'] = pd.date_range(start=min_date,end=max_date,freq='D')
all_dates = pd.DataFrame(all_dates)
refined_aggregated_data = pd.merge(left=all_dates,right=refined_aggregated_data,on='ds',how='outer')
refined_aggregated_data.fillna(0,inplace=True)
refined_aggregated_data['ds'] = pd.to_datetime(refined_aggregated_data['ds'])
data_copy = refined_aggregated_data.copy()
data_copy['ds'] = pd.to_datetime(data_copy['ds'])
refined_aggregated_data.sort_values(by='ds',inplace=True)
st.write("Initial data : ")
st.dataframe(refined_aggregated_data,use_container_width=True,hide_index=True)
refined_aggregated_data['ds'] = refined_aggregated_data['ds'].shift(-SHIFT)
refined_aggregated_data['SUM(TOTALCALLS)'] = refined_aggregated_data['SUM(TOTALCALLS)'].shift(-SHIFT)
refined_aggregated_data = refined_aggregated_data.iloc[12:-12]
# st.dataframe(refined_aggregated_data.isna().sum())
# refined_aggregated_data['ds'] = pd.to_datetime(refined_aggregated_data['ds'])
st.write("Our input data frame : ")
st.dataframe(refined_aggregated_data,hide_index=True,use_container_width=True)

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
st.write("Our holiday dataframe : ")
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
future_df = future_df.iloc[:SHIFT+1]
future_df.sort_values(by='ds',inplace=True)
future_df = future_df.iloc[:number_of_days]
future_df['ds'] = future_df['ds'].apply(lambda x : x+pd.Timedelta(days=SHIFT))
st.write("Our future dataframe : ")
st.dataframe(future_df,hide_index=True,use_container_width=True)


# ---------------------------------------------------------------------------------------------

# training model on input data
refined_aggregated_data.rename(columns={'SUM(TOTALCALLS)':'y'},inplace=True)
model = Prophet(weekly_seasonality=True,yearly_seasonality=True,holidays=holidays)
for col in [column for column in refined_aggregated_data.columns if column!='ds' and column!='y']:
    model.add_regressor(col)
model.fit(refined_aggregated_data)

# ---------------------------------------------------------------------------------------------

# predicting future values
prediction = model.predict(future_df)
future_df['yhat'] = prediction['yhat'].values
future_df['yhat_lower'] = prediction['yhat_lower'].values
future_df['yhat_upper'] = prediction['yhat_upper'].values
st.write("Predictions ")
st.dataframe(future_df[['ds','yhat','yhat_upper','yhat_lower']],hide_index=True,use_container_width=True)


# ---------------------------------------------------------------------------------------------
