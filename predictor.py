#!/usr/bin/env python

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
import argparse
import sys

def load_model():
    try:
        if not os.path.exists('flight_delay_xgboost_model.json'):
            print("Error: Model file 'flight_delay_xgboost_model.json' not found.")
            sys.exit(1)
            
        if not os.path.exists('flight_delay_xgboost_scaler.pkl'):
            print("Error: Scaler file 'flight_delay_xgboost_scaler.pkl' not found.")
            sys.exit(1)
            
        model = xgb.Booster()
        model.load_model('flight_delay_xgboost_model.json')
        
        with open('flight_delay_xgboost_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print("Model and scaler loaded successfully!\n")
        return model, scaler
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def get_user_inputs():
    print("\n===== Flight Delay Prediction Tool =====")
    print("Please enter the following flight details:\n")
    
    try:
        while True:
            date_input = input("Flight date (YYYY-MM-DD): ")
            try:
                flight_date = datetime.strptime(date_input, '%Y-%m-%d')
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD format.")
        
        print("\nAirline Codes:")
        airlines = {
            'AA': 'American Airlines',
            'DL': 'Delta Air Lines',
            'UA': 'United Airlines',
            'WN': 'Southwest Airlines',
            'B6': 'JetBlue Airways',
            'AS': 'Alaska Airlines',
            'NK': 'Spirit Airlines',
            'F9': 'Frontier Airlines',
            'HA': 'Hawaiian Airlines',
            'G4': 'Allegiant Air',
            '9E': 'Endeavor Air',
            'OH': 'PSA Airlines',
            'YX': 'Republic Airways',
            'MQ': 'Envoy Air',
            'OO': 'SkyWest Airlines'
        }
        
        for code, name in airlines.items():
            print(f"  {code}: {name}")
            
        while True:
            carrier = input("\nAirline code: ").upper()
            if carrier in airlines:
                break
            print("Invalid airline code. Please choose from the list above.")
        
        while True:
            try:
                flight_num = int(input("\nFlight number: "))
                if flight_num > 0 and flight_num < 10000:
                    break
                print("Invalid flight number. Please enter a number between 1 and 9999.")
            except ValueError:
                print("Please enter a valid number.")
        
        print("\nAirport Codes (Major airports shown below, but any valid code can be entered):")
        major_airports = {
            'ATL': 'Atlanta', 'DFW': 'Dallas/Fort Worth', 'DEN': 'Denver', 
            'ORD': 'Chicago O\'Hare', 'LAX': 'Los Angeles', 'CLT': 'Charlotte',
            'LAS': 'Las Vegas', 'PHX': 'Phoenix', 'MCO': 'Orlando', 'SEA': 'Seattle',
            'MIA': 'Miami', 'IAH': 'Houston', 'JFK': 'New York JFK', 'EWR': 'Newark',
            'SFO': 'San Francisco', 'DTW': 'Detroit', 'BOS': 'Boston', 'MSP': 'Minneapolis',
            'FLL': 'Fort Lauderdale', 'PHL': 'Philadelphia', 'LGA': 'New York LaGuardia',
            'BNA': 'Nashville', 'IAD': 'Washington Dulles', 'DCA': 'Washington Reagan',
            'SLC': 'Salt Lake City', 'SAN': 'San Diego', 'MDW': 'Chicago Midway'
        }
        
        airport_metadata = {
            'ATL': {'id': 10397, 'city': 'Atlanta', 'state': 'GA', 'state_name': 'Georgia', 
                   'lat': 33.6367, 'lon': -84.4281, 'alt': 1026, 'tz': 'America/New_York'},
            'DFW': {'id': 11298, 'city': 'Dallas/Fort Worth', 'state': 'TX', 'state_name': 'Texas',
                   'lat': 32.8968, 'lon': -97.0380, 'alt': 603, 'tz': 'America/Chicago'},
            'ORD': {'id': 13930, 'city': 'Chicago', 'state': 'IL', 'state_name': 'Illinois',
                   'lat': 41.9786, 'lon': -87.9048, 'alt': 668, 'tz': 'America/Chicago'},
            'LAX': {'id': 12892, 'city': 'Los Angeles', 'state': 'CA', 'state_name': 'California',
                   'lat': 33.9425, 'lon': -118.4081, 'alt': 125, 'tz': 'America/Los_Angeles'},
            'DEN': {'id': 11292, 'city': 'Denver', 'state': 'CO', 'state_name': 'Colorado',
                   'lat': 39.8617, 'lon': -104.6732, 'alt': 5431, 'tz': 'America/Denver'},
            'JFK': {'id': 12478, 'city': 'New York', 'state': 'NY', 'state_name': 'New York',
                   'lat': 40.6399, 'lon': -73.7787, 'alt': 13, 'tz': 'America/New_York'}
        }
        
        for code, name in major_airports.items():
            print(f"  {code}: {name}")
            
        all_airports = ['ABE', 'ABI', 'ABQ', 'ABR', 'ABY', 'ACK', 'ACT', 'ACV', 'ACY', 'ADK', 'ADQ', 'AEX', 'AGS', 'AKN', 'ALB', 'ALW', 'AMA', 'ANC', 'APN', 'ASE', 'ATL', 'ATW', 'AUS', 'AVL', 'AVP', 'AZA', 'AZO', 'BDL', 'BET', 'BFF', 'BFL', 'BGM', 'BGR', 'BHM', 'BIH', 'BIL', 'BIS', 'BJI', 'BLI', 'BLV', 'BMI', 'BNA', 'BOI', 'BOS', 'BPT', 'BQK', 'BQN', 'BRD', 'BRO', 'BRW', 'BTM', 'BTR', 'BTV', 'BUF', 'BUR', 'BWI', 'BZN', 'CAE', 'CAK', 'CDC', 'CDV', 'CHA', 'CHO', 'CHS', 'CID', 'CIU', 'CKB', 'CLE', 'CLL', 'CLT', 'CMH', 'CMI', 'CMX', 'COD', 'COS', 'COU', 'CPR', 'CRP', 'CRW', 'CSG', 'CVG', 'CWA', 'CYS', 'DAB', 'DAL', 'DAY', 'DCA', 'DDC', 'DEC', 'DEN', 'DFW', 'DHN', 'DIK', 'DLG', 'DLH', 'DRO', 'DSM', 'DTW', 'DVL', 'EAR', 'EAU', 'ECP', 'EGE', 'EKO', 'ELM', 'ELP', 'ESC', 'EUG', 'EVV', 'EWN', 'EWR', 'EYW', 'FAI', 'FAR', 'FAT', 'FAY', 'FCA', 'FLG', 'FLL', 'FNT', 'FOD', 'FSD', 'FSM', 'FWA', 'GCC', 'GCK', 'GEG', 'GFK', 'GGG', 'GJT', 'GNV', 'GPT', 'GRB', 'GRI', 'GRK', 'GRR', 'GSO', 'GSP', 'GST', 'GTF', 'GTR', 'GUC', 'GUM', 'HDN', 'HGR', 'HHH', 'HIB', 'HLN', 'HNL', 'HOB', 'HOU', 'HPN', 'HRL', 'HSV', 'HTS', 'HYA', 'HYS', 'IAD', 'IAG', 'IAH', 'ICT', 'IDA', 'ILM', 'IMT', 'IND', 'INL', 'ISP', 'ITH', 'ITO', 'JAC', 'JAN', 'JAX', 'JFK', 'JLN', 'JMS', 'JNU', 'JST', 'KOA', 'KTN', 'LAN', 'LAR', 'LAS', 'LAW', 'LAX', 'LBB', 'LBE', 'LBF', 'LBL', 'LCH', 'LCK', 'LEX', 'LFT', 'LGA', 'LGB', 'LIH', 'LIT', 'LNK', 'LRD', 'LSE', 'LWS', 'MAF', 'MBS', 'MCI', 'MCO', 'MCW', 'MDT', 'MDW', 'MEI', 'MEM', 'MFE', 'MFR', 'MGM', 'MGW', 'MHK', 'MHT', 'MIA', 'MKE', 'MLB', 'MLI', 'MLU', 'MOB', 'MOT', 'MQT', 'MRY', 'MSN', 'MSO', 'MSP', 'MSY', 'MTJ', 'MVY', 'MYR', 'OAJ', 'OAK', 'OGG', 'OKC', 'OMA', 'OME', 'ONT', 'ORD', 'ORF', 'ORH', 'OTH', 'OTZ', 'PAE', 'PBG', 'PBI', 'PDX', 'PGD', 'PHL', 'PHX', 'PIA', 'PIB', 'PIE', 'PIH', 'PIT', 'PLN', 'PNS', 'PPG', 'PQI', 'PRC', 'PSC', 'PSE', 'PSG', 'PSM', 'PSP', 'PVD', 'PVU', 'PWM', 'RAP', 'RDD', 'RDM', 'RDU', 'RFD', 'RHI', 'RIC', 'RIW', 'RKS', 'RNO', 'ROA', 'ROC', 'ROW', 'RST', 'RSW', 'SAF', 'SAN', 'SAT', 'SAV', 'SBA', 'SBN', 'SBP', 'SCC', 'SCE', 'SCK', 'SDF', 'SEA', 'SFB', 'SFO', 'SGF', 'SGU', 'SHR', 'SHV', 'SIT', 'SJC', 'SJT', 'SJU', 'SLC', 'SLN', 'SMF', 'SMX', 'SNA', 'SPI', 'SPN', 'SPS', 'SRQ', 'STC', 'STL', 'STS', 'STT', 'STX', 'SUN', 'SUX', 'SWF', 'SWO', 'SYR', 'TLH', 'TOL', 'TPA', 'TRI', 'TTN', 'TUL', 'TUS', 'TVC', 'TWF', 'TXK', 'TYR', 'TYS', 'USA', 'VCT', 'VLD', 'VPS', 'WRG', 'WYS', 'XNA', 'XWA', 'YAK', 'YUM']
        
        print("\nEnter any valid airport code. Type 'list' to see all airports.")
            
        while True:
            origin = input("\nOrigin airport code: ").upper()
            if origin == 'LIST':
                airports_per_row = 8
                for i in range(0, len(all_airports), airports_per_row):
                    row = all_airports[i:i+airports_per_row]
                    print("  ".join(row))
                continue
            
            if origin in all_airports:
                break
            print("Invalid airport code. Please enter a valid code or type 'list' to see all options.")
        
        if origin in airport_metadata:
            origin_metadata = airport_metadata[origin]
            origin_city = origin_metadata['city']
            origin_state = origin_metadata['state']
            origin_state_nm = origin_metadata['state_name']
            origin_airport_id = origin_metadata['id']
            origin_latitude = origin_metadata['lat']
            origin_longitude = origin_metadata['lon']
            origin_altitude = origin_metadata['alt']
            origin_timezone = origin_metadata['tz']
        else:
            print("\nOrigin airport details:")
            origin_city = input("City (e.g., Atlanta): ")
            origin_state = input("State code (e.g., GA): ").upper()
            origin_state_nm = input("State name (e.g., Georgia): ").title()
            origin_airport_id = 10000
            origin_latitude = 0
            origin_longitude = 0
            origin_altitude = 0
            origin_timezone = "America/New_York"
            
        while True:
            dest = input("\nDestination airport code: ").upper()
            if dest == 'LIST':
                airports_per_row = 8
                for i in range(0, len(all_airports), airports_per_row):
                    row = all_airports[i:i+airports_per_row]
                    print("  ".join(row))
                continue
                
            if dest in all_airports and dest != origin:
                break
            elif dest == origin:
                print("Destination cannot be the same as origin.")
            else:
                print("Invalid airport code. Please enter a valid code or type 'list' to see all options.")
        
        if dest in airport_metadata:
            dest_metadata = airport_metadata[dest]
            dest_city = dest_metadata['city']
            dest_state = dest_metadata['state']
            dest_state_nm = dest_metadata['state_name']
            dest_airport_id = dest_metadata['id']
            dest_latitude = dest_metadata['lat']
            dest_longitude = dest_metadata['lon']
            dest_altitude = dest_metadata['alt']
            dest_timezone = dest_metadata['tz']
        else:
            print("\nDestination airport details:")
            dest_city = input("City (e.g., Chicago): ")
            dest_state = input("State code (e.g., IL): ").upper()
            dest_state_nm = input("State name (e.g., Illinois): ").title()
            dest_airport_id = 20000
            dest_latitude = 0
            dest_longitude = 0
            dest_altitude = 0
            dest_timezone = "America/New_York"
        
        while True:
            dep_time = input("\nActual departure time (HHMM, 24-hour format, e.g. 1430 for 2:30 PM): ")
            if len(dep_time) == 4 and dep_time.isdigit():
                hour = int(dep_time[:2])
                minute = int(dep_time[2:])
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    break
            print("Invalid time format. Please use HHMM in 24-hour format.")
        
        while True:
            crs_dep_time = input("Scheduled departure time (HHMM, 24-hour format): ")
            if len(crs_dep_time) == 4 and crs_dep_time.isdigit():
                hour = int(crs_dep_time[:2])
                minute = int(crs_dep_time[2:])
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    break
            print("Invalid time format. Please use HHMM in 24-hour format.")
        
        while True:
            crs_arr_time = input("Scheduled arrival time (HHMM, 24-hour format): ")
            if len(crs_arr_time) == 4 and crs_arr_time.isdigit():
                hour = int(crs_arr_time[:2])
                minute = int(crs_arr_time[2:])
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    break
            print("Invalid time format. Please use HHMM in 24-hour format.")
        
        while True:
            try:
                distance = float(input("\nFlight distance (miles): "))
                if distance > 0:
                    break
                print("Distance must be greater than 0.")
            except ValueError:
                print("Please enter a valid number.")
        
        distance_group = 1
        if distance < 250:
            distance_group = 1
        elif distance < 500:
            distance_group = 2
        elif distance < 750:
            distance_group = 3
        elif distance < 1000:
            distance_group = 4
        elif distance < 1250:
            distance_group = 5
        elif distance < 1500:
            distance_group = 6
        elif distance < 1750:
            distance_group = 7
        elif distance < 2000:
            distance_group = 8
        elif distance < 2250:
            distance_group = 9
        else:
            distance_group = 10
            
        print("\nWeather conditions at origin airport:")
        weather_options = [
            "Clear", "Partly Cloudy", "Cloudy", "Light Rain", "Rain", 
            "Thunderstorms", "Snow", "Fog", "Wind"
        ]
        for i, option in enumerate(weather_options, 1):
            print(f"  {i}. {option}")
            
        while True:
            try:
                weather_idx = int(input("Select weather condition (1-9): "))
                if 1 <= weather_idx <= len(weather_options):
                    origin_conditions = weather_options[weather_idx-1]
                    break
                print(f"Please enter a number between 1 and {len(weather_options)}.")
            except ValueError:
                print("Please enter a valid number.")
                
        while True:
            try:
                origin_severity = int(input("Weather severity (0=mild, 10=severe): "))
                if 0 <= origin_severity <= 10:
                    break
                print("Please enter a number between 0 and 10.")
            except ValueError:
                print("Please enter a valid number.")
                
        print("\nWeather conditions at destination airport:")
        for i, option in enumerate(weather_options, 1):
            print(f"  {i}. {option}")
            
        while True:
            try:
                weather_idx = int(input("Select weather condition (1-9): "))
                if 1 <= weather_idx <= len(weather_options):
                    dest_conditions = weather_options[weather_idx-1]
                    break
                print(f"Please enter a number between 1 and {len(weather_options)}.")
            except ValueError:
                print("Please enter a valid number.")
                
        while True:
            try:
                dest_severity = int(input("Weather severity (0=mild, 10=severe): "))
                if 0 <= dest_severity <= 10:
                    break
                print("Please enter a number between 0 and 10.")
            except ValueError:
                print("Please enter a valid number.")
                
        holidays = ["None", "New Year's Day", "MLK Day", "Presidents Day", 
                    "Easter", "Memorial Day", "Independence Day", "Labor Day", 
                    "Columbus Day", "Veterans Day", "Thanksgiving", "Christmas"]
        
        print("\nIs this flight during a holiday period?")
        for i, holiday in enumerate(holidays):
            print(f"  {i}. {holiday}")
            
        while True:
            try:
                holiday_idx = int(input("Select holiday (0 for none): "))
                if 0 <= holiday_idx < len(holidays):
                    is_holiday = 1 if holiday_idx > 0 else 0
                    holiday_name = holidays[holiday_idx] if holiday_idx > 0 else ""
                    break
                print(f"Please enter a number between 0 and {len(holidays)-1}.")
            except ValueError:
                print("Please enter a valid number.")
                
        holiday_travel = 0
        if is_holiday:
            holiday_travel = 1 if input("Is this during peak holiday travel (y/n)? ").lower() == 'y' else 0
            
        month = flight_date.month
        if 3 <= month <= 5:
            season = "Spring"
        elif 6 <= month <= 8:
            season = "Summer"
        elif 9 <= month <= 11:
            season = "Fall"
        else:
            season = "Winter"
            
        user_inputs = {
            'YEAR': flight_date.year,
            'MONTH': flight_date.month,
            'FL_DATE': date_input,
            'OP_UNIQUE_CARRIER': carrier,
            'OP_CARRIER': airlines[carrier],
            'OP_CARRIER_FL_NUM': flight_num,
            'ORIGIN_AIRPORT_ID': origin_airport_id,
            'ORIGIN': origin,
            'ORIGIN_CITY_NAME': origin_city,
            'ORIGIN_STATE_ABR': origin_state,
            'ORIGIN_STATE_NM': origin_state_nm,
            'DEST_AIRPORT_ID': dest_airport_id,
            'DEST': dest,
            'DEST_CITY_NAME': dest_city,
            'DEST_STATE_ABR': dest_state,
            'DEST_STATE_NM': dest_state_nm,
            'DEP_TIME': int(dep_time),
            'CRS_DEP_TIME': int(crs_dep_time),
            'CRS_ARR_TIME': int(crs_arr_time),
            'FLIGHTS': 1,
            'DISTANCE': distance,
            'DISTANCE_GROUP': distance_group,
            'SOURCE_FILE': 'User Input',
            
            'ORIGIN_LATITUDE_x': origin_latitude,
            'ORIGIN_LONGITUDE_x': origin_longitude,
            'ORIGIN_ALTITUDE': origin_altitude,
            'ORIGIN_TIMEZONE': origin_timezone,
            'DEST_LATITUDE_x': dest_latitude,
            'DEST_LONGITUDE_x': dest_longitude,
            'DEST_ALTITUDE': dest_altitude,
            'DEST_TIMEZONE': dest_timezone,
            
            'ORIGIN_CONDITIONS': origin_conditions,
            'ORIGIN_WEATHER_SEVERITY': origin_severity,
            'DEST_CONDITIONS': dest_conditions,
            'DEST_WEATHER_SEVERITY': dest_severity,
            'MAX_WEATHER_SEVERITY': max(origin_severity, dest_severity),
            'ORIGIN_EXTREME_WEATHER': 1 if origin_severity >= 7 else 0,
            'DEST_EXTREME_WEATHER': 1 if dest_severity >= 7 else 0,
            'WEATHER_IMPACT_SCORE': (origin_severity + dest_severity) / 2,
            
            'IS_HOLIDAY': is_holiday,
            'HOLIDAY_NAME': holiday_name,
            'HOLIDAY_TRAVEL_PERIOD': holiday_travel,
            
            'DAY_OF_MONTH': flight_date.day,
            'DAY_OF_WEEK': flight_date.weekday() + 1,
            'IS_WEEKEND': 1 if flight_date.weekday() >= 5 else 0,
            'WEEK_OF_YEAR': flight_date.isocalendar()[1],
            'SEASON': season
        }
        
        return user_inputs
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)

def preprocess_inputs(user_inputs):
    df = pd.DataFrame([user_inputs])
    
    if 'DEP_HOUR' not in df.columns:
        df['DEP_HOUR'] = df['DEP_TIME'] // 100
    
    if 'DEP_HOUR_SIN' not in df.columns:
        df['DEP_HOUR_SIN'] = np.sin(2 * np.pi * df['DEP_HOUR'] / 24)
        df['DEP_HOUR_COS'] = np.cos(2 * np.pi * df['DEP_HOUR'] / 24)
    
    if 'MONTH_SIN' not in df.columns:
        df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
        df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    
    if 'DAY_OF_WEEK_SIN' not in df.columns:
        df['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
        df['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)
    
    if 'OP_UNIQUE_CARRIER_' + df['OP_UNIQUE_CARRIER'].iloc[0] not in df.columns:
        carriers = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9', 'HA', 'G4', '9E', 'OH', 'YX', 'MQ', 'OO']
        for carrier in carriers:
            col_name = 'OP_UNIQUE_CARRIER_' + carrier
            df[col_name] = 0
        
        carrier = df['OP_UNIQUE_CARRIER'].iloc[0]
        if 'OP_UNIQUE_CARRIER_' + carrier in df.columns:
            df['OP_UNIQUE_CARRIER_' + carrier] = 1
    
    if 'SEVERITY_DISTANCE_EFFECT' not in df.columns:
        df['SEVERITY_DISTANCE_EFFECT'] = df.apply(
            lambda row: (row['MAX_WEATHER_SEVERITY'] * 2) / np.log10(max(row['DISTANCE'], 100)), 
            axis=1
        )
    
    additional_columns = [
        'ORIGIN_CLUSTER_ID', 'DEST_CLUSTER_ID',
        'ORIGIN_TEMP_AVG', 'DEST_TEMP_AVG',
        'ORIGIN_PRECIPITATION', 'DEST_PRECIPITATION',
        'ORIGIN_WIND_SPEED', 'DEST_WIND_SPEED',
        'ORIGIN_CLOUD_COVER', 'DEST_CLOUD_COVER',
        'ORIGIN_VISIBILITY', 'DEST_VISIBILITY',
        'ORIGIN_WEATHER_ICON', 'DEST_WEATHER_ICON'
    ]
    
    for col in additional_columns:
        if col not in df.columns:
            df[col] = 0
    
    weather_to_icon = {
        'Clear': 'clear-day',
        'Partly Cloudy': 'partly-cloudy-day',
        'Cloudy': 'cloudy',
        'Light Rain': 'rain',
        'Rain': 'rain',
        'Thunderstorms': 'thunderstorm',
        'Snow': 'snow',
        'Fog': 'fog',
        'Wind': 'wind'
    }
    
    if df['ORIGIN_WEATHER_ICON'].iloc[0] == 0:
        origin_cond = df['ORIGIN_CONDITIONS'].iloc[0]
        dest_cond = df['DEST_CONDITIONS'].iloc[0]
        
        df['ORIGIN_WEATHER_ICON'] = weather_to_icon.get(origin_cond, 'unknown')
        df['DEST_WEATHER_ICON'] = weather_to_icon.get(dest_cond, 'unknown')
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col.endswith('_WEATHER_ICON') or col.endswith('_CONDITIONS'):
            unique_values = df[col].unique()
            mapping = {val: i for i, val in enumerate(unique_values)}
            df[col] = df[col].map(mapping)
        else:
            df[col] = df[col].astype('category').cat.codes
    
    df = df.fillna(0)
    
    drop_cols = ['OP_CARRIER', 'ORIGIN_CONDITIONS', 'DEST_CONDITIONS', 'HOLIDAY_NAME']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    return df

def predict_delay(df, model, scaler):
    try:
        df_copy = df.copy()
        
        scaler_columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
        
        if scaler_columns is not None:
            empty_df = pd.DataFrame(0, index=df_copy.index, columns=scaler_columns)
            
            for col in df_copy.columns:
                if col in scaler_columns:
                    empty_df[col] = df_copy[col]
                    
            df_to_scale = empty_df
        else:
            df_to_scale = df_copy
        
        df_scaled = pd.DataFrame(scaler.transform(df_to_scale), columns=df_to_scale.columns)
        
        dmatrix = xgb.DMatrix(df_scaled)
        
        prediction_prob = model.predict(dmatrix)[0]
        
        optimal_threshold = 0.49
        prediction = 1 if prediction_prob >= optimal_threshold else 0
        
        return prediction, prediction_prob, optimal_threshold
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns}")
        if hasattr(scaler, 'feature_names_in_'):
            print(f"Scaler expected columns: {scaler.feature_names_in_}")
        
        return None, None, None

def format_time_display(time_str):
    if not time_str or len(time_str) < 2:
        return "Unknown"
    
    try:
        time_str = time_str.zfill(4) if len(time_str) < 4 else time_str
        
        hour = int(time_str[:2])
        minute = time_str[2:]
        am_pm = "AM" if hour < 12 else "PM"
        hour = hour if hour <= 12 else hour - 12
        hour = 12 if hour == 0 else hour
        return f"{hour}:{minute} {am_pm}"
    except (ValueError, IndexError):
        return "Unknown"

def display_prediction(inputs, prediction, probability, threshold):
    print("\n" + "="*50)
    print(" FLIGHT DELAY PREDICTION RESULTS ")
    print("="*50)
    
    if prediction == 1:
        print("\n🚫 PREDICTION: Your flight is likely to be DELAYED")
    else:
        print("\n✓ PREDICTION: Your flight is likely to be ON TIME")
    
    print(f"\nProbability of delay: {probability*100:.2f}%")
    print(f"Decision threshold: {threshold*100:.1f}% (predictions above this are considered delays)")
    
    confidence = probability if prediction == 1 else 1 - probability
    confidence_text = "High" if confidence > 0.75 else "Medium" if confidence > 0.6 else "Low"
    print(f"Confidence level: {confidence_text} ({confidence*100:.1f}%)")
    
    print("\n"+"-"*50)
    print(" FLIGHT DETAILS")
    print("-"*50)
    
    try:
        flight_date = datetime.strptime(inputs['FL_DATE'], '%Y-%m-%d')
        formatted_date = flight_date.strftime('%A, %B %d, %Y')
    except (KeyError, ValueError, TypeError):
        formatted_date = "Date not available"
    
    print(f"Date: {formatted_date}")
    print(f"Airline: {inputs['OP_UNIQUE_CARRIER']}")
    print(f"Route: {inputs['ORIGIN']} → {inputs['DEST']}")
    print(f"Distance: {inputs['DISTANCE']} miles")
    
    try:
        dep_time = format_time_display(str(inputs['DEP_TIME']))
        print(f"Departure Time: {dep_time}")
    except (KeyError, ValueError, TypeError):
        print("Departure Time: Not available")
        
    try:
        sched_dep = format_time_display(str(inputs['CRS_DEP_TIME']))
        print(f"Scheduled Departure: {sched_dep}")
    except (KeyError, ValueError, TypeError):
        print("Scheduled Departure: Not available")
        
    try:
        sched_arr = format_time_display(str(inputs['CRS_ARR_TIME']))
        print(f"Scheduled Arrival: {sched_arr}")
    except (KeyError, ValueError, TypeError):
        print("Scheduled Arrival: Not available")
    
    print("\n"+"-"*50)
    print(" WEATHER CONDITIONS")
    print("-"*50)
    
    if 'ORIGIN_CONDITIONS' in inputs and 'DEST_CONDITIONS' in inputs:
        origin_weather = inputs.get('ORIGIN_CONDITIONS', 'Unknown')
        dest_weather = inputs.get('DEST_CONDITIONS', 'Unknown')
        origin_severity = inputs.get('ORIGIN_WEATHER_SEVERITY', 0)
        dest_severity = inputs.get('DEST_WEATHER_SEVERITY', 0)
        
        print(f"Origin Weather: {origin_weather} (Severity: {origin_severity}/10)")
        print(f"Destination Weather: {dest_weather} (Severity: {dest_severity}/10)")
        
        if origin_severity >= 7 or dest_severity >= 7:
            print("\n⚠️ Severe weather detected at one or both airports.")
    
    if inputs.get('IS_HOLIDAY', 0) == 1:
        print("\n"+"-"*50)
        print(" HOLIDAY INFORMATION")
        print("-"*50)
        holiday_name = inputs.get('HOLIDAY_NAME', 'Holiday')
        holiday_travel = "Yes" if inputs.get('HOLIDAY_TRAVEL_PERIOD', 0) == 1 else "No"
        print(f"Holiday: {holiday_name}")
        print(f"Peak Holiday Travel Period: {holiday_travel}")
    
    print("\n"+"-"*50)
    print(" DELAY RISK FACTORS")
    print("-"*50)
    
    risk_factors = []
    
    try:
        hour = int(str(inputs['DEP_TIME'])[:2])
        if 6 <= hour <= 9:
            risk_factors.append("Morning rush hour flight (higher delay risk)")
        elif 16 <= hour <= 19:
            risk_factors.append("Evening rush hour flight (higher delay risk)")
        elif hour >= 23 or hour <= 5:
            risk_factors.append("Red-eye flight (often less congested)")
        else:
            risk_factors.append("Mid-day flight (moderate delay risk)")
    except (KeyError, ValueError, TypeError, IndexError):
        pass
    
    try:
        max_severity = inputs.get('MAX_WEATHER_SEVERITY', 0)
        if max_severity >= 7:
            risk_factors.append(f"Severe weather (severity: {max_severity}/10)")
        elif max_severity >= 4:
            risk_factors.append(f"Moderate weather concerns (severity: {max_severity}/10)")
    except (KeyError, ValueError, TypeError):
        pass
    
    try:
        month = datetime.strptime(inputs['FL_DATE'], '%Y-%m-%d').month
        if month in [11, 12]:
            risk_factors.append("Holiday season (higher delay risk)")
        elif month in [6, 7, 8]:
            risk_factors.append("Summer travel season (higher delay risk)")
        elif month in [3, 4]:
            risk_factors.append("Spring break period (moderate delay risk)")
    except (KeyError, ValueError, TypeError):
        pass
    
    if inputs.get('IS_HOLIDAY', 0) == 1 and inputs.get('HOLIDAY_TRAVEL_PERIOD', 0) == 1:
        risk_factors.append("Peak holiday travel period (higher delay risk)")
    
    try:
        day_of_week = inputs.get('DAY_OF_WEEK', 0)
        if day_of_week in [5, 7]:
            risk_factors.append("Weekend travel day (higher delay risk)")
    except (KeyError, ValueError, TypeError):
        pass
    
    try:
        distance = float(inputs['DISTANCE'])
        if distance < 300:
            risk_factors.append("Short flight (may have higher variability)")
        elif distance > 2000:
            risk_factors.append("Long-haul flight (exposure to more airspace)")
    except (KeyError, ValueError, TypeError):
        pass
    
    if risk_factors:
        for factor in risk_factors:
            print(f"- {factor}")
    else:
        print("- No specific risk factors identified")
    
    print("\n"+"-"*50)
    print(" MODEL INFORMATION ")
    print("-"*50)
    print("Model type: XGBoost Classifier")
    print("Model accuracy: ~70.4%")
    print("Note: This prediction is based on historical patterns")
    print("      and may not account for all current factors.")
    print("="*50)

def main():
    try:
        model, scaler = load_model()
        
        user_inputs = get_user_inputs()
        
        df = preprocess_inputs(user_inputs)
        
        prediction, probability, threshold = predict_delay(df, model, scaler)
        
        if prediction is not None and probability is not None and threshold is not None:
            display_prediction(user_inputs, prediction, probability, threshold)
        else:
            print("\nCould not make prediction due to errors. Please try again with different inputs.")
        
        while True:
            again = input("\nMake another prediction? (y/n): ").lower()
            if again in ['y', 'yes']:
                user_inputs = get_user_inputs()
                df = preprocess_inputs(user_inputs)
                prediction, probability, threshold = predict_delay(df, model, scaler)
                if prediction is not None and probability is not None and threshold is not None:
                    display_prediction(user_inputs, prediction, probability, threshold)
                else:
                    print("\nCould not make prediction due to errors. Please try again with different inputs.")
            elif again in ['n', 'no']:
                print("\nThank you for using the Flight Delay Predictor!")
                break
            else:
                print("Please enter 'y' or 'n'.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()