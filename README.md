# ✈️ Flight Delay Prediction using XGBoost and Real-Time CLI

This project builds a real-time flight delay prediction system using machine learning. It combines extensive flight performance data from the U.S. Bureau of Transportation Statistics (BTS), weather API data, and holiday indicators to train a robust XGBoost model. The model is deployed as a command-line interface (CLI) tool that outputs the likelihood of a delay for any given flight using historical patterns.

---

## 📦 Dataset Overview

- **Source**: U.S. BTS On-Time Performance (July–December 2024)  
- **Records Processed**: ~3.6 million rows  
- **External Sources**:
  - [OpenFlights](https://github.com/jpatokal/openflights) – Airport geolocation
  - [Visual Crossing Weather API](https://www.visualcrossing.com/weather-data-editions) – Weather data at origin & destination
  - U.S. Federal Holiday calendar – Holiday & peak travel labels

---

## ⚙️ Preprocessing Highlights

- Merged monthly BTS data into a single DataFrame  
- Cleaned essential fields (`FL_DATE`, `CARRIER`, `ORIGIN`, `DEST`, etc.)  
- Parsed and standardized time fields  
- Removed nulls and duplicate records  
- Integrated airport coordinates for weather API lookup  

---

## 🔍 Exploratory Data Analysis (EDA)

- **Delay Definition**: Arrival delay > 15 minutes  
- **Overall Delay Rate**: `19.65%`

### 📊 Monthly Delay Rates

| Month    | Delay Rate |
|----------|------------|
| July     | 29.52%     |
| October  | 13.61%     |
| December | 21.65%     |

### 🛫 Airlines

- **Most Flights**: WN (Southwest Airlines) – `706,659`  
- **Highest Delay Rate**: NK (Spirit Airlines)

### 🧠 Key EDA Insights

- Delay rates spike around major holidays (e.g., Thanksgiving, Christmas)  
- Morning and evening flights during peak periods are more delay-prone  
- Poor weather conditions at either end increase risk significantly  

---

## 🛠 Feature Engineering

- **Time Features**: Hour of day, day of week, red-eye/rush-hour flags  
- **Cyclical Encoding**: Sine and cosine transforms for hour & month  
- **Weather Encoding**:
  - Weather type (e.g., fog, rain, clear)  
  - Severity scale (0–10)  
- **Holiday Features**: Binary flags for federal holidays & peak travel periods  
- **Airline & Airport Codes**: One-hot encoding  

---

## 🤖 Modeling

- **Target Variable**: `Delayed (1)` or `On Time (0)`  
- **Model**: XGBoost Classifier  
- **Tuning**: Manual tuning + `GridSearchCV`  
- **Validation**: Stratified 3-fold cross-validation  

### ✅ Final Model Performance

| Metric               | Score     |
|----------------------|-----------|
| Cross-Validation     | 70.40%    |
| Final Test Accuracy  | 70.44%    |
| ROC AUC (Test Set)   | 0.7622    |
| Decision Threshold   | 49.0%     |

---

## 💻 Deployment: Command-Line Interface (CLI)

A real-time CLI tool built in Python (`predictor.py`) allows users to input flight details and receive:

- ✅ Delay prediction (`Delayed` or `On Time`)  
- 📊 Probability of delay  
- 🔒 Confidence level  
- ⚠️ Delay risk factors (weather, holiday, timing)

---

## 🖥️ Sample CLI Execution

```bash
python predictor.py
```
🔧 User Prompts:

	•	Date, airline code, flight number
	•	Origin & destination airports
	•	Actual & scheduled departure/arrival times
	•	Flight distance
	•	Weather conditions (origin & destination)
	•	Holiday indicator

🧪 Sample CLI Output

```bash
==================================================
 FLIGHT DELAY PREDICTION RESULTS 
==================================================

🚫 PREDICTION: Your flight is likely to be DELAYED

Probability of delay: 89.05%
Decision threshold: 49.0%
Confidence level: High (89.1%)

--------------------------------------------------
 FLIGHT DETAILS
--------------------------------------------------
Date: Wednesday, January 01, 2025
Airline: AA
Route: DFW → STL
Distance: 550.0 miles
Departure Time: 10:48 PM
Scheduled Departure: 9:19 PM
Scheduled Arrival: 11:01 PM

--------------------------------------------------
 WEATHER CONDITIONS
--------------------------------------------------
Origin Weather: Partly Cloudy (Severity: 0/10)
Destination Weather: Partly Cloudy (Severity: 2/10)

--------------------------------------------------
 HOLIDAY INFORMATION
--------------------------------------------------
Holiday: New Year's Day
Peak Holiday Travel Period: Yes

--------------------------------------------------
 DELAY RISK FACTORS
--------------------------------------------------
- Mid-day flight (moderate delay risk)
- Peak holiday travel period (higher delay risk)

--------------------------------------------------
 MODEL INFORMATION 
--------------------------------------------------
Model type: XGBoost Classifier
Model accuracy: ~70.4%
Note: This prediction is based on historical patterns
      and may not account for all current factors.
==================================================
```

---

## 📂 File Structure
```bash
.
├── BigData_Final.ipynb              # Full pipeline: EDA, modeling, results
├── EDA_flights.ipynb                # In-depth exploratory analysis
├── predictor.py                     # CLI prediction script
├── geocoded_data.ipynb              # Geolocation integration
├── annotated-BigData_Final_Report.pdf
├── annotated-BigData_Final_Appendix.pdf
```

---

##🧠 Key Learnings

	•	Integrated large-scale government & third-party datasets
	•	Engineered meaningful time-aware and weather-sensitive features
	•	Tuned and validated a robust XGBoost model
	•	Deployed an interactive command-line tool to simulate real-world flight scenarios

 ---

 ##🛠 Skills & Tools Used
 
	•	Languages & Libraries: Python, Pandas, NumPy
	•	Modeling: Scikit-learn, XGBoost, GridSearchCV
	•	Visualization: Matplotlib, Seaborn
	•	APIs: Visual Crossing Weather API
	•	Deployment: Command-Line Interface using argparse
	•	Data Formats: Jupyter Notebook, JSON, Pickle
	•	Version Control: GitHub

---
