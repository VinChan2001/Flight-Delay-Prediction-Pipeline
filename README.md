# ✈️ Flight Delay Prediction using XGBoost and Real-Time CLI

This project builds a real-time flight delay prediction system using machine learning, combining extensive flight performance data from the U.S. Bureau of Transportation Statistics (BTS), enriched with weather and holiday metadata. The final model is deployed as a command-line interface (CLI) tool that predicts the likelihood of delay for a given flight using historical patterns.

## 📦 Dataset Overview

- **Source**: U.S. BTS On-Time Performance (July–December 2024)  
- **Records Processed**: ~3.6 million  
- **External Sources**:  
  - [OpenFlights](https://github.com/jpatokal/openflights) – airport geolocation  
  - Visual Crossing Weather API – origin/destination weather  
  - U.S. Federal Holiday data  

### ⚙️ Preprocessing Highlights

- Combined monthly BTS files into a single DataFrame  
- Cleaned fields like `FL_DATE`, `CARRIER`, `ORIGIN`, `DEST`, etc.  
- Parsed times, removed nulls/duplicates  
- Added airport geolocation for weather API querying  

---

## 🔍 Exploratory Data Analysis (EDA)

- **Delay Definition**: Arrival delay > 15 minutes  
- **Overall Delay Rate**: 19.65%  
- **Monthly Delay Rates**:
  - July: 29.52%  
  - October: 13.61%  
  - December: 21.65%  
- **Airlines**:
  - WN (Southwest) had most flights: 706,659  
  - NK (Spirit) had the highest delay rate  
- **Insights**:
  - Delays more frequent during holidays and peak hours  
  - Peak travel and bad weather significantly raise delay risk  

---

## 🛠 Feature Engineering

- **Time-based**: Hour of day, day of week, red-eye/rush hour  
- **Cyclical encoding**: Sine/cosine for hour, month  
- **Weather**: Categorical label (rain, fog, etc.) + severity score (0–10)  
- **Holiday flags**: Specific federal holidays + peak travel periods  
- **Airline & airport codes**: One-hot encoded  

---

## 🤖 Modeling

- **Goal**: Classify flights as Delayed (1) or On-Time (0)  
- **Algorithm**: XGBoost Classifier  
- **Tuning**: GridSearchCV  
- **Validation**: Stratified 3-fold cross-validation  

### ✅ Final Model Performance

| Metric               | Score    |
|----------------------|----------|
| Cross-Val Accuracy   | 70.40%   |
| Final Test Accuracy  | 70.44%   |
| Final Test ROC AUC   | 0.7622   |
| Decision Threshold   | 49.0%    |

---

## 💻 Deployment: CLI Tool

Users can input flight details through a command-line interface. The tool loads the trained model and outputs:

- **Prediction**: Delayed / On Time  
- **Probability of Delay**  
- **Confidence Level**  
- **Delay Risk Factors** (weather, holiday, timing)  

### ✈️ Sample CLI Input

```bash
python predictor.py
```
Prompts for:
	•	Date, airline code, flight number
	•	Origin & destination airports
	•	Actual & scheduled departure/arrival times
	•	Flight distance
	•	Weather conditions & severity
	•	Holiday indicator

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



⸻

🧠 Key Learnings
	•	Integrated large-scale government and third-party data sources
	•	Engineered time-aware and weather-sensitive features
	•	Validated a tuned XGBoost model for flight delay classification
	•	Built a fully functional CLI tool to simulate real-world user input

⸻

🛠 Skills & Tools Used
	•	Python, Pandas, NumPy
	•	Scikit-learn, XGBoost, GridSearchCV
	•	Matplotlib, Seaborn
	•	Visual Crossing Weather API
	•	Command-Line Interface (argparse)
	•	Jupyter Notebook, JSON, Pickle
	•	GitHub for version control

