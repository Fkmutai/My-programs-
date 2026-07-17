import kagglehub
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
except ImportError:
    # Fallback if xgboost is not installed or import cannot be resolved
    print("xgboost not available, falling back to sklearn GradientBoostingRegressor")
    XGBRegressor = GradientBoostingRegressor

# Download dataset
path = kagglehub.dataset_download("shuvojitdas/global-temperature-dataset")

print("Dataset downloaded to:", path)

# Replace CSV filename with actual dataset file - adjust filename as needed
df = pd.read_csv(f"{path}/GlobalTemperatures.csv")

# Rename dt to Date and ensure date format (mixed formats in data)
df = df.rename(columns={"dt": "Date"})
df["Date"] = pd.to_datetime(df["Date"], format="mixed")

# Sort data by date
df = df.sort_values("Date").reset_index(drop=True)

# Use LandAndOceanAverageTemperature as target, drop rows with NaN
df = df.dropna(subset=["LandAndOceanAverageTemperature"])

# Create time-series features
df["day_of_year"] = df["Date"].dt.dayofyear
df["month"] = df["Date"].dt.month
df["year"] = df["Date"].dt.year

# Create lagged features (temperature from previous months)
df["temp_1mo_ago"] = df["LandAndOceanAverageTemperature"].shift(1)
df["temp_12mo_ago"] = df["LandAndOceanAverageTemperature"].shift(12)

# Drop rows with NaN values created by shift
df = df.dropna()

features = [
    "day_of_year",
    "month",
    "year",
    "temp_1mo_ago",
    "temp_12mo_ago"
]

X = df[features]
y = df["LandAndOceanAverageTemperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print(f"MAE: {mae:.2f} °C")

joblib.dump(model, "temperature_model.pkl")

print("Model saved.")
