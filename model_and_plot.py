import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Load Data
DATA_FILE = 'restaurant_data.csv'
df = pd.read_csv(DATA_FILE)
df['Date'] = pd.to_datetime(df['Date'])

# 1. Aggregate Data: Daily Order Counts
daily_orders = df.groupby('Date')['Order_ID'].nunique().reset_index()
daily_orders.columns = ['Date', 'Order_Count']
daily_orders = daily_orders.set_index('Date')
daily_orders = daily_orders.asfreq('D').fillna(0) # Ensure continuous time series

# 2. Aggregate Data: Ingredient Usage (Example: Pizza Dough)
ingredient_usage = df.groupby(['Date', 'Ingredient_Name'])['Quantity_Used'].sum().reset_index()
pizza_dough_usage = ingredient_usage[ingredient_usage['Ingredient_Name'] == 'Pizza Dough'].set_index('Date')['Quantity_Used']
pizza_dough_usage = pizza_dough_usage.asfreq('D').fillna(0)

# --- Visualizations ---

# Plot 1: Daily Order Count History
plt.figure(figsize=(12, 6))
plt.plot(daily_orders.index, daily_orders['Order_Count'], label='Daily Orders', color='blue')
plt.title('Daily Restaurant Orders (1 Year)')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.legend()
plt.grid(True)
plt.savefig('daily_orders_history.png')
print("Saved daily_orders_history.png")

# Plot 2: Stock Levels (Example: Pizza Dough)
# We need to take the last stock value of the day for a proper stock level plot
daily_stock = df.sort_values(['Date', 'Order_ID']).groupby(['Date', 'Ingredient_Name'])['Stock_Available'].last().unstack()
if 'Pizza Dough' in daily_stock.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(daily_stock.index, daily_stock['Pizza Dough'], label='Pizza Dough Stock', color='green')
    plt.title('Pizza Dough Stock Level Over Time')
    plt.xlabel('Date')
    plt.ylabel('Stock (balls)')
    plt.legend()
    plt.grid(True)
    plt.savefig('stock_level_pizza_dough.png')
    print("Saved stock_level_pizza_dough.png")

# --- Time Series Modeling (ARIMA) ---

# Split into Train and Test
train_size = int(len(daily_orders) * 0.9)
train, test = daily_orders.iloc[:train_size], daily_orders.iloc[train_size:]

print(f"Training samples: {len(train)}, Testing samples: {len(test)}")

# Fit ARIMA Model
# Using order=(5,1,0) as a starting point. simpler models are often better for random data.
# (p,d,q): p=AR order, d=Differencing, q=MA order
model = ARIMA(train['Order_Count'], order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast_result = model_fit.forecast(steps=len(test))
forecast = forecast_result # pandas Series with index matching test

# Calculate Error
rmse = np.sqrt(mean_squared_error(test['Order_Count'], forecast))
print(f"Test RMSE: {rmse:.3f}")

# Plot 3: Forecast vs Actual
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Order_Count'], label='Training Data')
plt.plot(test.index, test['Order_Count'], label='Actual Valid Data', color='green')
plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='--')
plt.title(f'Order Forecast using ARIMA (RMSE: {rmse:.2f})')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.legend()
plt.grid(True)
plt.savefig('forecast_vs_actual.png')
print("Saved forecast_vs_actual.png")

# Future Forecast (Next 30 Days)
future_model = ARIMA(daily_orders['Order_Count'], order=(5,1,0)).fit()
future_forecast = future_model.forecast(steps=30)

# Plot 4: Future Forecast
plt.figure(figsize=(12, 6))
plt.plot(daily_orders.index, daily_orders['Order_Count'], label='Historical Data')
plt.plot(future_forecast.index, future_forecast, label='30-Day Forecast', color='orange', linestyle='--')
plt.title('Future Order Forecast (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.legend()
plt.grid(True)
plt.savefig('future_forecast.png')
print("Saved future_forecast.png")
