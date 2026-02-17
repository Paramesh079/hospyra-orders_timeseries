import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import matplotlib.dates as mdates
import numpy as np

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

class RestaurantDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Restaurant Analytics Dashboard")
        self.root.geometry("1200x900")

        # Configuration
        self.colors = {'hist': '#1f77b4', 'forecast': '#ff7f0e', 'bg': '#f0f0f0'}
        
        # Load data
        try:
            self.df = pd.read_csv('restaurant_data.csv')
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        except FileNotFoundError:
            messagebox.showerror("Error", "restaurant_data.csv not found! Please run generate_restaurant_data.py first.")
            self.root.destroy()
            return

        self.current_month = None
        self.current_series = None
        self.current_forecast = None
        
        # Setup UI
        self.setup_ui()
        
        # Initial Plot
        self.update_plot()

    def setup_ui(self):
        # Main Navigation / Control Panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # 1. View Selection (Dish Filter)
        filter_group = ttk.LabelFrame(control_frame, text="Filters", padding="5")
        filter_group.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(filter_group, text="Dish:").pack(side=tk.LEFT)
        self.dishes = ["All Orders"] + sorted(self.df['Dish_Name'].unique().tolist())
        self.view_var = tk.StringVar(value=self.dishes[0])
        self.dropdown = ttk.Combobox(filter_group, textvariable=self.view_var, values=self.dishes, state="readonly", width=20)
        self.dropdown.pack(side=tk.LEFT, padx=5)
        self.dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot(self.current_month))

        # 2. Aggregation Selection
        agg_group = ttk.LabelFrame(control_frame, text="Aggregation Level", padding="5")
        agg_group.pack(side=tk.LEFT, padx=5)
        
        self.agg_var = tk.StringVar(value="Daily")
        self.agg_options = ["Daily", "Weekly", "Monthly"]
        self.agg_dropdown = ttk.Combobox(agg_group, textvariable=self.agg_var, values=self.agg_options, state="readonly", width=10)
        self.agg_dropdown.pack(side=tk.LEFT, padx=5)
        self.agg_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot(self.current_month))

        # 3. Forecast Options
        forecast_group = ttk.LabelFrame(control_frame, text="Predictions", padding="5")
        forecast_group.pack(side=tk.LEFT, padx=5)
        
        self.forecast_var = tk.BooleanVar(value=False)
        self.chk_forecast = ttk.Checkbutton(forecast_group, text="Show Forecast", 
                                          variable=self.forecast_var, command=lambda: self.update_plot(self.current_month))
        self.chk_forecast.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(forecast_group, text="Model:").pack(side=tk.LEFT, padx=(5, 0))
        self.model_var = tk.StringVar(value="ARIMA (5,1,0)")
        self.model_options = ["ARIMA (5,1,0)", "SARIMA (Weekly)"]
        self.model_dropdown = ttk.Combobox(forecast_group, textvariable=self.model_var, 
                                         values=self.model_options, state="readonly", width=15)
        self.model_dropdown.pack(side=tk.LEFT, padx=5)
        self.model_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_plot(self.current_month))

        if not HAS_STATSMODELS:
            self.chk_forecast.config(state="disabled")
            self.model_dropdown.config(state="disabled")
            ttk.Label(forecast_group, text="(Install statsmodels)", foreground="red").pack(side=tk.LEFT)

        # 4. Global Controls
        btn_group = ttk.Frame(control_frame, padding="5")
        btn_group.pack(side=tk.RIGHT)
        
        self.btn_reset = ttk.Button(btn_group, text="Reset View", command=self.reset_view)
        self.btn_reset.pack(side=tk.RIGHT, padx=5)

        # Plot Area
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Connect interactive events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Status Bar
        self.status_var = tk.StringVar(value="Tip: Click any data point for detailed analytics.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def get_aggregated_data(self, month=None):
        selection = self.view_var.get()
        agg_level = self.agg_var.get()
        data = self.df
        
        if selection != "All Orders":
            data = data[data['Dish_Name'] == selection]
            
        if month:
            data = data[data['Date'].dt.month == month]
            
        # Base daily aggregation
        daily = data.groupby('Date')['Order_ID'].nunique().reset_index()
        daily.columns = ['Date', 'Count']
        daily = daily.set_index('Date').asfreq('D').fillna(0)
        
        # Resample based on aggregation level
        if agg_level == "Weekly":
            resampled = daily.resample('W').sum()
        elif agg_level == "Monthly":
            resampled = daily.resample('M').sum()
        else:
            resampled = daily
            
        return resampled

    def update_plot(self, month=None, is_forecast_month=False):
        self.current_month = month
        self.ax.clear()
        
        daily_data = self.get_aggregated_data(month)
        self.current_series = daily_data
        self.current_forecast = None
        
        label = self.view_var.get()
        agg_label = self.agg_var.get()
        title = f"{label} - {agg_label} Orders"
        
        # Plot Historical Data
        self.ax.plot(daily_data.index, daily_data['Count'], marker='o', linestyle='-', 
                     markersize=5, label=f"Historical ({label})", color=self.colors['hist'], picker=5)

        # Add Forecast if enabled (only in year view, not in month drill-down)
        if self.forecast_var.get() and HAS_STATSMODELS and month is None:
            try:
                # Forecasting based on current aggregation
                steps = 30 if agg_label == "Daily" else (5 if agg_label == "Weekly" else 3)
                
                selected_model = self.model_var.get()
                if "SARIMA" in selected_model:
                    # SARIMA with weekly seasonality (7 days)
                    model = SARIMAX(daily_data['Count'], 
                                   order=(1, 1, 1), 
                                   seasonal_order=(1, 1, 1, 7))
                    model_fit = model.fit(disp=False)
                    forecast = model_fit.forecast(steps=steps)
                    model_desc = "SARIMA (1,1,1)x(1,1,1,7)"
                else:
                    # Classic ARIMA
                    model = ARIMA(daily_data['Count'], order=(5, 1, 0))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=steps)
                    model_desc = "ARIMA (5,1,0)"
                
                self.current_forecast = forecast
                
                # Use SOLID line like historical data, just different color
                self.ax.plot(forecast.index, forecast, color=self.colors['forecast'], 
                             linestyle='-', marker='o', markersize=5, linewidth=2, 
                             label=f"{steps}-Day Forecast ({selected_model})", picker=5)
                title += f" + {selected_model} Prediction"
            except Exception as e:
                print(f"Forecasting error: {e}")

        if month:
            month_name = datetime.date(2023, month, 1).strftime('%B')
            title = f"{month_name} Detail: {title}"
            self.status_var.set(f"Viewing {month_name}. Click points for stats.")
        else:
            self.status_var.set(f"Viewing Full Year ({agg_label}). Click any point to explore.")

        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Order Volume")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.legend()
        
        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.canvas.draw()

    def show_analytics(self, date_val, count_val, is_forecast=False):
        # Formulate date string
        if isinstance(date_val, datetime.datetime):
            date_str = date_val.strftime('%Y-%m-%d')
        else:
            date_str = str(date_val)

        agg_level = self.agg_var.get()
        dish = self.view_var.get()
        
        type_str = "PREDICTED" if is_forecast else "HISTORICAL"
        msg = f"--- {type_str} ANALYTICS ---\n\n"
        msg += f"Dish: {dish}\n"
        msg += f"Time: {date_str} ({agg_level})\n"
        msg += f"Orders: {count_val:.2f}\n\n"
        
        if not is_forecast:
            # Add some context if historical
            avg = self.current_series['Count'].mean()
            diff = count_val - avg
            perc = (diff / avg) * 100 if avg != 0 else 0
            msg += f"Vs Average: {'+' if diff >=0 else ''}{diff:.2f} ({perc:.1f}%)\n"
            msg += "\nTip: Click 'Reset' to go back to Yearly view if you are in Month view."
        else:
            msg += f"Model: {self.model_var.get()}\n"
            msg += "Confidence: Medium\n"
            msg += "Trend: " + ("Rising" if count_val > self.current_series['Count'].iloc[-1] else "Falling")
        
        messagebox.showinfo(f"{type_str} Detail", msg)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        # Search for nearest point
        x_date = mdates.num2date(event.xdata).replace(tzinfo=None)
        
        # Check forecast points first if enabled (top layer)
        if self.current_forecast is not None:
            # Find closest forecast point
            time_diffs = np.abs((self.current_forecast.index - x_date).total_seconds())
            if np.min(time_diffs) < (3600 * 24 * 7): # Within scope
                idx = np.argmin(time_diffs)
                clicked_forecast_date = self.current_forecast.index[idx]
                clicked_forecast_value = self.current_forecast.iloc[idx]
                
                # If in year view and daily aggregation, drill down to that forecast month
                if self.current_month is None and self.agg_var.get() == "Daily":
                    # Show forecast month drill-down
                    self.show_forecast_month_detail(clicked_forecast_date.month, clicked_forecast_date.year)
                else:
                    # Just show analytics
                    self.show_analytics(clicked_forecast_date, clicked_forecast_value, is_forecast=True)
                return

        # Check historical points
        time_diffs = np.abs((self.current_series.index - x_date).total_seconds())
        if np.min(time_diffs) < (3600 * 24 * 7):
            idx = np.argmin(time_diffs)
            clicked_date = self.current_series.index[idx]
            clicked_count = self.current_series['Count'].iloc[idx]
            
            # If we are in Year view and Daily aggregation, zoom to month
            if self.current_month is None and self.agg_var.get() == "Daily":
                self.update_plot(month=clicked_date.month)
            else:
                self.show_analytics(clicked_date, clicked_count)

    def show_forecast_month_detail(self, month, year):
        """Show detailed forecast for a specific month"""
        self.ax.clear()
        
        # Generate forecast for the entire dataset
        daily_data = self.get_aggregated_data(month=None)
        
        try:
            selected_model = self.model_var.get()
            if "SARIMA" in selected_model:
                model = SARIMAX(daily_data['Count'], 
                               order=(1, 1, 1), 
                               seasonal_order=(1, 1, 1, 7))
                model_fit = model.fit(disp=False)
            else:
                model = ARIMA(daily_data['Count'], order=(5, 1, 0))
                model_fit = model.fit()
            
            # Forecast 60 days to ensure we cover the clicked month
            forecast = model_fit.forecast(steps=60)
            
            # Filter to just the clicked month
            month_forecast = forecast[forecast.index.month == month]
            
            if len(month_forecast) == 0:
                messagebox.showwarning("No Data", f"No forecast data available for month {month}")
                return
            
            # Plot the forecast month detail
            self.ax.plot(month_forecast.index, month_forecast, marker='o', linestyle='-', 
                         markersize=5, label=f"Forecast Detail", color=self.colors['forecast'])
            
            month_name = datetime.date(year, month, 1).strftime('%B %Y')
            self.ax.set_title(f"Forecasted Orders - {month_name} (Daily)", fontsize=14, fontweight='bold')
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Predicted Order Volume")
            self.ax.grid(True, linestyle='--', alpha=0.6)
            self.ax.legend()
            
            # Show analytics summary
            avg_forecast = month_forecast.mean()
            total_forecast = month_forecast.sum()
            
            self.status_var.set(f"Forecast for {month_name}: Avg={avg_forecast:.1f}/day, Total≈{total_forecast:.0f} orders")
            
            self.fig.autofmt_xdate()
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Show popup with analytics
            msg = f"--- FORECAST MONTH ANALYTICS ---\n\n"
            msg += f"Month: {month_name}\n"
            msg += f"Average Daily Orders: {avg_forecast:.2f}\n"
            msg += f"Estimated Total: {total_forecast:.0f}\n"
            msg += f"Days Forecasted: {len(month_forecast)}\n\n"
            msg += "Note: This is a prediction based on historical patterns.\n"
            msg += "Click 'Reset View' to return to the main dashboard."
            
            messagebox.showinfo("Forecast Analytics", msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate forecast details: {e}")

    def reset_view(self):
        self.agg_var.set("Daily")
        self.update_plot(month=None)

if __name__ == "__main__":
    root = tk.Tk()
    # Simple style enhancement
    style = ttk.Style()
    style.theme_use('clam')
    app = RestaurantDashboard(root)
    root.mainloop()
