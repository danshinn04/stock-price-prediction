import tkinter as tk
from tkinter import messagebox, scrolledtext
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from matplotlib.widgets import SpanSelector
import numpy as np
from pandas import to_datetime
from matplotlib.dates import num2date
import pytz
from Linear_Regression import perform_linear_regression


fig = None
span_selector = None
span_selector_active = False
span_cid = None
ax = None 
highlight_patch = None
model_current = None
models = ["Linear Regression", "Polynomial Regression", "Ridge Regression",
          "Lasso Regression", "Elastic Net Regression", "SVR",
          "Decision Tree Regression", "Random Forest Regression", "Gradient Boosting Regression"]


def fetch_stock_data(ticker, start="2022-01-01", end="2023-01-01"):
    """
    Fetches historical stock data for a given ticker within a specified date range.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end)
    return hist

highlight_patch = None

def toggle_span_selector():
    global span_selector_active, span_selector, fig, ax, highlight_patch

    span_selector_active = not span_selector_active
    
    if span_selector_active:
        # Activate the span selector
        span_selector.set_active(True)
        toggle_button.config(text="Disable Select")
    else:
        # Deactivate the span selector
        span_selector.set_active(False)
        toggle_button.config(text="Enable Select")
        # Clear the highlighted region when disabling selection
        if highlight_patch:
            highlight_patch.remove()
            highlight_patch = None
        fig.canvas.draw_idle()


def onselect(xmin, xmax):
    global highlight_patch, fig, ax, data
    
    # Convert matplotlib float dates to datetime objects
    xmin_datetime = num2date(xmin)
    xmax_datetime = num2date(xmax)

    # Assure that data.index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Use searchsorted to find the indices of the dates in the data index
    int_xmin = data.index.searchsorted(xmin_datetime)
    int_xmax = data.index.searchsorted(xmax_datetime)

    # Adjust if xmax is beyond the last data point
    int_xmax = min(int_xmax, len(data.index) - 1)

    # Extract the region from the DataFrame
    selected_data = data.iloc[int_xmin:int_xmax + 1]

    # Convert the datetime index to a numeric value for regression
    X = selected_data.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)
    y = selected_data['Close'].values

    # Perform linear regression on the selected region
    if model_current == "Linear Regression":

        model = perform_linear_regression(X, y)

        # Make predictions
        predictions = model.predict(X)

    # Clear the previous highlighted patch if it exists
    if highlight_patch:
        highlight_patch.remove()

    # Highlight the selected region
    # ax.axvspan(data.index[int_xmin], data.index[int_xmax], color='yellow', alpha=0.3)

    # Plot the regression line linear
    if model_current == "Linear Regression":
        ax.plot(data.index[int_xmin:int_xmax + 1], predictions, color='red')

    # Redraw the figure to show the changes
    fig.canvas.draw()

    # Print model details in the console
    if model_current == "Linear Regression":
        print("Model coefficients:", model.coef_)
        print("Model intercept:", model.intercept_)


def plot_stock_data(data, frame):
    global fig, ax, span_selector, span_selector_active
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.set_title('Stock Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()

    # Always recreate the span selector when plotting new data
    span_selector = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                                 props=dict(alpha=0.5, facecolor='tab:blue'),
                                 interactive=True, drag_from_anywhere=True)
    # Set the active state of the span selector based on span_selector_active
    span_selector.set_active(span_selector_active)

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    return span_selector

def get_data(selected_model):
    global span_selector, data, model_current
    ticker = ticker_entry.get()
    start = start_entry.get()
    end = end_entry.get()
    model_current = selected_model # Unnecessary code for function just for test
    print("Selected Regression Model:", model_current)  # Same here 
    if not ticker or not start or not end:
        messagebox.showerror("Input error", "All fields are required")
        return
    
    try:
        # Fetch the stock data
        data = fetch_stock_data(ticker, start, end)
        
        # Update the output text box
        output_text.config(state=tk.NORMAL)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, data.to_string())
        output_text.config(state=tk.DISABLED)

        # If we've selected a region, only use that for regression
        if highlight_patch is not None:
            # Convert datetime to ordinal
            selected_data = data.loc[highlight_start:highlight_end]
            X = np.array([selected_data.index.map(lambda dt: dt.toordinal())]).T
            y = selected_data['Close'].values
        else:
            # Convert datetime to ordinal
            X = np.array([data.index.map(lambda dt: dt.toordinal())]).T
            y = data['Close'].values
        
        # Perform the selected regression model
        if selected_model == "Linear Regression":
            model = perform_linear_regression(X, y)
        

        # Clear and update the plot
        for widget in graph_frame.winfo_children():
            widget.destroy()
        plot_stock_data(data, graph_frame)

    except Exception as e:
        messagebox.showerror("Fetch error", str(e))

def on_model_change(selection):
    global model_current
    model_current = selection
    print("New model selected:", model_current)
    # get_data(model_current)


def main():
    global ticker_entry, start_entry, end_entry, output_text, graph_frame, toggle_button, model_selector

    app = tk.Tk()
    app.title("Stock Data Fetcher")

    tk.Label(app, text="Ticker Symbol:").pack()
    ticker_entry = tk.Entry(app)
    ticker_entry.pack()

    tk.Label(app, text="Start Date (YYYY-MM-DD):").pack()
    start_entry = tk.Entry(app)
    start_entry.pack()

    tk.Label(app, text="End Date (YYYY-MM-DD):").pack()
    end_entry = tk.Entry(app)
    end_entry.pack()

    # Model selection dropdown
    tk.Label(app, text="Select Regression Model:").pack()
    selected_model = tk.StringVar(app)
    selected_model.set(models[0])  # default value
    selected_model.trace("w", lambda *args: on_model_change(selected_model.get()))
    model_selector = tk.OptionMenu(app, selected_model, *models)
    model_selector.pack()

    fetch_button = tk.Button(app, text="Fetch Data", command=lambda: get_data(selected_model.get()))
    fetch_button.pack()

    output_text = scrolledtext.ScrolledText(app, width=80, height=10)
    output_text.pack()
    output_text.config(state=tk.DISABLED)

    toggle_button = tk.Button(app, text="Enable Select", command=toggle_span_selector)
    toggle_button.pack()

    # Frame to hold the graph
    graph_frame = tk.Frame(app)
    graph_frame.pack(fill=tk.BOTH, expand=True)

    app.mainloop()


if __name__ == "__main__":
    main()