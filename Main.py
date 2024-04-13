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

fig = None
span_selector = None

def fetch_stock_data(ticker, start="2022-01-01", end="2023-01-01"):
    """
    Fetches historical stock data for a given ticker within a specified date range.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end)
    return hist

highlight_patch = None

def onselect(xmin, xmax):
    """
    Function to handle the event when a region is selected and highlight the region.
    """
    global highlight_patch, fig
    
    # Convert matplotlib float dates to datetime objects
    xmin_datetime = num2date(xmin)
    xmax_datetime = num2date(xmax)

    # Assure that data.index is a DatetimeIndex with timezone 'UTC'
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, utc=True)

    # Use searchsorted to find the indices of the dates in the data index
    int_xmin = data.index.searchsorted(xmin_datetime)
    int_xmax = data.index.searchsorted(xmax_datetime)

    # Adjust int_xmax if it's exactly on a data point
    if int_xmax < len(data.index) and data.index[int_xmax] == xmax_datetime:
        int_xmax += 1

    # Get the actual plot limits
    x_values = data.index.to_pydatetime()
    y_values = data['Close'].values

    # Clear the previous highlighted patch if it exists
    if highlight_patch:
        highlight_patch.remove()

    # Add a new patch to highlight the selected region
    ax = plt.gca()  # Get the current axes
    highlight_patch = ax.axvspan(x_values[int_xmin], x_values[int_xmax - 1], color='yellow', alpha=0.3)

    # Redraw the figure to show the changes
    fig.canvas.draw()

    print("Selected region from", data.index[int_xmin], "to", data.index[int_xmax - 1])

def plot_stock_data(data, frame):
    """
    Plots the stock data on a given frame and enables region selection.
    """
    global fig
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.set_title('Stock Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()

    # Enable the span selector on the plot
    span = SpanSelector(
    ax,
    onselect,
    "horizontal",
    useblit=True,
    props=dict(alpha=0.5, facecolor="tab:blue"),
    interactive=True,
    drag_from_anywhere=True
)

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    return span

def get_data():
    global span_selector
    ticker = ticker_entry.get()
    start = start_entry.get()
    end = end_entry.get()
    if not ticker or not start or not end:
        messagebox.showerror("Input error", "All fields are required")
        return
    try:
        global data
        data = fetch_stock_data(ticker, start, end)
        output_text.config(state=tk.NORMAL)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, data.to_string())
        output_text.config(state=tk.DISABLED)

        # Clear the previous frame content
        for widget in graph_frame.winfo_children():
            widget.destroy()

        span_selector = plot_stock_data(data, graph_frame)  # Update the global span_selector reference
    except Exception as e:
        messagebox.showerror("Fetch error", str(e))

def main():
    global ticker_entry, start_entry, end_entry, output_text, graph_frame

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

    fetch_button = tk.Button(app, text="Fetch Data", command=get_data)
    fetch_button.pack()

    output_text = scrolledtext.ScrolledText(app, width=80, height=10)
    output_text.pack()
    output_text.config(state=tk.DISABLED)

    # Frame to hold the graph
    graph_frame = tk.Frame(app)
    graph_frame.pack(fill=tk.BOTH, expand=True)

    app.mainloop()

if __name__ == "__main__":
    main()