import tkinter as tk
from tkinter import messagebox, scrolledtext
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from matplotlib.widgets import SpanSelector
import numpy as np

span_selector = None

def fetch_stock_data(ticker, start="2022-01-01", end="2023-01-01"):
    """
    Fetches historical stock data for a given ticker within a specified date range.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end)
    return hist


def onselect(xmin, xmax):
    """
    Function to handle the event when a region is selected.
    This function can be enhanced to perform operations on the selected region, such as predictions.
    """
    int_xmin, int_xmax = np.searchsorted(data.index, [xmin, xmax])
    selected_region = data.iloc[int_xmin:int_xmax + 1]
    print("Selected region from", data.index[int_xmin], "to", data.index[int_xmax])

def plot_stock_data(data, frame):
    """
    Plots the stock data on a given frame and enables region selection.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.set_title('Stock Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()

    # Enable the span selector on the plot
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))

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