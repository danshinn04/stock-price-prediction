import tkinter as tk
import matplotlib
from tkinter import messagebox, scrolledtext
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from matplotlib.widgets import SpanSelector
import numpy as np
from pandas import to_datetime
from matplotlib.dates import num2date, date2num
import pytz
from Linear_Regression import perform_linear_regression, perform_polynomial_regression, perform_gradient_boosting_regression, perform_random_forest_regression, perform_decision_tree_regression, perform_svr, perform_elasticnet_regression, perform_lasso_regression, perform_ridge_regression
from matplotlib.lines import Line2D
from Neural_Network_GUI import Neural_Network_GUI
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler


fig = None
span_selector = None
span_selector_active = False
span_cid = None
ax = None 
highlight_patch = None
model_current = None
prediction_line = None
fitted_model = None
prediction_result_label = None
nn_model = None
scaler = MinMaxScaler(feature_range=(0, 1))
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


def create_sequences(data, steps=60):
    X, y = [], []
    for i in range(steps, len(data)):
        X.append(data[i-steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def create_and_train_nn(data, network_structure, epochs=10, batch_size=32):
    global nn_model, scaler
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    X, y = create_sequences(scaled_data)
    
    nn_model = Sequential()
    nn_model.add(LSTM(units=network_structure[0], return_sequences=True, input_shape=(X.shape[1], 1)))
    nn_model.add(Dropout(0.2))
    for layer_size in network_structure[1:-1]:
        nn_model.add(LSTM(units=layer_size, return_sequences=True))
        nn_model.add(Dropout(0.2))
    nn_model.add(LSTM(units=network_structure[-1]))
    nn_model.add(Dropout(0.2))
    nn_model.add(Dense(units=1))
    
    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    nn_model.fit(X, y, epochs=epochs, batch_size=batch_size)

def predict_with_nn(data):
    global nn_model, scaler
    last_sequence = np.array(data)
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    last_sequence_scaled = np.reshape(last_sequence_scaled, (1, last_sequence_scaled.shape[0], 1))
    predicted_price_scaled = nn_model.predict(last_sequence_scaled)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    return predicted_price[0][0]


def onselect(xmin, xmax):
    global highlight_patch, fig, ax, data, prediction_line, model_current, fitted_model

    # Convert matplotlib float dates to datetime objects
    xmin_datetime = num2date(xmin)
    xmax_datetime = num2date(xmax)

    # Assure that data.index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Find the indices of the dates in the data index
    int_xmin = data.index.searchsorted(xmin_datetime)
    int_xmax = data.index.searchsorted(xmax_datetime)

    # Adjust if xmax is beyond the last data point
    int_xmax = min(int_xmax, len(data.index) - 1)

    # Extract the region from the DataFrame for regression
    selected_data = data.iloc[int_xmin:int_xmax + 1]
    X = selected_data.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)
    y = selected_data['Close'].values

    # Select and perform the correct regression based on the current model
    regression_functions = {
        "Linear Regression": perform_linear_regression,
        "Ridge Regression": perform_ridge_regression,
        "Lasso Regression": perform_lasso_regression,
        "Elastic Net Regression": perform_elasticnet_regression,
        "SVR": perform_svr,
        "Decision Tree Regression": perform_decision_tree_regression,
        "Random Forest Regression": perform_random_forest_regression,
        "Gradient Boosting Regression": perform_gradient_boosting_regression,
        "Polynomial Regression": perform_polynomial_regression
    }
    fitted_model = regression_functions[model_current](X, y)

    # Make predictions and update the plot
    if model_current == "Polynomial Regression":
        # For polynomial, plot over a finer grid
        fine_X = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
        predictions = fitted_model.predict(fine_X)
        fine_dates = [num2date(date2num(pd.Timestamp.fromordinal(int(date)))) for date in fine_X]
        if prediction_line:
            prediction_line.remove()
        prediction_line, = ax.plot(fine_dates, predictions, color='red', label='Polynomial Fit')
    else:
        # For other regressions, use the original data points
        predictions = fitted_model.predict(X)
        if prediction_line:
            prediction_line.remove()
        prediction_line, = ax.plot(selected_data.index, predictions, color='red', label='Regression Fit')

    # Redraw the figure to show changes
    ax.legend()
    fig.canvas.draw()

    # Print model details in the console if necessary
    if hasattr(fitted_model, 'coef_'):
        print("Model coefficients:", fitted_model.coef_)
    if hasattr(fitted_model, 'intercept_'):
        print("Model intercept:", fitted_model.intercept_)

        
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

def setup_and_train_nn():  # Pass nn_gui as an argument
    # Get user-defined network structure, e.g., from GUI elements
    global nn_gui
    network_structure = nn_gui.get_network_structure() 
    ticker = ticker_entry.get()
    start = start_entry.get()
    end = end_entry.get()
    data = fetch_stock_data(ticker, start, end)['Close'].tolist()
    create_and_train_nn(data, network_structure, epochs=50, batch_size=32)
    messagebox.showinfo("Training Complete", "The neural network has been trained successfully.")

def predict_on_new_data():
    date_str = prediction_date_entry.get()  # Assuming this gets the date to predict for
    data = fetch_stock_data(ticker_entry.get(), start=date_str, end=date_str)['Close'].tolist()
    predicted_price = predict_with_nn(data)
    prediction_result_label.config(text=f"Predicted price for {date_str}: ${predicted_price:.2f}")


def make_prediction(date_str):
    global fitted_model, prediction_result_label, ax, fig, data
    
    try:
        prediction_date = pd.to_datetime(date_str)
        prediction_ordinal = np.array([[prediction_date.toordinal()]])
        
        # Check if we have a fitted model to make a prediction with
        if fitted_model and hasattr(fitted_model, 'predict'):
            # Clear the current axes
            ax.clear()
            
            # Re-add the elements to the plot
            ax.plot(data.index, data['Close'], label='Close Price', color='blue')  # Re-plotting original data
            ax.set_title('Stock Price Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price')
            
            # Redraw the model prediction line
            if model_current == "Linear Regression":
                selected_indices = data.index.map(lambda x: x.toordinal()).values.reshape(-1, 1)
                predictions = fitted_model.predict(selected_indices)
                ax.plot(data.index, predictions, color='red', label='Latest Fitted Model for prediction')

            # Plot the prediction point
            predicted_price = fitted_model.predict(prediction_ordinal)
            prediction_matplotlib_date = date2num(prediction_date)
            ax.plot(prediction_matplotlib_date, predicted_price[0], 'bo', label='Prediction')  # 'bo' is for blue dot
            
            # Draw a vertical line at the prediction date
            ax.axvline(x=prediction_matplotlib_date, color='blue', linestyle='--', alpha=0.5)
            
            # Update the prediction result label
            if prediction_result_label:
                prediction_result_label.config(text=f"Predicted price for {date_str}: ${predicted_price[0]:.2f}")
            else:
                print("Prediction label is not initialized.")

            ax.legend(loc='best')  # Re-add the legend to display labels
            fig.canvas.draw()  # Redraw the figure to show the changes
        else:
            messagebox.showerror("Prediction error", "Please select a region and fetch data to fit the model before making a prediction.")
    except Exception as e:
        messagebox.showerror("Prediction error", str(e))


def main():
    global ticker_entry, start_entry, end_entry, output_text, graph_frame, toggle_button, model_selector, prediction_result_label, nn_gui  # Add nn_gui to global variables

    app = tk.Tk()
    app.title("Stock Data Fetcher")

    # Create a frame for the entries and buttons
    control_frame = tk.Frame(app)
    control_frame.pack(side=tk.TOP, fill=tk.X)

    tk.Label(control_frame, text="Ticker Symbol:").pack(side=tk.LEFT)
    ticker_entry = tk.Entry(control_frame)
    ticker_entry.pack(side=tk.LEFT)

    tk.Label(control_frame, text="Start Date (YYYY-MM-DD):").pack(side=tk.LEFT)
    start_entry = tk.Entry(control_frame)
    start_entry.pack(side=tk.LEFT)

    tk.Label(control_frame, text="End Date (YYYY-MM-DD):").pack(side=tk.LEFT)
    end_entry = tk.Entry(control_frame)
    end_entry.pack(side=tk.LEFT)

    # Model selection dropdown
    tk.Label(control_frame, text="Select Regression Model:").pack(side=tk.LEFT)
    selected_model = tk.StringVar(app)
    selected_model.set(models[0])  # default value
    selected_model.trace("w", lambda *args: on_model_change(selected_model.get()))
    model_selector = tk.OptionMenu(control_frame, selected_model, *models)
    model_selector.pack(side=tk.LEFT)

    # Neural Network GUI
    nn_gui_frame = tk.Frame(app)
    nn_gui_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Initialize the neural network GUI and place it in the nn_gui_frame
    nn_gui = Neural_Network_GUI(nn_gui_frame)

    nn_train_button = tk.Button(control_frame, text="Train Neural Network", command=setup_and_train_nn)
    nn_train_button.pack(side=tk.LEFT)

    predict_button = tk.Button(control_frame, text="Make Prediction", command=predict_on_new_data)
    predict_button.pack(side=tk.LEFT)

    # Fetch Data Button
    fetch_button = tk.Button(control_frame, text="Fetch Data", command=lambda: get_data(selected_model.get()))
    fetch_button.pack(side=tk.LEFT)

    toggle_button = tk.Button(app, text="Enable Select", command=toggle_span_selector)
    toggle_button.pack()

    # Prediction Date Entry and Button
    tk.Label(control_frame, text="Predict for Date (YYYY-MM-DD):").pack(side=tk.LEFT)
    prediction_date_entry = tk.Entry(control_frame)
    prediction_date_entry.pack(side=tk.LEFT)
    predict_button = tk.Button(control_frame, text="Make Prediction", command=lambda: make_prediction(prediction_date_entry.get()))
    predict_button.pack(side=tk.LEFT)

    # Label to display the prediction result
    prediction_result_label = tk.Label(control_frame, text="")
    prediction_result_label.pack(side=tk.LEFT)

    # Text box for stock data output below the control frame
    output_text = scrolledtext.ScrolledText(app, width=80, height=10)
    output_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    output_text.config(state=tk.DISABLED)

    # Frame to hold the graph, below the output text box
    graph_frame = tk.Frame(app)
    graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    app.mainloop()

if __name__ == "__main__":
    main()