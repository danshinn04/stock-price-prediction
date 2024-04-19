import pandas as pd
import os
from datetime import datetime, timedelta

def daterange(start_date, end_date):
    """Generates dates within the specified range."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def merge_csv_files(tickers, start_date, end_date, base_path):
    """Merges CSV files for each ticker across a specified date range, setting column names explicitly."""
    # Prepare date format
    date_format = "%Y%m%d"

    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    # Define the column names based on new sample data
    column_names = ['date', 'minute', 'open', 'high', 'low', 'close', 'volume', 'ignore1', 'ignore2', 'ignore3']

    # Dictionary to hold data frames for each ticker
    data_frames = {ticker: pd.DataFrame() for ticker in tickers}

    # Iterate over each date in the range
    for single_date in daterange(start_date, end_date):
        folder_name = f"allstocks_{single_date.strftime(date_format)}"
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the folder for the date exists
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue
        
        # Process each ticker
        for ticker in tickers:
            file_name = f"table_{ticker.lower()}.csv"
            file_path = os.path.join(folder_path, file_name)
            
            # Check if the CSV file exists
            if not os.path.exists(file_path):
                print(f"Skipping missing file: {file_path}")
                continue

            # Read the CSV file without headers, setting the column names explicitly
            df = pd.read_csv(file_path, header=None, names=column_names)
            
            # Select only the necessary columns to avoid 'ignore' columns
            df = df[['date', 'minute', 'open', 'high', 'low', 'close', 'volume']]
            
            # Append the data frame to the corresponding ticker's data frame
            data_frames[ticker] = pd.concat([data_frames[ticker], df], ignore_index=True)

    # Save merged files
    for ticker, df in data_frames.items():
        output_file = os.path.join(base_path, f"{ticker}_merged.csv")
        df.to_csv(output_file, index=False)
        print(f"Data for {ticker} saved to {output_file}")

# Configuration
tickers = ["nvda", "aapl", "amzn", "msft", "nflx", "tsla", "amd", "xom", "v", "fb"]
start_date = "20190325"
end_date = "20190425"
base_path = '2-year sample'  # Modify <yourusername> accordingly

# Call the function
merge_csv_files(tickers, start_date, end_date, base_path)
