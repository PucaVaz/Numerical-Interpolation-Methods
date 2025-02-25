def load_data(path):
    """
    Load data from a CSV file.

    Parameters:
    path (str): The file path to the CSV file.

    Returns:
    pandas.DataFrame: The data loaded into a DataFrame.
    """
    import pandas as pd
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
        return None