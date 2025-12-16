# Import Dependencies
import logging
import os
from zipfile import ZipFile
import pandas as pd

# Setup logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create Class for ZIP Ingestion
class ZipDataIngestor():
    def ingest(self, file_path:str) -> pd.DataFrame:
        """Extracts a zip file and returns the content as a pandas DataFrame"""
        try:
            # Ensure the file is a zip file
            if not file_path.endswith('.zip'):
                raise ValueError("The provided file is not a zip file")
            
            # Extract zip file
            with ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall("extracted_data")
            
            # Find CSV file in extracted files
            extracted_files = os.listdir("extracted_data")
            csv_files = [f for f in extracted_files if f.endswith('.csv')]
            
            
            if len(csv_files) == 0:
                raise FileNotFoundError("No CSV file found in the extracted data.")
            
            # Read the CSV files into DataFrame
            csv_file_path = os.path.join("extracted_data", csv_files[0])
            df = pd.read_csv(csv_file_path)
            
            logging.info("Data Ingestion Completed Successfully")
            # Return the dataframe
            return df
        except Exception as e:
            logging.error(f"Error in Data Ingestion: {e}")
            raise e

class CSVDataIngestor():
    def ingest(self, file_path:str) -> pd.DataFrame:
        """Reads a CSV file and returns the content as a pandas DataFrame"""
        try:
            # Ensure the file is a CSV file
            if not file_path.endswith('.csv'):
                raise ValueError("The provided file is not a CSV file")
            
            # Read the CSV file into DataFrame
            df = pd.read_csv(file_path)
            
            logging.info("CSV Data Ingestion Completed Successfully")
            # Return the dataframe
            return df
        except Exception as e:
            logging.error(f"Error in CSV Data Ingestion: {e}")
            raise e