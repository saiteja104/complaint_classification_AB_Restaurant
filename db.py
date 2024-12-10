from datetime import datetime
import pandas as pd
import mysql.connector
from mysql.connector import Error

# Function to insert complaint into the MySQL database
def insert_review(timestamp, complaint, category, severity, urgency, service_issue, food_quality, atmosphere, value_for_money, hygiene, food_options, positive_review):
    try:
        conn = mysql.connector.connect(
            host='localhost',         # e.g., 'localhost'
            database='restaurant_review',
            user='root',     # e.g., 'root'
            password='Threshold@123'   # Your MySQL password
        )
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO restaurant_ab(timestamp, complaint, category, severity, urgency, Service_Issue, Food_Quality, Value_for_Money, Food_Options, Atmosphere, Hygiene, Positive_Review) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    timestamp,
                    complaint,
                    category,
                    severity,
                    urgency,
                    int(service_issue),  # Convert numpy.int64 to native Python int
                    int(food_quality),   # Convert numpy.int64 to native Python int
                    int(value_for_money),# Convert numpy.int64 to native Python int
                    int(food_options),   # Convert numpy.int64 to native Python int
                    int(atmosphere),     # Convert numpy.int64 to native Python int
                    int(hygiene),        # Convert numpy.int64 to native Python int
                    int(positive_review) # Convert numpy.int64 to native Python int
                )
            )
            conn.commit()
            cursor.close()
    except Error as e:
        print(f"Error: {e}")
    finally:
        if conn.is_connected():
            conn.close()

# Example usage
dataset = pd.read_csv(r"E:\Gradious_Final_Project\database\Final_Dataset.csv")
print(dataset.head(1))

# Create a new DataFrame that matches the structure of the reviews table
category_columns = ['Service Issue', 'Food Quality', 'Atmosphere', 'Value for Money', 'Hygiene', 'Food Options', 'Positive Review']
transformed_data = pd.DataFrame({
    'timestamp': dataset['Review_Date'],
    'complaint': dataset['Cleaned_Review'],
    'category': dataset['Mapped_Category'],  # Use the predicted category
    'severity': dataset['Severity'],
    'urgency': dataset['Urgency'],
    'Service Issue': dataset['Service Issue'],
    'Food Quality': dataset['Food Quality'],
    'Atmosphere': dataset['Atmosphere'],
    'Value for Money': dataset['Value for Money'],
    'Hygiene': dataset['Hygiene'],
    'Food Options': dataset['Food Options'],
    'Positive Review': dataset['Positive Review']
})

transformed_data.dropna(inplace=True)

for i in range(len(transformed_data)):
    insert_review(
        transformed_data['timestamp'][i],
        transformed_data['complaint'][i],
        transformed_data['category'][i],
        transformed_data['severity'][i],
        transformed_data['urgency'][i],
        transformed_data['Service Issue'][i],
        transformed_data['Food Quality'][i],
        transformed_data['Atmosphere'][i],
        transformed_data['Value for Money'][i],
        transformed_data['Hygiene'][i],
        transformed_data['Food Options'][i],
        transformed_data['Positive Review'][i]
    )