import torch
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from textblob import TextBlob
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import warnings



# Initialize the tokenizer and model from a local directory
model_directory = "E:/Gradious_Final_Project/model"  # Path to the directory containing the model files

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_directory)
model = BertForSequenceClassification.from_pretrained(model_directory, num_labels=7).to("cuda" if torch.cuda.is_available() else "cpu")

# Load the DataFrame
try:
    df = pd.read_csv(r"E:\Gradious_Final_Project\Preprocessing\Final_Dataset.csv")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Define categories to add
categories_to_add = ['Service Issue', 'Food Quality', 'Atmosphere', 'Value for Money', 'Hygiene', 'Food Options', 'Positive Review']

def predict_multi_label(review_text):
    """Predicts the multi-label categories for a given review text."""
    inputs = tokenizer(review_text, truncation=True, padding=True, max_length=128, return_tensors='pt').to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits).cpu().numpy()
    threshold = 0.5
    predicted_labels = [label for i, label in enumerate(categories_to_add) if predictions[0][i] >= threshold]
    if not predicted_labels:
        return [label for i, label in enumerate(categories_to_add) if predictions[0][i] == max(predictions[0])]
    else:
        return predicted_labels

def get_sentiment(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity

# Function to determine severity based on sentiment score
def get_severity(sentiment_score):
    if sentiment_score > 0.1:
        return "Low"
    elif sentiment_score >= -0.1:
        return "Medium"
    else:
        return "High"

# Function to determine urgency based on sentiment score
def get_urgency(sentiment_score):
    return 'Urgent' if sentiment_score < -0.1 else 'Non-Urgent'

def generate_response(categories, severity, urgency):
    base_response = (
        "Hi,\n"
        "We regret the inconveniences you have faced."
    )

    category_sentences = {
        'Service Issue': "We understand that service issues can be frustrating, and we are committed to resolving this.",
        'Food Options': "We appreciate your feedback on our food options, and we are continually working to enhance our menu.",
        'Food Quality': "We take food quality seriously, and we apologize for not meeting your expectations.",
        'Atmosphere': "Creating a pleasant atmosphere is important to us, and we are sorry that we fell short.",
        'Value for Money': "We strive to provide value for money, and we appreciate your input on this matter.",
        'Hygiene': "Hygiene is our top priority, and we are dedicated to maintaining the highest standards.",
        'Positive Review': "We are glad to hear that you had a positive experience with us."
    }
    if categories == ['Positive Review']:
        return f"We are delighted to hear about your positive experience with us. We look forward to serving you again soon.\n\n"
    


    category_responses = [category_sentences[category] for category in categories if category in category_sentences]

    if severity == "High":
        severity_response = "We recognize the urgency of this matter."
    elif severity == "Medium":
        severity_response = "We acknowledge the issues you've raised."
    else:
        severity_response = "We appreciate your feedback."

    if urgency == "Urgent":
        urgency_response = "We will prioritize your concern and address it immediately."
    else:
        urgency_response = "We will take your feedback into consideration and work on improvements."

    full_response = f"{base_response} {' '.join(category_responses)} {severity_response}\n{urgency_response}"

   

    full_response += "\n\nWe request you to share your contact information at wecare@restaurant.com. We shall connect with you in no time and assist you with the issue raised. We assure you that this won't happen again next time.\n\nTeam Restaurant"

    return full_response

# Function to fetch recent reviews
def fetch_recent_reviews(days):
    try:
        conn = mysql.connector.connect(
            host='localhost',         # e.g., 'localhost'
            database='restaurant_review',
            user='root',     # e.g., 'root'
            password='Threshold@123'   # Your MySQL password
        )
        if conn.is_connected():
            # Calculate the date 10 days ago
            days_ago = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Query using STR_TO_DATE to convert your timestamp column
            query = """
                SELECT * 
                FROM restaurant_c 
                WHERE STR_TO_DATE(timestamp, '%d-%m-%Y') >= %s
            """
            
            # Execute the query
            cursor = conn.cursor()
            cursor.execute(query, (days_ago,))
            result = cursor.fetchall()
            columns = [i[0] for i in cursor.description]
            reviews_df = pd.DataFrame(result, columns=columns)
            cursor.close()
            return reviews_df
    except Error as e:
        print(f"Error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    finally:
        if conn.is_connected():
            conn.close()

def get_complaint_info(days):
    # Filter for recent complaints based on Review_Days
    recent_complaints = fetch_recent_reviews(days)
    total_complaints = recent_complaints.shape[0]
    # st.write(recent_complaints.head())
    # Define the category columns you want to analyze
    category_columns = ['Service Issue', 'Food Quality', 'Atmosphere', 'Value for Money', 'Hygiene', 'Food Options', 'Positive Review']
    
    # Create a dictionary to hold counts for each category
    category_counts = {category: 0 for category in category_columns}
    
    # Count occurrences for each category based on boolean values (assumed 1 for complaint, 0 for no complaint)
    for category in category_columns:
        category_counts[category] = recent_complaints[category].sum()
    
    # Prepare the data for returning
    categories = ', '.join(category_counts.keys())
    counts = ', '.join(map(str, category_counts.values()))
    
    return {
        'total_complaints': total_complaints,
        'categories': categories,
        'counts': counts,
        'reviews': recent_complaints
    }
# Sidebar selection for user type
user_type = st.sidebar.radio("Select User Type", ("Customer", "Restaurant Owner"))

# Main content based on user type
st.title("Restaurant Complaint Management System")

if user_type == "Restaurant Owner":
    # Section for Restaurant Owner
    st.header("Complaint Counts Dashboard")
    days = st.number_input("Enter the number of days:", min_value=0, value=30)

    if st.button("Get Complaint Info"):
        if days < 0:
            st.error("Please enter a positive number of days.")
        else:
            with st.spinner("Fetching complaint information..."):
                result = get_complaint_info(days)

            # Display results
            st.write("Total Complaints:", result['total_complaints'])

            # Calculate category counts from specific columns
            category_columns = ['Service Issue', 'Food Quality', 'Atmosphere', 'Value for Money', 'Hygiene', 'Food Options']
            category_counts = {col: result['reviews'][col].sum() for col in category_columns}

            st.write("Category Counts:")
            for category, count in category_counts.items():
                st.write(f"{category}: {count}")

            st.write("Reviews Information:")
            reviews_df = result['reviews']

            # Select relevant columns to display
            selected_columns = ['complaint','severity', 'urgency', 'category','timestamp']
            reviews_df = reviews_df[selected_columns]
            
            # Display the DataFrame
            st.dataframe(reviews_df.reset_index(drop=True))

def insert_review(timestamp, complaint, category, severity, urgency, service_issue, food_quality, atmosphere, value_for_money, hygiene, food_options, positive_review):
    conn = None
    try:
        # Ensure timestamp is in the correct format 'dd-mm-yyyy'
        if isinstance(timestamp, datetime):
            timestamp = timestamp.strftime('%d-%m-%Y')  # Convert to 'dd-mm-yyyy' format
        elif isinstance(timestamp, str):
            # Check if timestamp is already in 'dd-mm-yyyy' format
            try:
                datetime.strptime(timestamp, '%d-%m-%Y')  # Try parsing string
            except ValueError:
                st.error(f"Incorrect timestamp format: {timestamp}. Expected format is 'dd-mm-yyyy'.")
                return
        
        # Establish connection to the MySQL database
        conn = mysql.connector.connect(
            host='localhost',
            database='restaurant_review',
            user='root',
            password='Threshold@123'
        )

        # Check if connection is established
        if conn.is_connected():
            cursor = conn.cursor()

            # Insert the review into the database
            cursor.execute(
                "INSERT INTO restaurant_c(timestamp, complaint, category, severity, urgency, `Service Issue`, `Food Quality`, `Value for Money`, `Food Options`, `Atmosphere`, Hygiene, `Positive Review`) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    timestamp,  # Insert the timestamp in 'dd-mm-yyyy' format
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

            # Commit changes to the database
            conn.commit()

            if cursor.rowcount == 1:
                st.success("Review inserted successfully into the database.")

                # Optional: Fetch and display the last inserted record for verification
                cursor.execute("SELECT * FROM restaurant_c ORDER BY id DESC LIMIT 1")
                last_record = cursor.fetchone()
                st.write("Last inserted record:", last_record)
            else:
                st.error("Failed to insert the review.")

            cursor.close()
        else:
            st.error("Failed to establish database connection.")

    except Error as e:
        st.error(f"Error: {e}")  # Provide error message in Streamlit output

    finally:
        if conn and conn.is_connected():
            conn.close()  # Ensure the connection is closed properly

if user_type == "Customer":
    # Section for Customer
    st.header("Submit Your Complaint")
    
    user_review = st.text_area("Enter your review:")
    submit_button = st.button("Submit Review")
    
    if submit_button and user_review.strip():
        with st.spinner("Processing your review..."):
            sentiment_score = get_sentiment(user_review)
            categories = predict_multi_label(user_review)
            severity = get_severity(sentiment_score)
            urgency = get_urgency(sentiment_score)
    
            # Generate auto-response
            auto_response = generate_response(categories, severity, urgency)
            
            # Display auto-response
            st.write(auto_response)
            
            # Record new entry in the DataFrame
            new_entry = pd.DataFrame({
                'Review': [user_review],
                'Predicted_Category': [', '.join(categories)],
                'Severity': [severity],
                'Urgency': [urgency],
                'Review_Days': [0],  # Set this to 0 for new entries; adjust based on your logic
                'Response_Days': [0],  # Adjust if needed
                # Include additional columns as necessary (e.g., Service Issue, Food Quality, etc.)
                'Service Issue': [int("Service Issue" in categories)],
                'Food Quality': [int("Food Quality" in categories)],
                'Atmosphere': [int("Atmosphere" in categories)],
                'Value for Money': [int("Value for Money" in categories)],
                'Hygiene': [int("Hygiene" in categories)],
                'Food Options': [int("Food Options" in categories)],
                'Positive Review': [int("Positive Review" in categories)]
            })
            insert_review(datetime.now(), user_review, ', '.join(categories), severity, urgency, int("Service Issue" in categories), int("Food Quality" in categories), int("Atmosphere" in categories), int("Value for Money" in categories), int("Hygiene" in categories), int("Food Options" in categories), int("Positive Review" in categories))

            
            st.success("Your review has been submitted and saved.")
            
        # # Optional display for confirmation
        # st.write("Review Summary:")
        # st.write(new_entry)
    else:
        st.info("Please enter a review to submit.")