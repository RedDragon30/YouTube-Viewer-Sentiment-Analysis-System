import logging
import os
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create Class for Data Preprocessing
class DataPreprocessing():
    def clean_data(self, reddit:pd.DataFrame, youtube:pd.DataFrame) -> pd.DataFrame:
        try:
            # Step 1: Merge the dataframes
            # Rename the columns to be same in the dataframes
            reddit.rename(columns={'clean_comment': 'comment'}, inplace=True)
            youtube.rename(columns={'Comment': 'comment', 'Sentiment': 'category'}, inplace=True)
            
            # Convert string categories to numerical categories in youtube like reddit
            youtube['category'] = youtube['category'].map({'negative': -1, 'neutral': 0, 'positive': 1})
            
            # Now we will concatenate the dataframes
            comments = pd.concat([reddit, youtube], ignore_index=True)
            logging.info('Merge the dataframes Step Completed')
            
            # Step 2: Drop Missing Values
            comments.dropna(inplace=True)
            logging.info('Drop Missing Values Step Completed')
            
            # Step 3: Drop Duplicates
            comments.drop_duplicates(inplace=True)
            logging.info('Drop Duplicates Step Completed')
            
            # Step 4: Standarize the data
            # Remove empty comments
            comments = comments[~(comments['comment'].str.strip() == '')]
            
            # Convert to lowercase
            comments['comment'] = comments['comment'].str.lower()
            
            # Remove trailing and leading whitespaces
            comments['comment'] = comments['comment'].str.strip()
            
            # Remove URLs from comments
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            comments['comment'] = comments['comment'].str.replace(url_pattern, ' ', regex=True)
            
            # Remove '\n' character from comments
            comments['comment'] = comments['comment'].str.replace('\n', ' ', regex=True)
            
            # Remove non-alphanumeric characters, except punctuation
            comments['comment'] = comments['comment'].str.replace(r'[^A-Za-z0-9\s!?.,]', '', regex=True)
            
            # Remove stopwords but retain important ones for sentiment analysis
            stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
            comments['comment'] = comments['comment'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in stop_words])
            )
            
            # Lemmatize the words
            lemmatizer = WordNetLemmatizer()
            comments['comment'] = comments['comment'].apply(
            lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
            )
            logging.info('Data Standrizing Step Completed')
            
            # Step 5: Remap the class labels from [-1, 0, 1] to [2, 0, 1]
            comments['category'] = comments['category'].map({-1: 2, 0: 0, 1: 1})
            comments = comments.dropna(subset=['category'])
            logging.info('Remap the categories Step Completed')
            
            logging.info('Data Cleaning Completed Successfully')
            return comments
        except Exception as e:
            logging.error(f'Error in Data Cleaning: {e}')
            raise e
    
    def vectorize_data(self, X_train, X_test):
        try:
            # Initialize TF-IDF Vectorizer
            ngram_range = (1, 3)  # Trigram
            max_features = 1000  # Set max_features to 1000
            # Perform TF-IDF transformation
            vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
            # Fit on training data and transform both training and test data
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            
            # Save the vectorizer
            # Ensure the artifacts directory exists
            if not os.path.exists(r'.\artifacts'):
                os.makedirs(r'.\artifacts')
            
            # Ensure the pickle file is not existing
            file_path = os.path.join(r'.\artifacts', f'tfidf_vectorizer.pkl')
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    pickle.dump(vectorizer, f)
            
            logging.info('Data Vectorizing Completed Successfully')
            return X_train_tfidf, X_test_tfidf
        except Exception as e:
            logging.error(f'Error in Data Vectorizing: {e}')
            raise e