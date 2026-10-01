import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "input_data"
OUTPUT_DIR = "."

print("--- Phase 3: Data Cleaning & Feature Engineering ---")

# 1. Load Raw Datasets
print("Loading raw CSV files...")
users_raw = pd.read_csv(f"{DATA_DIR}/Users.csv", sep=None, engine='python', on_bad_lines='skip')
books_raw = pd.read_csv(f"{DATA_DIR}/Books.csv", sep=None, engine='python', on_bad_lines='skip')
ratings_raw = pd.read_csv(f"{DATA_DIR}/Ratings.csv", sep=None, engine='python', on_bad_lines='skip')

print(f"Raw shapes -> Users: {users_raw.shape}, Books: {books_raw.shape}, Ratings: {ratings_raw.shape}")

# --- Clean Users ---
print("Cleaning Users dataset...")
users = users_raw.copy()
users['User-ID'] = pd.to_numeric(users['User-ID'], errors='coerce').astype('Int64')
users = users.dropna(subset=['User-ID']).copy()
users['User-ID'] = users['User-ID'].astype(int)

# Age cleaning & outlier treatment (Ages < 5 or > 100 set to NaN)
users['Age'] = pd.to_numeric(users['Age'], errors='coerce')
users['Age_Is_Missing'] = users['Age'].isna().astype(int)
users.loc[(users['Age'] < 5) | (users['Age'] > 100), 'Age'] = np.nan

# Impute median age for binning
median_age = users['Age'].median()
users['Age_Imputed'] = users['Age'].fillna(median_age)

# Binning Age into Age_Group
def get_age_group(age):
    if age < 18:
        return 'Children'
    elif age <= 35:
        return 'Youth'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'

users['Age_Group'] = users['Age_Imputed'].apply(get_age_group)

# Location parsing (City, State, Country)
location_split = users['Location'].astype(str).str.split(',', expand=True)
users['Country'] = location_split[2].str.strip() if location_split.shape[1] > 2 else 'Unknown'
users['Country'] = users['Country'].fillna('Unknown').replace('', 'Unknown')

users_cleaned = users[['User-ID', 'Location', 'Age', 'Age_Is_Missing', 'Age_Imputed', 'Age_Group', 'Country']]
print(f"Cleaned Users shape: {users_cleaned.shape}")

# --- Clean Books ---
print("Cleaning Books dataset...")
books = books_raw.copy()
books['ISBN'] = books['ISBN'].astype(str).str.strip().str.upper()
books = books.drop_duplicates('ISBN', keep='first').copy()

# Year-Of-Publication cleaning
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
valid_years = books.loc[(books['Year-Of-Publication'] >= 1800) & (books['Year-Of-Publication'] <= 2026), 'Year-Of-Publication']
median_year = valid_years.median() if len(valid_years) > 0 else 2000
books.loc[(books['Year-Of-Publication'] < 1800) | (books['Year-Of-Publication'] > 2026), 'Year-Of-Publication'] = median_year
books['Year-Of-Publication'] = books['Year-Of-Publication'].fillna(median_year).astype(int)

# String cleaning
for col in ['Book-Title', 'Book-Author', 'Publisher']:
    if col in books.columns:
        books[col] = books[col].fillna('Unknown').astype(str).str.strip()

books_cleaned = books.copy()
print(f"Cleaned Books shape: {books_cleaned.shape}")

# --- Clean Ratings / Interactions ---
print("Cleaning Ratings dataset...")
ratings = ratings_raw.copy()
ratings['User-ID'] = pd.to_numeric(ratings['User-ID'], errors='coerce').astype('Int64')
ratings['ISBN'] = ratings['ISBN'].astype(str).str.strip().str.upper()
ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')

ratings = ratings.dropna(subset=['User-ID', 'ISBN', 'Book-Rating']).copy()
ratings['User-ID'] = ratings['User-ID'].astype(int)
ratings['Book-Rating'] = ratings['Book-Rating'].astype(float)

# Filter interactions to known books/users
ratings = ratings.drop_duplicates(subset=['User-ID', 'ISBN'], keep='last').reset_index(drop=True)

# Explicit vs Implicit flag
ratings['Is_Explicit'] = (ratings['Book-Rating'] > 0).astype(int)
ratings['Has_Interaction'] = 1

# Calculate User-level features
print("Engineering User Features...")
explicit_ratings = ratings[ratings['Is_Explicit'] == 1]
user_agg = explicit_ratings.groupby('User-ID')['Book-Rating'].agg(
    User_Average_Rating='mean',
    User_Rating_Count='count',
    User_Rating_Std='std',
    User_Rating_Min='min',
    User_Rating_Max='max'
).reset_index()

user_imp = ratings[ratings['Is_Explicit'] == 0].groupby('User-ID').size().reset_index(name='User_Implicit_Count')

user_features = pd.merge(users_cleaned[['User-ID']], user_agg, on='User-ID', how='left')
user_features = pd.merge(user_features, user_imp, on='User-ID', how='left')

global_rating_mean = explicit_ratings['Book-Rating'].mean()
global_rating_std = explicit_ratings['Book-Rating'].std()

user_features['User_Average_Rating'] = user_features['User_Average_Rating'].fillna(global_rating_mean)
user_features['User_Rating_Count'] = user_features['User_Rating_Count'].fillna(0).astype(int)
user_features['User_Rating_Std'] = user_features['User_Rating_Std'].fillna(global_rating_std)
user_features['User_Rating_Min'] = user_features['User_Rating_Min'].fillna(0)
user_features['User_Rating_Max'] = user_features['User_Rating_Max'].fillna(0)
user_features['User_Implicit_Count'] = user_features['User_Implicit_Count'].fillna(0).astype(int)

# Calculate Book-level features
print("Engineering Book Features...")
book_agg = explicit_ratings.groupby('ISBN')['Book-Rating'].agg(
    Book_Average_Rating='mean',
    Book_Rating_Count='count'
).reset_index()

book_total_counts = ratings.groupby('ISBN').size().reset_index(name='Total_Interactions')

book_features = pd.merge(books_cleaned[['ISBN']], book_agg, on='ISBN', how='left')
book_features = pd.merge(book_features, book_total_counts, on='ISBN', how='left')

book_features['Book_Average_Rating'] = book_features['Book_Average_Rating'].fillna(global_rating_mean)
book_features['Book_Rating_Count'] = book_features['Book_Rating_Count'].fillna(0).astype(int)
book_features['Total_Interactions'] = book_features['Total_Interactions'].fillna(0).astype(int)
book_features['Log_Book_Popularity'] = np.log1p(book_features['Total_Interactions'])

# Calculate Interaction features
print("Engineering Interaction Features...")
interaction_features = pd.merge(ratings, user_features[['User-ID', 'User_Average_Rating']], on='User-ID', how='left')
interaction_features['Rating_Deviation'] = np.where(
    interaction_features['Is_Explicit'] == 1,
    interaction_features['Book-Rating'] - interaction_features['User_Average_Rating'],
    0.0
)
interaction_features = interaction_features[['User-ID', 'ISBN', 'Book-Rating', 'Is_Explicit', 'Rating_Deviation', 'Has_Interaction']]

# --- Save CSVs ---
print("Saving engineered CSV datasets...")
users_cleaned.to_csv(f"{OUTPUT_DIR}/users_cleaned_engineered.csv", index=False)
books_cleaned.to_csv(f"{OUTPUT_DIR}/books_cleaned_engineered.csv", index=False)
user_features.to_csv(f"{OUTPUT_DIR}/user_features.csv", index=False)
book_features.to_csv(f"{OUTPUT_DIR}/book_features.csv", index=False)
interaction_features.to_csv(f"{OUTPUT_DIR}/interaction_features.csv", index=False)

print("\n--- Phase 3 Completed Successfully ---")
print(f"Explicit Ratings Count: {(ratings['Is_Explicit'] == 1).sum():,}")
print(f"Implicit Ratings Count: {(ratings['Is_Explicit'] == 0).sum():,}")
print(f"Unique Users in Features: {len(user_features):,}")
print(f"Unique Books in Features: {len(book_features):,}")
