# Import packages
import os
import pandas as pd


#Use:
#process_and_save_data(MOVIELENS_DIR)
#MOVIELENS_DIR:url of dat files

#MOVIELENS_DIR = 'dat'
USER_DATA_FILE = 'users.dat'
MOVIE_DATA_FILE = 'movies.dat'
RATING_DATA_FILE = 'ratings.dat'

# Specify User's Age and Occupation Column
AGES = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
OCCUPATIONS = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }

def process_and_save_data(MOVIELENS_DIR,ratings_file='ratings.csv', users_file='users.csv', movies_file='movies.csv'):
    """Process and save ratings, users, and movies data into CSV files."""
    # Process Ratings
    ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE),
                        sep='::',
                        engine='python',
                        encoding='latin-1',
                        names=['user_id', 'movie_id', 'rating', 'timestamp'])

    max_userid = ratings['user_id'].drop_duplicates().max()
    max_movieid = ratings['movie_id'].drop_duplicates().max()

    ratings['user_emb_id'] = ratings['user_id'] - 1
    ratings['movie_emb_id'] = ratings['movie_id'] - 1

    print(len(ratings), 'ratings loaded')
    ratings.to_csv(ratings_file,
                   sep='\t',
                   header=True,
                   encoding='latin-1',
                   columns=['user_id', 'movie_id', 'rating', 'timestamp', 'user_emb_id', 'movie_emb_id'])
    print('Ratings saved to', ratings_file)

    # Process Users
    users = pd.read_csv(os.path.join(MOVIELENS_DIR, USER_DATA_FILE),
                        sep='::',
                        engine='python',
                        encoding='latin-1',
                        names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
    users['age_desc'] = users['age'].apply(lambda x: AGES[x])
    users['occ_desc'] = users['occupation'].apply(lambda x: OCCUPATIONS[x])

    print(len(users), 'descriptions of', max_userid, 'users loaded.')
    users.to_csv(users_file,
                 sep='\t',
                 header=True,
                 encoding='latin-1',
                 columns=['user_id', 'gender', 'age', 'occupation', 'zipcode', 'age_desc', 'occ_desc'])
    print('Users saved to', users_file)

    # Process Movies
    movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE),
                        sep='::',
                        engine='python',
                        encoding='latin-1',
                        names=['movie_id', 'title', 'genres'])

    print(len(movies), 'descriptions of', max_movieid, 'movies loaded.')
    movies.to_csv(movies_file,
                  sep='\t',
                  header=True,
                  columns=['movie_id', 'title', 'genres'])
    print('Movies saved to', movies_file)

if __name__ == '__main__':
    process_and_save_data(ratings_file='ratings.csv', users_file='users.csv', movies_file='movies.csv')
