import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from crm import MovieRecommender
from llm import LLMRecommender

def prepare_data(save_path):
    path = kagglehub.dataset_download("zygmunt/goodbooks-10k")
    # load dataset
    ratings = pd.read_csv(os.path.join(path, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(path, 'movies.csv'))

    # drop duplicates
    ratings = ratings.drop_duplicates(subset=['user_id', 'movie_id'])

    # top 5000 users
    user_votes = ratings['user_id'].value_counts().head(5000).index

    # top 3000 books
    book_ratings = ratings['movie_id'].value_counts().head(3000).index

    # filter ratings
    ratings = ratings[ratings['user_id'].isin(user_votes) & ratings['movie_id'].isin(book_ratings)]

    # split data into trainging and testing sets
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    
    # convert test to binary
    test['rating'] = np.where(test['rating'] > 3, 1, 0)

    # save data as csv
    train.to_csv(os.path.join(save_path, 'ratings.csv'), index=False)
    test.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    movies.to_csv(os.path.join(save_path, 'movies.csv'), index=False)

def main():
    data_path = 'data/movie/'
    prepare_data(data_path)
    movies= pd.read_csv(data_path + 'movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title'])
    movie_id_to_title = dict(zip(movies['movie_id'], movies['title']))
    test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    movie_recommender = MovieRecommender('data/movie')
    llm_recommender = LLMRecommender()
    instructions=["Given the user's preference, identify whether the user will like the target movie by answering \"Yes.\" or \"No.\"."]
    inputs=["User Preference: <PREFERENCE>\nWhether the user will like the target movie <TARGET>?"]

    predictions = []
    for user in test:
        recommendations = movie_recommender.recommend(user['user_id'])
        inputs = inputs.replace('<PREFERENCE>', ' '.join(recommendations)).replace('<TARGET>', movie_id_to_title[user['moive_id']])
        ans = llm_recommender.recommend(instructions, inputs).split('\n')[0]
        predictions.append(1 if ans == 'Yes' else 0)

    #caculate auc
    auc = roc_auc_score(test['rating'], predictions)
    print(f"AUC: {auc}")


if __name__ == "__main__":
    main()