import pandas as pd
import numpy as np
import kagglehub
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from crm import MovieRecommender
from llm import LLMRecommender

def prepare_data(save_path):
    path = kagglehub.dataset_download("snehal1409/movielens")
    # load dataset
    ratings = pd.read_csv(os.path.join(path, 'ratings.csv'), usecols=['userId', 'movieId', 'rating'])
    movies= pd.read_csv(os.path.join(path, 'movies.csv'), usecols=['movieId', 'title'])

    # change the arttribute names
    ratings.columns = ['user_id', 'movie_id', 'rating']
    movies.columns = ['movie_id', 'title']

    # sample 100 users
    ratings = ratings[ratings['user_id'].isin(ratings['user_id'].sample(100, random_state=42))]
    movies = movies[movies['movie_id'].isin(ratings['movie_id'])]

    # split data into trainging and testing sets
    train, test = train_test_split(ratings, test_size=0.1, random_state=42)

    # make sure all movies in test set are in training set
    test = test[test['movie_id'].isin(train['movie_id'])]
    
    # convert test to binary
    test['rating'] = np.where(test['rating'] > 3, 1, 0)

    # save data as csv
    train.to_csv(os.path.join(save_path, 'ratings.csv'), index=False)
    test.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    movies.to_csv(os.path.join(save_path, 'movies.csv'), index=False)

def main():
    data_path = './data/movie/'
    prepare_data(data_path)
    movies= pd.read_csv(data_path + 'movies.csv')
    movie_id_to_title = dict(zip(movies['movie_id'], movies['title']))
    test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    movie_recommender = MovieRecommender(data_path)
    llm_recommender = LLMRecommender()
    instructions=["Given the user's preference, identify whether the user will like the target movie by answering \"Yes.\" or \"No.\"."]
    inputs=["User Preference: <PREFERENCE>\nWhether the user will like the target movie <TARGET>?"]

    predictions = []
    for index, row in tqdm(test.iterrows(), total=len(test)):
        recommendations = movie_recommender.recommend(row['user_id'])
        inputs = [input.replace('<PREFERENCE>', ' '.join(recommendations)).replace('<TARGET>', movie_id_to_title[row['movieId']])for input in inputs]
        ans, logits = llm_recommender.recommend(instructions, inputs).split('\n')[0]
        print(ans)
        predictions.append(logits[0])

    #caculate auc
    auc = roc_auc_score(test['rating'], predictions)
    print(f"AUC: {auc}")


if __name__ == "__main__":
    main()