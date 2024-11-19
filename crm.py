import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader

#Initial
#MovieRecommender(path)
#path: url of csv of movies.csv and ratings.csv

#Recommend
#xxx.recommend(user_id)
#return title of movies with rating above 3

#attention:
#numoy version may need to be under 2

class MovieRecommender:
    def __init__(self, path):
        """
        :param path: url of csv
        """
        self.ratings = pd.read_csv(path+'ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])
        self.movies = pd.read_csv(path+'movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title'])

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings[['user_id', 'movie_id', 'rating']], reader)

        self.trainset = data.build_full_trainset()
        self.model = SVD(n_factors=80, n_epochs=40, biased=True, lr_all=0.01, reg_all=0.05)
        self.model.fit(self.trainset)

        self.movie_id_to_title = dict(zip(self.movies['movie_id'], self.movies['title']))

    def recommend(self, user_id, threshold=3):
        """
        :param threshold: movies with rating above threshold will be recommended
        :return: movie title
        """
        if user_id not in self.ratings['user_id'].unique():
            return f"User {user_id} do not exist！"

        all_movie_ids = self.movies['movie_id'].unique()

        recommendations = []
        for movie_id in all_movie_ids:
            predicted_rating = self.model.predict(user_id, movie_id).est
            rounded_rating = round(predicted_rating)
            if rounded_rating > threshold:
                recommendations.append((movie_id, rounded_rating))

        recommendations.sort(key=lambda x: x[0])

        recommended_titles = [self.movie_id_to_title[movie_id] for movie_id, _ in recommendations]
        return recommended_titles
