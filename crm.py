import ast
import pandas as pd
from surprise import SVD, Dataset, Reader

#Initial
#MovieRecommender(path)
#path: url of csv of movies.csv and ratings.csv

#Recommend
#xxx.recommend(user_id)
#return title of movies with rating above 3


class MovieRecommender:
    def __init__(self, path, predictions_path="predictions.csv"):
        self.predictions_path = predictions_path
        self.movies= pd.read_csv(path + 'movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title'])
        self.movie_id_to_title = dict(zip(self.movies['movie_id'], self.movies['title']))

        try:
            self.predictions = pd.read_csv(self.predictions_path)
            print("Predictions file exists. Loaded successfully!")
        except FileNotFoundError:
            print("Predictions file does not exist. Starting prediction...")
            self._train_and_save_predictions(path)

    def _train_and_save_predictions(self, path):
        ratings = pd.read_csv(path + 'ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()

        model = SVD(n_factors=80, n_epochs=40, biased=True, lr_all=0.01, reg_all=0.05)
        model.fit(trainset)

        user_ids = [trainset.to_raw_uid(uid) for uid in trainset.all_users()]
        movie_ids = [trainset.to_raw_iid(iid) for iid in trainset.all_items()]

        predictions = {}
        for user_id in user_ids:
            liked_movies = []
            for movie_id in movie_ids:
                predicted_rating = round(model.predict(user_id, movie_id).est)  
                if predicted_rating > 3:  
                    liked_movies.append(movie_id)
            predictions[user_id] = liked_movies

        self.predictions = pd.DataFrame(list(predictions.items()), columns=['user_id', 'movie_ids'])
        self.predictions.to_csv(self.predictions_path, index=False)
        print("Predictions saved!")
        self.predictions = pd.read_csv(self.predictions_path)

    def recommend(self, user_id):
        user_predictions = self.predictions[self.predictions['user_id'] == user_id]
        if user_predictions.empty:
            return f"User {user_id} does not exist!"
        movie_ids = ast.literal_eval(user_predictions.iloc[0]['movie_ids'])

        movie_titles = [self.movie_id_to_title[movie_id] for movie_id in movie_ids if movie_id in self.movie_id_to_title]
        return movie_titles
