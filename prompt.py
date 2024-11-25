import os
import pandas as pd
from openai import OpenAI
from func import get_user_watch_history
from sklearn.model_selection import train_test_split
from src.func import *
import re

client = OpenAI()

merged_df = pd.read_csv("data/movies.csv")
unique_movie_titles = merged_df['title'].unique().tolist()
data = merged_df
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
system_prompt = "You are a movie recommender system that will compare user previous watch history and ratings."

USER_ID = 23
USER_ID2 = 445

user_id = USER_ID
if user_id in data['userId'].values:
    train_data_user = train_data[train_data["userId"] == user_id]
    titles_list, ratings_list = train_data_user["title"], train_data_user["rating"]
    messages=[
        {"role": "system", "content": f"{system_prompt} Based on the genre and ratings of \
            those movies that this particular user gave, please recommend 10 movie that this \
            user would enjoy watching and predict the rating of these 10 movies given by this user"},
        {"role": "user", "content": f"User {user_id}'s previously watch movies include {titles_list}\
            and the corresponded ratings are {ratings_list}. Please suggest 10 movie based on the \
            watch history and predict ratings of those 10 movies based on user preferences. \
            Format: [n. <Movie Name> (<Year>) - Predicted Rating: <Rating>]"}
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    response = completion.choices[0].message.content
    recommendations1 = extract_string1(response)
    print(response)
else:
    print(f"{user_id} is not a valid user ID, please try again")

userA, userB = USER_ID, USER_ID2
if userA and userB in data['userId'].values:
    train_data_userA = train_data[train_data["userId"] == userA]
    titles_list_A, ratings_list_A = train_data_userA["title"], train_data_userA["rating"]
    train_data_userB = train_data[train_data["userId"] == userB]
    titles_list_B, ratings_list_B = train_data_userB["title"], train_data_userB["rating"]
    messages=[
        {"role": "system", "content": f"{system_prompt} Based on the genre and ratings of movies from \
            2 users, please recommend 10 movie that these 2 users would enjoy watching together and \
            predict the rating of these 10 movies"},
        {"role": "user", "content": f"User {userA}'s previously watch movies include {titles_list_A} \
            and the corresponded ratings are {ratings_list_A}. User {userB}'s previously watch movies \
            include {titles_list_B} and the corresponded ratings are {ratings_list_B} Please suggest \
            10 movie that both users would enjoy watching based on the watch histories and predict \
            specific ratings of those 10 movies based on user preferences. Format: \
            [n. <Movie Name> (<Year>) - Predicted Rating: <Rating>]"}
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    response = completion.choices[0].message.content
    print(response)
    recommendations2 = extract_string1(response)
else:
    print(f"{userA} or {userB} is not a valid user ID, please try again")

if user_id in data['userId'].values:
    user_data = train_data[train_data["userId"] == user_id]
    user_data['genres'] = user_data['genres'].str.split('|')
    genres_expanded = user_data.explode('genres')
    genre_ratings = genres_expanded.groupby('genres')['rating'].mean()
    top_3_genres = genre_ratings.sort_values(ascending=False).head(3).index.tolist()
    messages=[
        {"role": "system", "content": "You are a movie recommender system that will recommend \
            10 movie in this user's favorite genres and predict the rating of these 10 movies \
            given by this user"},
        {"role": "user", "content": f"User {user_id}'s favorite genre include {top_3_genres}. \
            Please suggest 10 movie in these genre for the user. \
            Format: [n. <Movie Name> (<Year>) - Predicted Rating: <Rating>]"}
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    response = completion.choices[0].message.content
    recommendations3 = extract_string1(response)
    print(response)
else:
    print(f"{userId} is not a valid user ID, please try again")

if user_id in data['userId'].values:
    most_similar = get_similar_users(user_id, train_data)
    train_data_user = train_data[train_data["userId"] == most_similar]
    titles_list, ratings_list = train_data_user["title"], train_data_user["rating"]
    messages=[
        {"role": "system", "content": "You are a movie recommender system that will suggest movies \
            that the user may also like based on similar user's watch histories. From the most similar \
            user's watch histories, please recommend 10 movies that this user would enjoy watching and \
            predict the rating of these 10 movies given by this user"},
        {"role": "user", "content": f"User {user_id}'s most similar user is user {most_similar}, \
            which has previously watched {titles_list} and the corresponded ratings are {ratings_list}. \
            Please suggest 10 movies that user {user_id} may also like and provide rating prediction of \
            those 10 movies based on user preferences. \
            Format: [n. <Movie Name> (<Year>) - Predicted Rating: <Rating>]"}
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    response = completion.choices[0].message.content
    print(response)
    recommendations4 = extract_string1(response)
else:
    print(f"{userId} is not a valid user ID, please try again")

if user_id in data['userId'].values:
    train_data_user = train_data[train_data["userId"] == user_id]
    train_titles_list, train_ratings_list = train_data_user["title"], train_data_user["rating"]
    test_data_user = test_data[test_data["userId"] == user_id]
    test_title_list, true_ratings = test_data_user["title"], test_data_user["rating"]
    messages=[
        {"role": "system", "content": "You are a movie rating prediction system that will predict \
            rating with a list of titles given"},
        {"role": "user", "content": f"User {user_id}'s previously watch movies include {train_titles_list} \
            and the corresponded ratings are {train_ratings_list}. Based on these ratings, \
            please provide a list of prediction rating corresponded to {test_title_list}. \
            Format: [n. <Movie Name> (<Year>) - Predicted Rating: <Rating>]"}
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    response = completion.choices[0].message.content
    print(response)
    recommendations5 = extract_string1(response)
else:
    print(f"{userId} is not a valid user ID, please try again")

from func import extract_ratings
predicted_ratings = []
for movie, rating in recommendations5:
    predicted_ratings.append(float(rating))