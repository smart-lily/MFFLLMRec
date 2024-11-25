import os
import pandas as pd
from openai import OpenAI
from func import get_user_watch_history
from sklearn.model_selection import train_test_split

client = OpenAI()

data = pd.read_csv("data/movies.csv")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
system_prompt = "You are a movie recommender system that will compare user previous watch history and ratings."
unique_movie_titles = set(data['title'].unique().tolist())

user_id = 123
candidate_size = 100

train_user_df = train_data[train_data["userId"] == user_id]
test_user_df = test_data[test_data["userId"] == user_id]

train_title, train_rating = train_user_df["title"], train_user_df["rating"]
test_title, test_rating = test_user_df["title"], test_user_df["rating"]

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def get_similar_users(user_id, data):
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
    user_movie_matrix = user_movie_matrix.fillna(0)
    similarity_matrix = cosine_similarity(user_movie_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    similar_users = similarity_df[user_id].sort_values(ascending=False)
    return similar_users

similar_users = list(get_similar_users(user_id, train_data).iloc[:candidate_size].index)
train_similar_df = train_data[train_data["userId"].isin(similar_users)]
movie_popularity = train_similar_df.groupby('title').size().sort_values(ascending=False)
candidate1 = list(movie_popularity.head(candidate_size).head(candidate_size).index)

train_movie = train_user_df["movieId"]
watched = pd.unique(train_movie).tolist()
user_item_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')
user_item_matrix = user_item_matrix.fillna(0)
item_similarity = cosine_similarity(user_item_matrix.T)


def find_similar_movies(target_item_id, data):
    target_item_index = user_item_matrix.columns.get_loc(target_item_id)
    similarities = item_similarity[target_item_index]
    similar_items_df = pd.DataFrame({'movieId': user_item_matrix.columns, 'similarity_score': similarities})
    similar_items_df = similar_items_df.sort_values(by='similarity_score', ascending=False)
    N = candidate_size
    top_similar_items = similar_items_df.head(N)
    return top_similar_items

similar_movies = []
for movie in watched:
    similar_movies.append(find_similar_movies(movie, train_data))
similar_df = pd.concat(similar_movies)

movie_popularity = similar_df.groupby('movieId').size().sort_values(ascending=False)
candidate2 = list(movie_popularity.head(candidate_size).head(candidate_size).index)
candidate2 = train_data.loc[train_data['movieId'].isin(candidate2), 'title'].tolist()
candidate = list(set(candidate1) | set(candidate2))

def clean_candidate(candidate, train_title, test_title):
    to_remove = []

    for movie in candidate:
        if movie in train_title:
            ro_remove.append(movie)
            print(f"to remove: {movie} ")
    count1 = 0
    count2 = 0
    for movie in to_remove:
        candidate.append(movie)
    for movie in test_title:
        if movie not in candidate:
            count1 += 1
        else:
            count2 += 1
    print(count1, count2)

    return candidate


candidate = clean_candidate(candidate, train_title, test_title)
print(f"The length of the training set: {len(train_title)}")
print(f"The length of the testing set: {len(test_title)}")
print(f"The length of the candidate set: {len(candidate)}")

import numpy as np
import random
movie_df = list(pd.unique(data["title"]))

movie_rating = ""
for i in range(len(train_title)):
    movie_rating += f"{train_title.iloc[i]}: {train_rating.iloc[i]} \n"

if user_id in data['userId'].values:
    titles_list, ratings_list = get_user_watch_history(user_id, data)
    messages=[
        {"role": "user", "content": f"The movies I have watched(watched movies): {movie_rating}"},
        {"role": "user", "content": f"Step 1: What features are most important to me when selecting movies? "},
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )

answer1 = (completion.choices[0].message.content)
answer1

messages = [
    {"role": "user", "content": f"The movies I have watched(watched movies) and their ratings: {movie_rating}"},
    {"role": "user", "content": f"Step 1: What features are most important to me when selecting movies? "},
]

messages.append({"role": "assistant", "content": answer1})

step2 = "You will select the movies (at most 10 movies) that appeal to me the most from the list of movies \
    I have watched, based on my personal preferences. The selected movies will be presented in descending \
    order of preference. (Format: no. a watched movie)."

messages.append({"role": "user", "content": step2})

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages
)

answer2 = (completion.choices[0].message.content)
print(answer2)

messages.append({"role": "assistant", "content": answer2})

step3 = "Can you recommend 10 different movies only from the Candidate Set similar to the selected \
    movies I've watched (Format: [<n>. <a watched movie> : <a candidate movie>])?" + f"Candidate Set\
    (candidate movies): {', '.join(candidate)}"

messages.append({"role": "user", "content": step3})

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages
)

answer3 = (completion.choices[0].message.content)
print(answer3)

import re
def parse_answer3(answer3):
    lines = answer3.split('\n')
    pattern = r': (.*?) \((\d+)\)'
    movie_pred = []
    for line in lines:
        match = re.search(pattern, line)
        if match:
            title = match.group(1)
            year = match.group(2)
            movie_pred.append((title, year))
    return movie_pred

def accuracy(movie_pred, test_title):
    correct = 0
    test_title_list = list(test_title)
    for title, year in movie_pred:
        if "The" in title:
            title = title[4:]
        find_candidate = 0
        for movie in candidate:
            if title in movie:
                find_candidate = 1
                break
        if not find_candidate:
            pass
        for movie in test_title_list:
            if title in movie:
                correct += 1
                break
    return correct / len(movie_pred)

def accuracy_baseline(movie_pred, test_title):
    correct = 0
    test_title_list = list(test_title)
    for movie in movie_pred:
        if movie in test_title_list:
            correct += 1
            break
    return correct / len(movie_pred)

movie_popularity = train_data.groupby('title').size().sort_values(ascending=False)
baseline_pred = []
for movie in movie_popularity.index:
    if movie in candidate:
        baseline_pred.append(movie)
    if len(baseline_pred) >= 10:
        break
accuracy_baseline(baseline_pred, test_title)
print(accuracy_baseline(baseline_pred, test_title))