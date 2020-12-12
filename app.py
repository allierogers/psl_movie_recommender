import boto3
import json
import numpy as np
import os
import pandas as pd

from collections import defaultdict
from flask import Flask, request, redirect, url_for, jsonify, render_template
from surprise import dump, Reader, Dataset, SVD


app = Flask(__name__)


base_url = 'https://liangfgithub.github.io/MovieData/'
RATINGS = pd.read_csv(base_url + 'ratings.dat?raw=true', 
                      sep='::',
                      engine='python',
                      nrows=12975,
                      header=None,
                      names=['user_id', 'movie_id', 'rating', 'ts'])
MOVIES = pd.read_csv(base_url + 'movies.dat?raw=true', 
                      sep='::',
                      engine='python',
                      header=None,
                      names=['movie_id', 'movie_name', 'movie_genres'],
                      encoding='latin-1')


def get_collab_recs(ratings):
    # returns a dictionary of top 10 ranked results
    usable_ratings = [v for v in ratings.values() if v != 0]
    rated_item_ids = [k for k in ratings.keys() if ratings[k] != 0]
    uids = [9999 for x in range(len(rated_item_ids))]

    ratings_dict = {'item_id': list(RATINGS['movie_id']) + rated_item_ids,
                    'user_id': list(RATINGS['user_id']) + uids,
                    'rating': list(RATINGS['rating']) + usable_ratings}
    df = pd.DataFrame(ratings_dict)

    # TRAIN
    reader = Reader(rating_scale=(1.0, 5.0))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    # PREDICT
    all_movies = df['item_id'].unique()

    preds = {}
    for iid in all_movies:
        uid, this_iid, true_r, est, _ = algo.predict(9999, iid, verbose=False)
        preds[iid] = est

    # Then sort the predictions for each user and retrieve the k highest ones.
    top_movie_ids = sorted(preds, key=preds.get, reverse=True)[:10]
    movie_id_to_name = dict(zip(MOVIES['movie_id'], MOVIES['movie_name']))
    movie_names = [movie_id_to_name[mid] for mid in top_movie_ids]
    results = dict(zip(list(range(10)), movie_names))

    return results


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/genre', methods=['GET', 'POST'])
def genre_recs():
    if request.method == 'POST':
        selected_genre = request.form['selected_genre']
        
        s3 = boto3.resource('s3')
        s3object = s3.Object('allie-random', 'recs_by_genre.txt')
        recs_str = s3object.get()['Body'].read().decode('UTF-8')
        recs_by_genre = json.loads(recs_str) 
        results = recs_by_genre[selected_genre]['highest_rated']['movie_name']
        return render_template('genre_recs.html', result=results)

    return render_template('genre_page.html')


@app.route('/collab', methods=['GET', 'POST'])
def collab_filtering():
    movies = pd.read_csv('https://liangfgithub.github.io/MovieData/movies.dat?raw=true', 
                      sep='::',
                      engine='python',
                      header=None,
                      names=['movie_id', 'movie_name', 'movie_genres'],
                      encoding='latin-1')
    movies['movie_name'] = movies['movie_name'].apply(lambda x: x[:-6])
    movie_subset = movies.head(200)
    
    if request.method == 'POST':
        ratings = {}
        ratings[1] = int(request.form['1'])
        ratings[2] = int(request.form['2'])
        ratings[3] = int(request.form['3'])
        ratings[4] = int(request.form['4'])
        ratings[5] = int(request.form['5'])
        ratings[6] = int(request.form['6'])
        ratings[7] = int(request.form['7'])
        ratings[8] = int(request.form['8'])
        ratings[9] = int(request.form['9'])
        ratings[10] = int(request.form['10'])
        ratings[11] = int(request.form['11'])
        ratings[12] = int(request.form['12'])
        ratings[13] = int(request.form['13'])
        ratings[14] = int(request.form['14'])
        ratings[15] = int(request.form['15'])
        ratings[16] = int(request.form['16'])
        ratings[17] = int(request.form['17'])
        ratings[18] = int(request.form['18'])
        ratings[19] = int(request.form['19'])
        ratings[20] = int(request.form['20'])

        results = get_collab_recs(ratings)

        return render_template('collab_recs.html', result=results)
    
    return render_template('ratings_page.html', movies=list(movie_subset['movie_name']))


if __name__ == "__main__":
    app.run(debug=True)
