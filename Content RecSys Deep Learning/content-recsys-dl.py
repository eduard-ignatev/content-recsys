import os
import pandas as pd

from dotenv import load_dotenv
from typing import List
from datetime import datetime
from catboost import CatBoostClassifier

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
from pydantic import BaseModel
from loguru import logger


# Database connection settings
load_dotenv()
DATABASE_URI = os.environ['DATABASE_URI']


# PostGet response model
class PostGet(BaseModel):
    id: int
    text: str
    topic: str
    class Config:
        orm_mode = True


# FastAPI initialization
app = FastAPI()


def batch_load_sql(query: str):
    # Loading SQL data into Pandas without running out of memory
    engine = create_engine(DATABASE_URI)
    conn = engine.connect().execution_options(stream_results=True)

    chunks_arr = []
    for chunk in pd.read_sql(query, conn, chunksize=500000):
        chunks_arr.append(chunk)
        logger.info('got chunk 500000...')
    conn.close()
    return pd.concat(chunks_arr, ignore_index=True)


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_posts():
    # Load all posts
    logger.info('loading posts')
    df_posts = pd.read_sql(
        '''
        SELECT *
        FROM public.post_text_df
        ''',
        DATABASE_URI
    )
    df_posts = df_posts.set_index('post_id')
    return df_posts


def load_posts_features():
    # Load text clustering features
    logger.info('loading posts features')
    df_posts_features = pd.read_sql(
        '''
        SELECT *
        FROM public.ignatev_df_posts_clustering_roberta
        ''',
        DATABASE_URI
    )
    df_posts_features = df_posts_features.set_index('post_id')
    return df_posts_features


def load_users():
    # Load all users
    logger.info('loading users')
    df_users = pd.read_sql(
        '''
        SELECT *
        FROM public.user_data
        ''',
        DATABASE_URI
    )
    return df_users


def load_users_features():
    # Load total likes per user
    logger.info('loading users features')
    df_likes = pd.read_sql(
        '''
        SELECT user_id, COUNT(post_id) AS user_total_likes
        FROM public.feed_data
        WHERE timestamp < '2021-12-15 00:00:00' AND target = 1
        GROUP BY user_id
        ''',
        DATABASE_URI
    )
    df_users_features = pd.merge(
        df_users,
        df_likes,
        on='user_id',
        how='left'
    )
    df_users_features['user_total_likes'] = df_users_features['user_total_likes'].fillna(0).astype(int)
    df_users_features = df_users_features.set_index('user_id')
    return df_users_features


def load_likes():
    # Loading user_id-post_id like pairs
    logger.info('loading likes')
    likes_query = '''
        SELECT DISTINCT user_id, post_id
        FROM public.feed_data
        WHERE action = 'like'
        '''
    df_likes = batch_load_sql(likes_query)
    return df_likes


def load_models():
    # Loading Catboost pretrained model
    logger.info('loading models')
    model_path = get_model_path("/home/edd-ign/Code/Projects/Content RecSys Deep Learning/catboost_model_02_roberta")
    model = CatBoostClassifier()
    model.load_model(fname=model_path)
    return model


# Load all posts, additional features and models on service startup
df_posts = load_posts()
df_posts_features = load_posts_features()
df_users = load_users()
df_users_features = load_users_features()
df_likes = load_likes()
model = load_models()
logger.info('service is up and running')


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    # Check if input user id is in database
    if id not in df_users_features.index:
        raise HTTPException(status_code=404, detail="User ID not found")

    # Load features of specific user
    user_info = df_users_features.loc[id]
    
    # Put together the final dataframe for prediction
    df_predict = df_posts_features # taking posts with features as a basis
    df_predict = df_predict.assign(**user_info) # adding user info
    df_predict['month'] = time.month
    df_predict['day_of_week'] = time.weekday()
    df_predict['hour'] = time.hour
    df_predict = df_predict[[
        'gender', 'age', 'country', 'city', 'exp_group', 'os', 'source', 'month', 'day_of_week', 'hour', 'topic', 
        'TextCluster', 'DistanceTo1thCluster', 'DistanceTo2thCluster', 'DistanceTo3thCluster', 'DistanceTo4thCluster',
        'DistanceTo5thCluster', 'DistanceTo6thCluster', 'DistanceTo7thCluster', 'DistanceTo8thCluster', 'DistanceTo9thCluster',
        'DistanceTo10thCluster', 'DistanceTo11thCluster', 'DistanceTo12thCluster', 'DistanceTo13thCluster', 'DistanceTo14thCluster',
        'DistanceTo15thCluster', 'user_total_likes']]

    # Predict probability of like
    df_predict['like_probability'] = model.predict_proba(df_predict)[:, 1]

    # Remove already liked posts from recommendations and rank them
    already_liked = df_likes[df_likes.user_id == id].post_id.values
    rec_posts = df_predict[~df_predict.index.isin(already_liked)]
    rec_posts = rec_posts.sort_values('like_probability')[-limit:].index

    # Return appropriate response data
    return [
        PostGet(**{
            'id': i,
            'text': df_posts.loc[i].text,
            'topic': df_posts.loc[i].topic,
        }) for i in rec_posts
    ]
