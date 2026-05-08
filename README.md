# Exploratory Data Analysis

# Step by step reasoning

## 0. Item to user matrix

## Objective

First, we want to transform the raw data into a matrix that can be used by recommender systems. The original data is a table of user-book interactions. This is converted into a **user-item matrix**, where rows are users and columns are movies.

## Method

We use implicit feedback. The model only considers whether a user had a positive interaction with a movie.

$$
R_{u,i} = 1
$$

This means that user $u$ liked movie $i$.

If:

$$
R_{u,i} = 0
$$

then no positive interaction is observed.

## Thought process

This task prepares the data in the correct format. Collaborative filtering methods need a numerical matrix showing which users interacted with which items. This matrix is the basis for computing similarities and making recommendations.

## 1. Item-to-item CF

## Objective

The goal is to recommend movies based on item similarity.

The intuition is:

If a user liked a movie, they may also like movies that are similar to it.

## Method

Each movie is represented by the users who liked it. To do this, the user-item matrix is transposed, so that movies become the rows.

Then, cosine similarity is used to compare movies. Two movies are considered similar if they were liked by similar groups of users.

This gives an item-item similarity matrix.

The model then predicts a score for each user-movie pair using a weighted average:

$$
\widehat{R}_{u,i} = \frac{\sum_j sim(i,j)R_{u,j}}{\sum_j sim(i,j)}
$$

This means that the score for movie $i$ depends on how similar it is to the movies $j$ already liked by user $u$.

A high score means that the movie is similar to movies the user liked before.

## Thought process

Item-based collaborative filtering learns relationships between movies from user behavior. It does not use movie descriptions or genres. Instead, it assumes that if the same users liked two movies, these movies are probably related in terms of preference.

A limitation is that similarity can be unreliable for rare movies, because two items with very few interactions may appear artificially similar.

## 2. User-to-user CF

## Objective

The goal is to recommend movies based on user similarity.

The intuition is:

If two users liked similar movies in the past, they may like similar movies in the future.

## Method

Each user is represented by the movies they liked. Cosine similarity is then used to compare users.

Two users are considered similar if they liked many of the same movies.

This gives a user-user similarity matrix.

The predicted score is computed as:

$$
\widehat{R}_{u,i} = \frac{\sum_v sim(u,v)R_{v,i}}{\sum_v sim(u,v)}
$$

This means that the score for movie $i$ depends on whether users similar to $u$ liked that movie.

A high score means that many similar users liked the movie.

## Thought process

User-based collaborative filtering uses the preferences of similar users to make recommendations. Instead of asking which movies are similar, it asks which users are similar, and then recommends what those users liked.

A limitation is that user similarities can be noisy when users have few interactions. It can also become expensive when the number of users is large.

# Comparison

Both methods use collaborative filtering and only rely on user-item interactions.

| **Method** | **Compares** | **Recommendation idea** |
|---|---|---|
| Item-based CF | Movies | Recommend movies similar to those already liked |
| User-based CF | Users | Recommend movies liked by similar users |

Overall, we first build the interaction matrix, then uses it in two ways: comparing items and comparing users. Both approaches rely on the idea that past interaction patterns can help predict future preferences.

# Performance table

|  | User-to-user | Item-to-item | Alternative technique 1 | Alternative technique 2 |
|---|---|---|---|---|
| Precision@10 |  |  |  |  |
| Recall@10 |  |  |  |  |

# Discussion

# Link to YouTube video
