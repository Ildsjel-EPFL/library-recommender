# Exploratory Data Analysis
In the context of building these recommendation systems, the Exploratory Data Analysis (EDA) should focus on quantifying data sparsity, evaluating metadata quality, and visualizing interaction patterns to inform model selection.

## Interaction Data Profiling

### Unique Counts
We calculated the total number of unique users, unique items, and overall interactions statistics to establish the basic scale of the dataset.

| # users | # total books | # unique books in interaction | avg books read per user| min # books read by a specific user | max # books read by a specific user | percentiles in the repartition of # books read by a specific user | std |
|---|---|---|---|---|---|---|---|
|7838|15291|15109|11.11|3|385|25% : 3, 50% : 6, 75% : 11|16.44|

### Sparsity Analysis
We can see that some books in the `items.csv` file do not appear in the `interaction.csv` file. We can also observe that the data is extremely sparse, and we have a heavily skewed distribution.

Also, if we plot the user-item interaction matrix, we can clearly see this sparcity : 

<img src=data/user_item_mtx.png style="width: 500px; height: 500px">

## Item Metadata & Content Evaluation
### Feature Availability:
We checked for the presence and completeness of fields such as Title, Author, Subjects, Genres, Summary, Page Count, and Average User Rating using OpenLibrary API and secondly GoogleBooks API.

With some of them being really reare (e,g. Average User Rating), we could not use all of them. For the Page Count, we imputed the missing values by taking the median value and then proceed to a z-normalization.

## Relational & Dimensionality Analysis
### Similarity Visualization: 
By plotting the item-item and user-user similarity heatmaps using cosine similarity, we see that only rare items naturally cluster are formed based on user behavior.
| user-user | item-item |
|---|---|
|<img src=data/user_user_mtx.png style="width: 500px; height: 500px">|<img src=data/item_item_mtx.png style="width: 500px; height: 500px">|

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

# 1. Item-to-item CF

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

# 2. User-to-user CF

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

# Hybrid Neural Collaborative Filtering (NCF) and Hybrid Matrix-Blending models
## Objective
The primary goal is to build a robust recommendation system capable of predicting the top 10 books a user is most likely to interact with. While traditional baselines rely solely on interaction history, these models aim to leverage "hybrid" signals ; combining user behavioral patterns with item-specific metadata like titles, authors, and summaries. This multi-faceted approach is designed to increase recommendation accuracy and better handle diverse user interests.

## Description of the models :
### NCF
- **Architecture:** This model utilizes a deep learning framework built in PyTorch, featuring an embedding layer for users and items followed by a Multi-Layer Perceptron (MLP).

- **Input Features:** The MLP processes a concatenated vector of user/item embeddings, dense text embeddings, and scaled numerical data such as page counts.

- **Training Protocol:** The model uses Negative Sampling, where four unread books are randomly selected for every actual interaction to teach the model to distinguish between positive and negative signals.

- **Optimization:** Hyperparameters (embedding dimensions, learning rate, and dropout) are tuned using the Optuna framework to minimize validation loss.
### Hybrid Matrix-Blending
- **Similarity Computation:** The model calculates two separate item-item similarity matrices using `cosine_similarity`: one based on the user interaction matrix (Collaborative Filtering) and one based on text embeddings (Content-Based).
- **Feature Extraction:** Metadata is encoded into high-dimensional vectors using Sentence Transformers, specifically the `multilingual-e5-base model`.
- **Matrix Blending:** A hybrid similarity matrix is created by blending the two perspectives using a weighted parameter, $\alpha$, which is found using a cross-validated grid-search.

### Tought process
The development process was driven by several strategic considerations:
- **Solving the Cold Start Problem:** By including text embeddings, the models can recommend books based on semantic similarity to a user's previous reads even if those books have few interactions.
- **Capturing Non-Linearity:** The NCF model was introduced to capture complex relationships between user traits and book characteristics that simple similarity scores might overlook.
- **Data Integrity & Leakage:** To ensure realistic performance metrics, a Temporal Split was utilized, training models on the first 80% of interactions and validating on the most recent 20%.
- **Balancing Efficiency and Power:** The Hybrid Matrix model serves as a high-speed alternative to NCF, allowing for rapid iteration via linear algebra rather than intensive iterative training.

### Technical details.
For both model, we tried multiple Embeddings models, such as `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-base` from infloat and the state of the art `embeddinggemma-300m` by Google DeepMind. In fine, the e5 model was the one to give the best results, but really close to the other ones.

### Failure Analysis of the NCF Model
Despite its complexity, the Neural Collaborative Filtering model failed to provide meaningful recommendations for several critical reasons:
- **Extreme Sparsity:** The dataset is extremely sparse, with users having a median of only 6 interactions.
- **Insufficient Data for Training:** A neural network requires thousands of examples to properly update the weights in its `nn.Embedding` layers. Because of this sparsity, the network was essentially guessing.
- **The Behavioral Link Gap:** While memory-based Collaborative Filtering can instantly link users who share even 2 books , a neural network struggles to establish that connection with only 6 data points.
- **Feature Imbalance:** The model suffered from "Drowning in Text," where the size-64 user/item embeddings were overwhelmed by the size-768 Hugging Face text embeddings during concatenation. Consequently, the network stopped performing collaborative filtering and recommended books based purely on text similarity.
- **Optimization Shortcuts:** Training with Binary Cross Entropy (BCE) loss allowed the network to learn a "lazy shortcut". Since users only read a tiny fraction of the 15,000 available books, the model achieved 99.9% accuracy simply by predicting 0 (not read) for everything.
- **Ranking Misalignment:** While the model optimized for absolute 1s and 0s, the evaluation metric (MAP@10) requires effective ranking, which the model failed to prioritize.

# Comparison

The key differences between the hybrid models and traditional Collaborative Filtering (CF) lie in the source of data, how similarity is calculated, and their ability to handle data sparsity.

### Data Source: Behavior vs. Content

* **Traditional CF (User-to-User / Item-to-Item)**: These models are "memory-based" and rely exclusively on a binary interaction matrix (1 if a user read a book, 0 otherwise). They do not know anything about the book's title, author, or genre.
* **Hybrid Models (NCF & Matrix-Blend)**: Both models integrate Content-Based features. They use high-dimensional text embeddings from models to understand the semantic meaning of the books alongside the user behavior.

### Similarity Calculation: Linear vs. Non-Linear

* **Item-to-Item CF**: Calculates similarity using a direct cosine similarity dot-product between columns of the interaction matrix. It is a linear approach that looks for raw overlaps in who read what.
* **Hybrid Matrix-Blending**: Enhances the item-item similarity by mathematically blending the behavioral similarity with a semantic text similarity matrix using a weighted "alpha" parameter.
* **Hybrid NCF**: Moves away from direct similarity scores entirely. It uses a Multi-Layer Perceptron (MLP) to learn complex, non-linear interactions between user embeddings, item embeddings, and item metadata.

### Handling Sparse Data (The Cold Start Problem)

* **Traditional CF**: These models fail if there is no overlap between users or items. For instance, if a book has never been read, it can never be recommended because its column in the matrix is empty.
* **Hybrid Models**: Because these models incorporate text embeddings (metadata), they can recommend a book even if it has zero interactions. The models can recognize that a new book's summary is similar to a book the user has enjoyed in the past.

### Direct Linkage vs. Model Learning

* **Traditional CF**: Memory-based CF doesn't "train", it just looks at raw overlaps. If User A and User B read the same two books, traditional CF instantly identifies them as similar.
* **Hybrid NCF**: This is a "model-based" approach that requires thousands of training examples to update its internal weights. In sparse datasets (like ours, with a median of 6 interactions per user), NCF often struggles to find the same strong links that traditional CF finds instantly because it tries to learn patterns rather than just looking at the raw overlaps.

### Summary Comparison Table

| Feature | Traditional CF | Hybrid Matrix-Blend | Hybrid NCF |
| --- | --- | --- | --- |
| **Logic** | Raw Overlaps (Memory-based) | Blended Similarities | Deep Learning (Model-based) |
| **Inputs** | User-Item Interactions only | Interactions + Text Embeddings | IDs + Text + Numbers |
| **Cold Start** | Poor | Good | Good |
| **Relationship** | Linear (Dot-product) | Linear (Weighted average) | Non-linear (MLP layers) |
| **Speed** | Moderate | Very Fast | Slower (requires GPU training) |

# Performance table

|  | User-to-user | Item-to-item | NCF | Hybrid Matrix-Blending |
|---|---|---|---|---|
| Precision@10 | 0.0565 | 0.0557 | 0.0021 | 0.0628 |
| Recall@10 | 0.2904 | 0.2641 | 0.0106 | 0.3067 |

# Discussion

# Link to YouTube video
