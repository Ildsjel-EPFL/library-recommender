import numpy as np
import numpy.typing as npt
import pandas as pd


def create_data_mtx(data : pd.DataFrame, interactions : pd.DataFrame, n_users: int) -> npt.NDArray[np.float64]:
    """
    Create a user-item interaction matrix from the given data.

    :param data: A DataFrame containing user-item interactions with columns "u" for user IDs and "i" for item IDs.
    :type data: pd.DataFrame
    :param interactions: The original interactions DataFrame, used to determine the maximum item ID for matrix dimensions.
    :type interactions: pd.DataFrame
    :param n_users: The number of users.
    :type n_users: int
    :return: A user-item interaction matrix.
    :rtype: npt.NDArray[np.float64]
    """
    data_mtx = np.zeros((n_users, max(interactions.i.unique())+1))
    data_mtx[data["u"].values, data["i"].values] = 1
    return data_mtx

def item_based_predict(interactions : pd.DataFrame, similarity : npt.NDArray[np.float64], epsilon=1e-9):
    """
    Predicts user-item interactions based on item-item similarity.

    :param interactions: The user-item interaction matrix.
    :type interactions: pd.DataFrame
    :param similarity: The item-item similarity matrix.
    :type similarity: npt.NDArray[np.float64]
    :param epsilon: Small constant added to the denominator to avoid division by zero.
    :type epsilon: float
    :return: The predicted interaction scores for each user-item pair.
    :rtype: npt.NDArray[np.float64]
    """
    # np.dot does the matrix multiplication. Here we are calculating the
    # weighted sum of interactions based on item similarity
    pred = similarity.dot(interactions.T) / (similarity.sum(axis=1)[:, np.newaxis] + epsilon)
    return pred.T  # Transpose to get users as rows and items as columns

def create_submission(sample_submission_df : pd.DataFrame, predictions : npt.NDArray[np.float64], output_path : str):
    """
    Create a submission file from the predicted interaction scores.

    :param sample_submission_df: A DataFrame containing the sample submission format with columns "u" for user IDs and "i" for item IDs.
    :type sample_submission_df: pd.DataFrame
    :param predictions: The predicted interaction scores for each user-item pair.
    :type predictions: npt.NDArray[np.float64]
    :param output_path: The file path to save the generated submission CSV.
    :type output_path: str
    """
    predictions_item = []
    for u in sample_submission_df["user_id"]:
        u_scores = predictions[u, :].copy()
        top_10 = np.argsort(u_scores)[-10:][::-1]
        predictions_item.append(" ".join(map(str, top_10)))
    sample_submission_df["recommendation"] = predictions_item
    sample_submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
