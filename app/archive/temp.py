full_data_mtx = create_data_mtx(interactions, n_users, n_items)
print("Number of interactions in full set:", np.sum(full_data_mtx))

# 2. Recompute Item-Item similarities and predictions on the full dataset
print("Computing full Item-based predictions...")
full_item_similarity = cosine_similarity(full_data_mtx.T)
full_item_prediction = item_based_predict(full_data_mtx, full_item_similarity)

# 3. Recompute User-User similarities and predictions on the full dataset
print("Computing full User-based predictions...")
full_user_similarity = cosine_similarity(full_data_mtx)
full_user_prediction = user_based_predict(full_data_mtx, full_user_similarity)

# 4. Build the final submission using the completely retrained predictions
sample_submission = pd.read_csv(Path.cwd().parent/"data"/"sample_submission.csv")

submission_user = pd.DataFrame()
submission_user["user_id"] = sample_submission["user_id"]

submission_item = pd.DataFrame()
submission_item["user_id"] = sample_submission["user_id"]

predictions_user = []
predictions_item = []

for u in submission_user["user_id"]:
    
    # Copy the scores for the user
    u_scores_user_cf = full_user_prediction[u, :].copy()
    u_scores_item_cf = full_item_prediction[u, :].copy()

    top_10_users_cf = np.argsort(u_scores_user_cf)[-10:]
    top_10_items_cf = np.argsort(u_scores_item_cf)[-10:]

    temp_u = []
    temp_i = []
    i_in_order = interactions["i in order"].to_list()
    for i in top_10_users_cf:
        temp_u.append(interactions["i"][i_in_order.index(i)])
    top_10_users_cf = np.array(temp_u)
    for i in top_10_items_cf:
        temp_i.append(interactions["i"][i_in_order.index(i)])
    top_10_items_cf = np.array(temp_i)
    
    predictions_user.append(" ".join(map(str, top_10_users_cf)))
    predictions_item.append(" ".join(map(str, top_10_items_cf)))

submission_user["recommendation"] = predictions_user
submission_item["recommendation"] = predictions_item

# Save files
submission_user.to_csv(Path.cwd().parent/"submissions"/"submission_user_full_unfiltered_v3.csv", index=False)
submission_item.to_csv(Path.cwd().parent/"submissions"/"submission_item_full_unfiltered_v3.csv", index=False)