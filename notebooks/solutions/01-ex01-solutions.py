from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1)
rf.fit(X_train, y_train)

rf.score(X_val, y_val)

rf_feature_importance = rf.feature_importances_
top_rf_importance_indices = rf_feature_importance.argsort()[::-1][:20]

top_rf_important_features = np.array(feature_names)[top_rf_importance_indices]
top_rf_important_features
