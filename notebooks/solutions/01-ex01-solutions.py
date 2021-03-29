from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)

rf.score(X_test, y_test)

rf_feature_importance = rf.feature_importances_

top_rf_importance_argsort = rf_feature_importance.argsort()

top_20 = top_rf_importance_argsort[-20:]

top_rf_important_features = np.array(feature_names)[top_20]
top_rf_important_features
