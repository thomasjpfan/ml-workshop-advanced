from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42, max_depth=3)

rfc.fit(X_train, y_train)

rfc.score(X_test, y_test)

rfc_feature_importances = rfc.feature_importances_

rf_top_15 = rfc_feature_importances.argsort()[-20:]

feature_names[rf_top_15]
