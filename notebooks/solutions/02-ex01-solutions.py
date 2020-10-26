over_rf = make_imb_pipeline(
    RandomOverSampler(), LogisticRegression(random_state=42))

over_rf_scores = cross_validate(
    over_rf, X_train, y_train, cv=10,
    scoring=['roc_auc', 'average_precision'])

rf_base_auc, rf_base_ap

under_rf_auc, under_rf_ap

over_rf_auc = over_rf_scores['test_roc_auc'].mean()
over_rf_auc

over_rf_ap = over_rf_scores['test_average_precision'].mean()
over_rf_ap
