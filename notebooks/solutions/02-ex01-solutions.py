over_rf = make_imb_pipeline(RandomOverSampler(), RandomForestClassifier(random_state=42))

base_rf_metrics

compute_metrics(over_rf)
