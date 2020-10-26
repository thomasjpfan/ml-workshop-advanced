rf_pipe = Pipeline([
    ('vectorizer', CountVectorizer(min_df=5, stop_words='english')),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

rf_pipe.fit(text_train, y_train)
rf_pipe.score(text_val, y_val)
