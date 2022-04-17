rfc_pipe = Pipeline([
    ("vectorizer", CountVectorizer(min_df=2, stop_words='english')),
    ("rf", RandomForestClassifier(random_state=42, max_depth=3))
])

rfc_pipe.fit(text_train, y_train)

rfc_pipe.score(text_test, y_test)
