from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_generator import training_df  # import data sample set

# Extract data
X_text = training_df['text']
y = training_df['label']

# TD-iDF vectorize 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# new samples for testing
new_examples = [
    "I have depression and live in Toronto",
    "I’m looking for a therapist in Vancouver",
    "Can you help me with ADHD?"
]

X_new = vectorizer.transform(new_examples)
predictions = clf.predict(X_new)
# for text, label in zip(new_examples, predictions):
#     print(f"'{text}' → {label}")



# ------- USER STATE --------

# Initial empty state
user_state = {
    "symptom": None,
    "location": None,
    "provider_type": None
}

def update_user_state(message, classifier, vectorizer, state):
    """
    message: str, new user chat message
    classifier: trained ML model
    vectorizer: TF-IDF vectorizer
    state: dict, current user state
    """
    # Transform the message into TF-IDF
    X_msg = vectorizer.transform([message])
    
    # Predict label
    label = classifier.predict(X_msg)[0]
    
    # Update state
    if label in state:
        state[label] = message  # store the text that matches this label
    
    return state

for msg in new_examples:
    user_state = update_user_state(msg, clf, vectorizer, user_state)
    print(user_state)