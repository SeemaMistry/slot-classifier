import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# small sample dataset of user message to MIRA
data = {
    "text": [
        "I feel depressed",
        "I live in Toronto",
        "Can you find me a therapist?",
        "My anxiety is really bad",
        "I’m looking for a clinic in Alberta",
        "I need a psychologist"        
    ],
    "label": [
        "symptom",
        "location",
        "provider_type",
        "symptom",
        "location",
        "provider_type"
    ]
}

df = pd.DataFrame(data)
print(df)


# vectorize sample data and transform using TF-IDF
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df["text"])
y =  df["label"]

print("Shape of TF-IDF matrix:", x.shape)
print("Vocabulary:", vectorizer.get_feature_names_out())


# Train logistic regression classifier
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(x, y)

y_pred = clf.predict(x)
print(classification_report(y, y_pred))

symptoms = ["Anxiety", "Depression", "Bipolar disorder", "Schizophrenia", "Post‑traumatic stress disorder (PTSD)", "Obsessive‑compulsive disorder (OCD)", "Eating disorders", "Attention‑deficit/hyperactivity disorder (ADHD)", "Autism spectrum disorder", "Substance use disorders"]
locations = ["British Columbia", "Alberta", "Saskatchewan", "Manitoba", "Ontario", "Quebec", "Nova Scotia", "Prince Edward Island", "Northwest Territories", "Nunavut"]
provider_type = ["Therapist", "Psychologist", "Psychiatrist", "Primary Care Physician", "Clinical Social Worker", "Counsellor", "Peer Support", "Crisis Hotline", "Online Therapy", "Support Group", "Medication Management", "Addiction Services", "Rehabilitation Services", "Mental Health Clinic", "Health Care Provider", "Mental Health Counselor", "Mental Health Nurse"]
