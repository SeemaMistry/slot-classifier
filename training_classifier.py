from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_generator import training_df, locations, symptoms, provider_type  # import data sample set

import time

import spacy
# Load SpaCy NER model for location extraction
nlp = spacy.load("en_core_web_sm")

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



# ------- MULTI LABEL DATA EXTRACTION WITH NER --------

# Multi-label data extrations

# Our personal currated list of knows (for speed and retain what 100% we care about)
def extract_entities(message, label_list):
    """Return a list of label keywords found in the message (case-insensitive)"""
    message_lower = message.lower()
    return [item for item in label_list if item.lower() in message_lower]

# SpacCy to extract locations
def extract_locations_ner(message):
    doc = nlp(message)
    return [ent.text for ent in doc.ents if ent.label_ == "GPE"]



# ------- USER STATE --------

# Initial empty state
user_state = {
    "symptom": None,
    "location": None,
    "provider_type": None
}

def update_user_state_multilabel(message, state):
    """
    message: str, new user chat message
    state: dict, current user state
    Update user_state for symptom, location, and provider_type all at once.
    """
    # Symptoms (keyword look up only)
    # would like to update to SciSpacy and en_ner_bionlp13cg_md but wont download on my system
    found_symptoms = set(extract_entities(message, symptoms))
    if found_symptoms:
        if state["symptom"] is None:
            state["symptom"] = list(found_symptoms)
        else:
            state["symptom"] = list(set(state["symptom"]) | found_symptoms)
    
    # Locations (with NER)
    found_locations = set(extract_entities(message, locations) + extract_locations_ner(message))
    if found_locations:
        if state["location"] is None:
            state["location"] = list(found_locations)
        else:
            state["location"] = list(set(state["location"]) | set(found_locations))

    # Provider Type
    # would like to update to SciSpacy and en_ner_bionlp13cg_md but wont download on my system
    found_provider = set(extract_entities(message, provider_type))
    if found_provider:
        if state["provider_type"] is None:
            state["provider_type"] = list(found_provider)
        else:
            state["provider_type"] = list(set(state["provider_type"]) | found_provider)


    return state

# multi_label_messages = [
#     "I have depression and live in Perth but can also go to Ottawa",
#     "I’m looking for a therapist in Ontario",
#     "Can you help me with ADHD and anxiety?",
#     "I live in Russell but I have a car. I can go to any location in the Ottawa area."

# ]

# for msg in multi_label_messages:
#     user_state = update_user_state_multilabel(msg, user_state)
#     print(user_state)


# Sample conversations object
sample_chats = [
    {
        "id": 1,
        "user": "User1",
        "messages": [
            "Hi, I'm feeling really anxious lately and can't sleep at night.",
            "I think I need to see a therapist near Toronto.",
            "Also, sometimes I get headaches and fatigue during the day."
        ]
    },
    {
        "id": 2,
        "user": "User2",
        "messages": [
            "Hey, I've been depressed and stressed for weeks.",
            "Do you know a good psychiatrist in Vancouver?",
            "I also feel dizzy and have chest pain occasionally."
        ]
    },
    {
        "id": 3,
        "user": "User3",
        "messages": [
            "Hello, my child might have ADHD and needs help.",
            "Looking for a pediatrician or psychologist nearby.",
            "They get frustrated easily and have trouble focusing at school."
        ]
    },
    {
        "id": 4,
        "user": "User4",
        "messages": [
            "I sometimes feel very sad and hopeless.",
            "Is there a support group or online therapy I can join?",
            "Also, I've had nausea and fatigue for the past week."
        ]
    },
    {
        "id": 5,
        "user": "User5",
        "messages": [
            "Hi, I'm looking for addiction services in Calgary.",
            "I have trouble with substance use and anxiety.",
            "Do you have any mental health counselor recommendations?"
        ]
    }
]


for chat in sample_chats:
    print(f"------ Chat id: {chat['id']} ------ Messages and User State Information: --------\n" )
    
    chat_start = time.time() # start total chat timer

    for msg in chat['messages']:
        msg_start = time.time()  # start per-message timer
        state = update_user_state_multilabel(msg, user_state)
        msg_end = time.time()
        print(f"{msg}\n{state}")
        print(f"Time for this message: {msg_end - msg_start:.4f} seconds\n") # message time
    
    chat_end = time.time() # end total chat timer 
    print(f"Total time for chat: {chat_end - chat_start:.4f} seconds\n") # chat time
    print(f"--------------------------------------------------\n\n")
    # reset for next chat
    user_state = {
    "symptom": None,
    "location": None,
    "provider_type": None
}



