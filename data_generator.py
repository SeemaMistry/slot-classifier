import pandas as pd

symptoms = ["Anxiety", "Depression", "Bipolar disorder", "Schizophrenia", "Post‑traumatic stress disorder (PTSD)", "Obsessive‑compulsive disorder (OCD)", "Eating disorders", "Attention‑deficit/hyperactivity disorder (ADHD)", "Autism spectrum disorder", "Substance use disorders",
             "fatigue", "sad", "stressed", "anxious"  # common layperson terms
             "ADHD",
               "headache", "fever", "cough", "fatigue", "nausea", "dizziness", 
    "pain", "shortness of breath", "rash", "chest pain",
        # Mental health conditions (formal)
    "anxiety", "depression", "bipolar disorder", "schizophrenia", 
    "post-traumatic stress disorder", "ptsd", "obsessive-compulsive disorder", "ocd",
    "eating disorders", "adhd", "attention-deficit/hyperactivity disorder", 
    "autism spectrum disorder", "substance use disorders", "addiction",

    # Layperson mental health terms
    "sad", "stressed", "anxious", "overwhelmed", "nervous", "scared", 
    "down", "burned out", "tired", "moody", "panic attacks", "worrying too much", 
    "can't focus", "trouble sleeping", "sleep issues", "low energy", "brain fog",

    # Physical symptoms
    "headache", "fever", "cough", "fatigue", "nausea", "dizziness", 
    "pain", "shortness of breath", "rash", "chest pain", "stomach ache", 
    "back pain", "muscle pain", "joint pain", "numbness", "tingling", "heart palpitations"

             ]
locations = ["British Columbia", "Alberta", "Saskatchewan", "Manitoba", "Ontario", "Quebec", "Nova Scotia", "Prince Edward Island", "Northwest Territories", "Nunavut",  "Toronto",
    "Montreal",
    "Vancouver",
    "Calgary",
    "Edmonton",
    "Ottawa",
    "Winnipeg",
    "Quebec City",
    "Hamilton",
    "Kitchener",
    "London",
    "Victoria",
    "Halifax",
    "Oshawa",
    "Windsor",
    "Saskatoon",
    "Regina",
    "St. John's",
    "Kelowna",
    "Barrie"]
provider_type = ["Therapist", "Psychologist", "Psychiatrist", "Primary Care Physician", "Clinical Social Worker", "Counsellor", "Peer Support", "Crisis Hotline", "Online Therapy", "Support Group", "Medication Management", "Addiction Services", "Rehabilitation Services", "Mental Health Clinic", "Health Care Provider", "Mental Health Counselor", "Mental Health Nurse",
                     "gp", "general practitioner", "cardiologist", "dermatologist",
    # Mental health
    "therapist", "psychologist", "psychiatrist", "primary care physician", "clinical social worker", 
    "counsellor", "peer support", "crisis hotline", "online therapy", "support group", 
    "medication management", "addiction services", "rehabilitation services", "mental health clinic", 
    "health care provider", "mental health counselor", "mental health nurse",

    # General and specialist doctors
    "gp", "general practitioner", "family doctor", "doctor", "cardiologist", "dermatologist", 
    "pediatrician", "surgeon", "oncologist", "internist", "neurologist", "endocrinologist",

    # Casual user terms
    "shrink", "therapist online", "online doctor", "doctor appointment", "clinic", "walk-in clinic"
         
                 ]

data = []

# symptoms 
for s in symptoms:
    data.append((f"I have {s.lower()}", "symptom"))
    data.append((f"I'm struggling with {s.lower()}", "symptom"))
    data.append((f"Can you help me with my {s.lower()}?", "symptom"))
    data.append((f"I’ve been diagnosed with {s.lower()}", "symptom"))
    data.append((f"My main issue is {s.lower()}", "symptom"))
    data.append((f"Lately I’ve been dealing with {s.lower()}", "symptom"))


# locations 
for l in locations:
    data.append((f"I live in {l}", "location"))
    data.append((f"Do you have resources in {l}?", "location"))
    data.append((f"I'm from {l}", "location"))
    data.append((f"Can I get help in {l}", "location"))
    data.append((f"Resources near {l}", "location"))
    data.append((f"I'm located in {l}", "location"))

# providers 
for p in provider_type:
    data.append((f"I need a {p.lower()}", "provider_type"))
    data.append((f"Can you find me a {p.lower()}?", "provider_type"))
    data.append((f"I'm looking for a {p.lower()}", "provider_type"))
    data.append((f"Do you know a good {p.lower()}", "provider_type"))
    data.append((f"Where can I find a {p.lower()}", "provider_type"))
    data.append((f"I’m searching for {p.lower()}", "provider_type"))
    data.append((f"I’m want a {p.lower()}", "provider_type"))


df = pd.DataFrame(data, columns=['text', 'label']) # current size = 359
# df.to_csv("training_data.csv", index=False)
# print(df.head(10))
# print(f"Total examples: {len(df)}")

training_df = df