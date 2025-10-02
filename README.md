# slot-classifier
Slot classifier of key user profile data for mental health library resource chatbot (MDSC)

## Multi-Label Slot Extraction Overview

This project extracts user-provided information from conversational text into structured slots: symptom, location, and provider_type. It currently uses spaCy with the small English model (en_core_web_sm) for named entity recognition (NER) to detect locations (GPE) and a keyword-based approach to identify symptoms and provider types from predefined lists.

## Techniques Used:

NER with spaCy for location extraction.

Keyword matching with normalization (case-insensitive) for symptoms and provider types.

Set operations to maintain unique slot values across multiple messages in a conversation.

## Limitations & Future Improvements:

Current keyword matching is simple and may miss synonyms or casual language variations.

Symptoms and provider types can be noisy or duplicated due to overlapping keywords.

Future integration with a science-based or domain-specific NER model could improve extraction accuracy, handle synonyms, plural forms, and more complex entity patterns automatically.

## Examples 
Test cases of user chats sents to MIRA and the data updated to User State each time. Chat messages can contain multiple slots to collect at once. 

```bash
------ Chat id: 1 ------ Messages and User State Information: --------

Hi, I'm feeling really anxious lately and can't sleep at night.
{'symptom': ['anxious'], 'location': None, 'provider_type': None}
Time for this message: 0.0085 seconds

I think I need to see a therapist near Toronto.
{'symptom': ['anxious'], 'location': ['Toronto'], 'provider_type': ['therapist', 'Therapist']}
Time for this message: 0.0055 seconds

Also, sometimes I get headaches and fatigue during the day.
{'symptom': ['anxious', 'headache', 'fatigue'], 'location': ['Toronto'], 'provider_type': ['therapist', 'Therapist']}
Time for this message: 0.0106 seconds

Total time for chat: 0.0270 seconds

--------------------------------------------------


------ Chat id: 2 ------ Messages and User State Information: --------

Hey, I've been depressed and stressed for weeks.
{'symptom': ['stressed'], 'location': None, 'provider_type': None}
Time for this message: 0.0066 seconds

Do you know a good psychiatrist in Vancouver?
{'symptom': ['stressed'], 'location': ['Vancouver'], 'provider_type': ['Psychiatrist', 'psychiatrist']}
Time for this message: 0.0108 seconds

I also feel dizzy and have chest pain occasionally.
{'symptom': ['chest pain', 'stressed', 'pain'], 'location': ['Vancouver'], 'provider_type': ['Psychiatrist', 'psychiatrist']}
Time for this message: 0.0054 seconds

Total time for chat: 0.0248 seconds

--------------------------------------------------


------ Chat id: 3 ------ Messages and User State Information: --------

Hello, my child might have ADHD and needs help.
{'symptom': ['adhd'], 'location': None, 'provider_type': None}
Time for this message: 0.0087 seconds

Looking for a pediatrician or psychologist nearby.
{'symptom': ['adhd'], 'location': None, 'provider_type': ['Psychologist', 'psychologist', 'pediatrician']}
Time for this message: 0.0052 seconds

They get frustrated easily and have trouble focusing at school.
{'symptom': ['adhd'], 'location': None, 'provider_type': ['Psychologist', 'psychologist', 'pediatrician']}
Time for this message: 0.0104 seconds

Total time for chat: 0.0263 seconds

--------------------------------------------------


------ Chat id: 4 ------ Messages and User State Information: --------

I sometimes feel very sad and hopeless.
{'symptom': ['sad'], 'location': None, 'provider_type': None}
Time for this message: 0.0053 seconds

Is there a support group or online therapy I can join?
{'symptom': ['sad'], 'location': None, 'provider_type': ['online therapy', 'Support Group', 'Online Therapy', 'support group']}
Time for this message: 0.0083 seconds

Also, I've had nausea and fatigue for the past week.
{'symptom': ['nausea', 'fatigue', 'sad'], 'location': None, 'provider_type': ['online therapy', 'Support Group', 'Online Therapy', 'support group']}
Time for this message: 0.0059 seconds

Total time for chat: 0.0215 seconds

--------------------------------------------------


------ Chat id: 5 ------ Messages and User State Information: --------

Hi, I'm looking for addiction services in Calgary.
{'symptom': ['addiction'], 'location': ['Calgary'], 'provider_type': ['addiction services', 'Addiction Services']}
Time for this message: 0.0077 seconds

I have trouble with substance use and anxiety.
{'symptom': ['addiction', 'Anxiety', 'anxiety'], 'location': ['Calgary'], 'provider_type': ['addiction services', 'Addiction Services']}
Time for this message: 0.0058 seconds

Do you have any mental health counselor recommendations?
{'symptom': ['addiction', 'Anxiety', 'anxiety'], 'location': ['Calgary'], 'provider_type': ['addiction services', 'mental health counselor', 'Mental Health Counselor', 'Addiction Services']}
Time for this message: 0.0096 seconds

Total time for chat: 0.0252 seconds

--------------------------------------------------
```