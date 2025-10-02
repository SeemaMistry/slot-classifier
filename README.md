# slot-classifier
Slot classifier of key user profile data for mental health library resource chatbot (MDSC)

Purpose: filter user input from MDSC chatbot to extract out key "slot information" on symptoms, location, provider_type. Slots stored in user_state which will be passed to MIRA LLM to create more accurate responses without 'forgetting' or 're-asking' previously user defined data. 