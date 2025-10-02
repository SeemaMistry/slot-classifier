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


