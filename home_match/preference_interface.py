from utils import generate_embeddings


def get_preferences_and_generate_embeddings():
    questions = [
        "How big do you want your house to be?",
        "What are 3 most important things for you in choosing this property?",
        "Which amenities would you like?",
        "Which transportation options are important to you?",
        "How urban do you want your neighborhood to be?",
    ]
    default_answers = [
        "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
        "A quiet neighborhood, good local schools, and convenient shopping options.",
        "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
        "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters.",
    ]

    preferences = []

    for (question, default_answer) in zip(questions, default_answers):
        answer = input(f"{question} (default: {default_answer}): ")
        if not answer:
            answer = default_answer

        preferences.append(question + ": " + answer)

    text_preferences = " ".join(preferences)

    return (generate_embeddings(text_preferences), text_preferences)
