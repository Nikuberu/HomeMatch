import os

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from utils import connect_lancedb_result

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"


def get_recommendations(recommendationDf, preferences):
    model_name = "gpt-3.5-turbo"
    # Load the model
    llm = OpenAI(model_name=model_name, temperature=0.5, max_tokens=4000)
    # Create a prompt template
    recommendations = recommendationDf.apply(connect_lancedb_result, axis=1).tolist()

    data_gen_template = """
        Here is the list of top 3 recommendations for a custoer based on their preferences.
        Our top 3 recommendations: {recommendations}
        Our customer's preferences: {preferences}.
        We have not revield our recommendations to the customer yet.
        Provide a friendly recommendation that is personalized to the customer's preferences.
    """

    prompt = PromptTemplate.from_template(data_gen_template).format(
        recommendations=recommendations, preferences=preferences
    )

    final_recommendation = llm(prompt)

    # save the listings to a file in the data directory
    return final_recommendation


if __name__ == "__main__":
    get_recommendations()
