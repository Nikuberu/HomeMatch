import os

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"


def generate_listings():
    model_name = "gpt-3.5-turbo"
    # Load the model
    llm = OpenAI(model_name=model_name, temperature=0.5, max_tokens=4000)
    sample = {
        "Neighborhood": "Green Oaks",
        "Price": "$800,000",
        "Bedrooms": 3,
        "Bathrooms": 2,
        "House Size": "2,000 sqft",
        "Description": "Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.",
        "Neighborhood Description": "Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.",
    }

    # Create a prompt template
    data_gen_template = """
        Generate 10 real estate listings for homes around the world. It should be csv formatted with 7 columns. Here is the sample data: {sample}.

        CSV:
    """

    print(data_gen_template)

    prompt = PromptTemplate.from_template(data_gen_template).format(sample=sample)

    listings = llm(prompt)

    print(listings)

    # save the listings to a file in the data directory
    with open("data/listings.csv", "w") as f:
        f.write(listings)

    return listings


if __name__ == "__main__":
    generate_listings()
