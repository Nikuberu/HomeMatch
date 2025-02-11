import warnings

import generate_listings
import lance_db
import preference_interface
import recommendation

# just for the sake of this demo, we will ignore warnings
warnings.filterwarnings("ignore")


def main():
    print("HomeMatch application starting...")
    # Generating listing csv file in data folder
    # commenting this out as the sample data is already generated
    # generate_listings.generate_listings()
    print("Listings generated successfully.")

    # Initialize LanceDB with data
    tbl = lance_db.init_lance_db_with_data()

    # Get user preferences and generate embeddings
    (
        embeddings,
        preferences,
    ) = preference_interface.get_preferences_and_generate_embeddings()

    # Get recommendations
    recommendationDf = tbl.search(embeddings).limit(3).to_pandas()
    recommendation_msg = recommendation.get_recommendations(
        recommendationDf, preferences
    )
    print("===================================================== \n")
    print(recommendation_msg)


if __name__ == "__main__":
    main()
