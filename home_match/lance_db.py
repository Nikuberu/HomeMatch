import lancedb
import pandas as pd
from lancedb.pydantic import LanceModel, vector

from utils import connect_listing, generate_embeddings


class Listing(LanceModel):
    neighborhood: str
    price: str
    bedrooms: int
    bathrooms: int
    house_size: str
    description: str
    neighborhood_description: str
    embedding: vector(384)


def init_lance_db_with_data() -> lancedb.table:
    table_name = "listings"
    # Drop table
    drop_table(table_name)
    # Initialize LanceDB
    uri = "data/home_match_lancedb"
    print("Connecting to LanceDB...")
    db = lancedb.connect(uri)
    print("Creating table...")
    tbl = db.create_table(table_name, schema=Listing)
    # Load data as Listing objects
    print("Loading data...")
    data = load_data()
    # Add data to the table
    print("Adding data to the table...")
    tbl.add(data)

    return tbl


def load_data(path: str = "./data/listings.csv") -> list[Listing]:
    # load pre-generated data
    with open("./data/listings.csv") as f:
        df = pd.read_csv(f)
        df["text"] = df.apply(connect_listing, axis=1)
        df["embedding"] = list(generate_embeddings(df["text"].tolist()))
        data = []
        for _, row in df.iterrows():
            data.append(
                Listing(
                    neighborhood=row["Neighborhood"],
                    price=row["Price"],
                    bedrooms=row["Bedrooms"],
                    bathrooms=row["Bathrooms"],
                    house_size=row["House Size"],
                    description=row["Description"],
                    neighborhood_description=row["Neighborhood Description"],
                    embedding=row["embedding"],
                )
            )
        return data


def drop_table(table_name: str, ignore_missing=True):
    uri = "data/home_match_lancedb"
    db = lancedb.connect(uri)
    try:
        db.drop_table(table_name, ignore_missing=ignore_missing)
        print(f"Table '{table_name}' dropped successfully.")
    except Exception as e:
        print(f"Error dropping table '{table_name}': {e}")


if __name__ == "__main__":
    init_lance_db_with_data()
