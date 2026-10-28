import os
import ast

from dotenv import load_dotenv
from supabase import create_client, Client


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")


if not SUPABASE_URL:
    raise ValueError(
        "SUPABASE_URL is missing from .env"
    )

if not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_KEY is missing from .env"
    )


# --------------------------------------------------
# CREATE SUPABASE CLIENT
# --------------------------------------------------

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# --------------------------------------------------
# VECTOR CONVERSION
# --------------------------------------------------

def parse_vector(vector):
    """
    Convert a PostgreSQL vector returned by Supabase
    into a Python list of floats.

    Supabase may return vector(768) as a string:

        "[-0.02, -0.01, ...]"

    This function converts it into:

        [-0.02, -0.01, ...]
    """

    if vector is None:
        return None

    # Already a Python list
    if isinstance(vector, list):
        return [
            float(value)
            for value in vector
        ]

    # PostgreSQL vector returned as string
    if isinstance(vector, str):

        try:
            parsed = ast.literal_eval(vector)

            if isinstance(parsed, list):
                return [
                    float(value)
                    for value in parsed
                ]

        except Exception as error:
            raise ValueError(
                f"Could not parse vector: {error}"
            )

    raise ValueError(
        f"Unsupported vector type: {type(vector)}"
    )


# --------------------------------------------------
# GET FOUND ITEMS
# --------------------------------------------------

def get_found_items():

    response = (
        supabase
        .table("found_items")
        .select("*")
        .execute()
    )

    found_items = []

    for item in response.data:

        image_url = item["image_url"]

        found_items.append({

            # Database ID
            "id": item["id"],
            "report_id": item["report_id"],

            # Original filename
            "image": (
                image_url.split("/")[-1]
                if image_url
                else None
            ),

            # Full Supabase Storage URL
            "image_url": image_url,

            # Item title
            "title": item["title"],

            # Location
            "location": {
                "campus": item["campus"],
                "area": item["area"],
                "details": item["details"]
            },

            # Category
            "category": item["category"],

            # --------------------------------------------------
            # SIGLIP EMBEDDINGS
            # --------------------------------------------------

            "image_embedding": parse_vector(
                item["image_embedding"]
            ),

            "text_embedding": parse_vector(
                item["text_embedding"]
            )
        })

    return found_items

# --------------------------------------------------
# GET LOST ITEMS
# --------------------------------------------------

def get_lost_items():
    response = (
        supabase
        .table("lost_items")
        .select("*")
        .eq("status", "active")
        .execute()
    )

    lost_items = []

    for item in response.data:
        image_url = item["image_url"]

        lost_items.append({
            "id": item["id"],
            "report_id": item["report_id"],
            "image": (
                image_url.split("/")[-1]
                if image_url
                else None
            ),
            "image_url": image_url,
            "title": item["title"],
            "location": {
                "campus": item["campus"],
                "area": item["area"],
                "details": item["details"]
            },
            "category": item["category"],
            "status": item["status"],
            "image_embedding": parse_vector(
                item["image_embedding"]
            ),
            "text_embedding": parse_vector(
                item["text_embedding"]
            )
        })

    return lost_items