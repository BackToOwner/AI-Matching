from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from PIL import Image
from io import BytesIO
import requests

from matcher import (
    match_lost_against_found,
    match_found_against_lost
)

from database import (
    supabase,
    get_found_items,
    get_lost_items
)

from lost_item_service import create_lost_item
from found_item_service import create_found_item

from model import (
    get_image_embedding,
    get_text_embedding
)
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Lost & Found AI",
    description="Multimodal Lost & Found matching API",
    version="1.0.0"
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Lost & Found AI API is running!",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "google/siglip-base-patch16-224"
    }


# ============================================================
# GENERATE EMBEDDINGS FOR AN EXISTING REPORT-LINKED ITEM
# ============================================================

@app.post("/embeddings/{item_type}/{report_id}")
def generate_existing_item_embeddings(item_type: str, report_id: str):
    if item_type not in ("lost", "found"):
        raise HTTPException(status_code=400, detail="Invalid item type.")

    table = "lost_items" if item_type == "lost" else "found_items"

    try:
        response = (
            supabase.table(table)
            .select("id, report_id, title, image_url")
            .eq("report_id", report_id)
            .maybe_single()
            .execute()
        )

        item = response.data
        if not item:
            raise HTTPException(
                status_code=404,
                detail="Existing AI item for this report was not found."
            )

        image_url = item.get("image_url")
        if not image_url:
            raise HTTPException(
                status_code=400,
                detail="The existing item has no image URL."
            )

        image_response = requests.get(image_url, timeout=30)
        image_response.raise_for_status()

        image = Image.open(
            BytesIO(image_response.content)
        ).convert("RGB")

        image_embedding = (
            get_image_embedding(image).detach().cpu().tolist()
        )
        text_embedding = (
            get_text_embedding(item["title"]).detach().cpu().tolist()
        )

        if len(image_embedding) != 768 or len(text_embedding) != 768:
            raise HTTPException(
                status_code=500,
                detail="Unexpected SigLIP embedding dimensions."
            )

        saved = (
            supabase.table(table)
            .update({
                "image_embedding": image_embedding,
                "text_embedding": text_embedding
            })
            .eq("id", item["id"])
            .eq("report_id", report_id)
            .select("id, report_id")
            .execute()
        )

        if not saved.data:
            raise HTTPException(
                status_code=500,
                detail="Embeddings were not confirmed saved."
            )

        return {
            "success": True,
            "item_id": item["id"],
            "report_id": report_id,
            "image_embedding_dimension": len(image_embedding),
            "text_embedding_dimension": len(text_embedding)
        }

    except HTTPException:
        raise
    except Exception as error:
        print(f"Embedding generation failed for {report_id}: {error}")
        raise HTTPException(
            status_code=500,
            detail="Could not generate or save embeddings."
        )


# ============================================================
# CREATE LOST ITEM
# ============================================================

@app.post("/lost-items")
async def create_lost_item_endpoint(
    title: str = Form(...),
    campus: str = Form(...),
    area: str = Form(...),
    details: str = Form(""),
    category: str = Form(...),
    image_url: str = Form(...)
):
    """
    Create a persistent lost-item record.

    The image must already be uploaded to Supabase Storage.

    The service automatically generates:

        Image embedding -> SigLIP
        Text embedding  -> SigLIP

    Both embeddings are stored in Supabase.
    """

    if not image_url.strip():
        raise HTTPException(
            status_code=400,
            detail="image_url cannot be empty."
        )

    try:

        result = create_lost_item(
            title=title,
            campus=campus,
            area=area,
            details=details,
            category=category,
            image_url=image_url
        )

        return {
            "success": True,
            "message": "Lost item created successfully.",
            "lost_item": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# CREATE FOUND ITEM
# ============================================================

@app.post("/found-items")
async def create_found_item_endpoint(
    title: str = Form(...),
    campus: str = Form(...),
    area: str = Form(...),
    details: str = Form(""),
    category: str = Form(...),
    image_url: str = Form(...)
):
    """
    Create a persistent found-item record.

    The image must already be uploaded to Supabase Storage.

    The service automatically generates:

        Image embedding -> SigLIP
        Text embedding  -> SigLIP

    Both embeddings are stored in Supabase.
    """

    if not image_url.strip():
        raise HTTPException(
            status_code=400,
            detail="image_url cannot be empty."
        )

    try:

        result = create_found_item(
            title=title,
            campus=campus,
            area=area,
            details=details,
            category=category,
            image_url=image_url
        )

        return {
            "success": True,
            "message": "Found item created successfully.",
            "found_item": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# MATCH LOST ITEM AGAINST FOUND ITEMS
# ============================================================

@app.post("/match")
async def match_item(
    image: UploadFile = File(...),
    title: str = Form(...),
    campus: str = Form(...),
    area: str = Form(...),
    details: str = Form(""),
    category: str = Form(...)
):

    if (
        not image.content_type
        or not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    image_bytes = await image.read()

    try:

        lost_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # --------------------------------------------------------
    # Generate temporary embeddings for instant matching
    # --------------------------------------------------------

    print(
        "\nGenerating lost-item image embedding..."
    )

    image_embedding = get_image_embedding(
        lost_image
    )

    print(
        "Generating lost-item text embedding..."
    )

    text_embedding = get_text_embedding(
        title
    )

    lost_item = {
        "title": title,

        "location": {
            "campus": campus,
            "area": area,
            "details": details
        },

        "category": category,

        "image_embedding":
            image_embedding
            .detach()
            .cpu()
            .tolist(),

        "text_embedding":
            text_embedding
            .detach()
            .cpu()
            .tolist()
    }

    # --------------------------------------------------------
    # Get active found items
    # --------------------------------------------------------

    found_items = get_found_items()

    # --------------------------------------------------------
    # Lost -> Found
    # --------------------------------------------------------

    results = match_lost_against_found(
        lost_item=lost_item,
        found_items=found_items
    )

    matches = []

    for result in results:

        matches.append({
            "id": result["id"],
            "report_id": result.get("report_id"),
            "image": result["image_url"],
            "title": result["title"],
            "location": result["location"],
            "category": result["category"],

            "image_score":
                result["image_score"],

            "text_score":
                result["text_score"],

            "location_score":
                result["location_score"],

            "category_score":
                result["category_score"],

            "match_score":
                result["final_score"],

            "match_percentage":
                round(
                    result["final_score"] * 100,
                    2
                )
        })

    return {
        "success": True,

        "lost_item": {
            "title": title,

            "location": {
                "campus": campus,
                "area": area,
                "details": details
            },

            "category": category
        },

        "matches": matches
    }


# ============================================================
# MATCH FOUND ITEM AGAINST LOST ITEMS
# ============================================================

@app.post("/match-found")
async def match_found_item(
    image: UploadFile = File(...),
    title: str = Form(...),
    campus: str = Form(...),
    area: str = Form(...),
    details: str = Form(""),
    category: str = Form(...)
):

    if (
        not image.content_type
        or not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    image_bytes = await image.read()

    try:

        found_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # --------------------------------------------------------
    # Generate temporary embeddings
    # --------------------------------------------------------

    print(
        "\nGenerating found-item image embedding..."
    )

    image_embedding = get_image_embedding(
        found_image
    )

    print(
        "Generating found-item text embedding..."
    )

    text_embedding = get_text_embedding(
        title
    )

    found_item = {
        "title": title,

        "location": {
            "campus": campus,
            "area": area,
            "details": details
        },

        "category": category,

        "image_embedding":
            image_embedding
            .detach()
            .cpu()
            .tolist(),

        "text_embedding":
            text_embedding
            .detach()
            .cpu()
            .tolist()
    }

    # --------------------------------------------------------
    # Get active lost items
    # --------------------------------------------------------

    lost_items = get_lost_items()

    # --------------------------------------------------------
    # Found -> Lost
    # --------------------------------------------------------

    results = match_found_against_lost(
        found_item=found_item,
        lost_items=lost_items
    )

    matches = []

    for result in results:

        matches.append({
            "id": result["id"],
            "report_id": result.get("report_id"),
            "image": result["image_url"],
            "title": result["title"],
            "location": result["location"],
            "category": result["category"],

            "image_score":
                result["image_score"],

            "text_score":
                result["text_score"],

            "location_score":
                result["location_score"],

            "category_score":
                result["category_score"],

            "match_score":
                result["final_score"],

            "match_percentage":
                round(
                    result["final_score"] * 100,
                    2
                )
        })

    return {
        "success": True,

        "found_item": {
            "title": title,

            "location": {
                "campus": campus,
                "area": area,
                "details": details
            },

            "category": category
        },

        "matches": matches
    }

