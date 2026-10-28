import torch

TOP_K = 5

# Matching weights
IMAGE_WEIGHT = 0.55
TEXT_WEIGHT = 0.30
LOCATION_WEIGHT = 0.05
CATEGORY_WEIGHT = 0.10

# Minimum score required to be considered a possible match
MIN_MATCH_SCORE = 0.70


def cosine_similarity(embedding1, embedding2):
    if not isinstance(embedding1, torch.Tensor):
        embedding1 = torch.tensor(
            embedding1,
            dtype=torch.float32
        )

    if not isinstance(embedding2, torch.Tensor):
        embedding2 = torch.tensor(
            embedding2,
            dtype=torch.float32
        )

    if embedding1.shape != embedding2.shape:
        raise ValueError(
            "Embedding dimensions do not match: "
            f"{embedding1.shape} vs {embedding2.shape}"
        )

    return torch.dot(embedding1, embedding2).item()


def calculate_location_score(
    item1_location,
    item2_location
):
    item1_campus = (
        item1_location.get("campus", "")
        .strip()
        .lower()
    )

    item2_campus = (
        item2_location.get("campus", "")
        .strip()
        .lower()
    )

    if not item1_campus or not item2_campus:
        return 0.0

    if item1_campus == item2_campus:
        return 1.0

    return 0.0


def calculate_category_score(
    item1_category,
    item2_category
):
    item1 = item1_category.strip().lower()
    item2 = item2_category.strip().lower()

    if item1 == item2:
        return 1.0

    return 0.0


def calculate_match_score(item1, item2):

    image_embedding1 = item1.get("image_embedding")
    image_embedding2 = item2.get("image_embedding")

    text_embedding1 = item1.get("text_embedding")
    text_embedding2 = item2.get("text_embedding")

    if image_embedding1 is None or image_embedding2 is None:
        return None

    if text_embedding1 is None or text_embedding2 is None:
        return None

    # --------------------------------------------------
    # 1. IMAGE SIMILARITY
    # --------------------------------------------------

    image_score = cosine_similarity(
        image_embedding1,
        image_embedding2
    )

    # --------------------------------------------------
    # 2. TEXT SIMILARITY
    # --------------------------------------------------

    text_score = cosine_similarity(
        text_embedding1,
        text_embedding2
    )

    # --------------------------------------------------
    # 3. LOCATION
    # --------------------------------------------------

    location_score = calculate_location_score(
        item1.get("location", {}),
        item2.get("location", {})
    )

    # --------------------------------------------------
    # 4. CATEGORY
    # --------------------------------------------------

    category_score = calculate_category_score(
        item1.get("category", ""),
        item2.get("category", "")
    )

    # --------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------

    final_score = (
        image_score * IMAGE_WEIGHT
        + text_score * TEXT_WEIGHT
        + location_score * LOCATION_WEIGHT
        + category_score * CATEGORY_WEIGHT
    )

    return {
        "image_score": round(image_score, 4),
        "text_score": round(text_score, 4),
        "location_score": round(location_score, 4),
        "category_score": round(category_score, 4),
        "final_score": round(final_score, 4)
    }


def match_lost_against_found(
    lost_item,
    found_items
):

    results = []

    for found_item in found_items:

        # --------------------------------------------------
        # CATEGORY FILTER
        # --------------------------------------------------

        lost_category = (
            lost_item.get("category", "")
            .strip()
            .lower()
        )

        found_category = (
            found_item.get("category", "")
            .strip()
            .lower()
        )

        # Do not compare completely different categories
        if lost_category != found_category:
            continue

        # --------------------------------------------------
        # CALCULATE AI MATCH SCORE
        # --------------------------------------------------

        scores = calculate_match_score(
            lost_item,
            found_item
        )

        if scores is None:
            continue

        # --------------------------------------------------
        # MINIMUM SCORE FILTER
        # --------------------------------------------------

        if scores["final_score"] < MIN_MATCH_SCORE:
            continue

        results.append({
            "id": found_item.get("id"),
            "report_id": found_item.get("report_id"),
            "image": found_item.get("image"),
            "image_url": found_item.get("image_url"),
            "title": found_item.get("title"),
            "location": found_item.get("location", {}),
            "category": found_item.get("category"),
            **scores
        })

    results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return results[:TOP_K]


def match_found_against_lost(
    found_item,
    lost_items
):

    results = []

    for lost_item in lost_items:

        # --------------------------------------------------
        # CATEGORY FILTER
        # --------------------------------------------------

        found_category = (
            found_item.get("category", "")
            .strip()
            .lower()
        )

        lost_category = (
            lost_item.get("category", "")
            .strip()
            .lower()
        )

        # Do not compare completely different categories
        if found_category != lost_category:
            continue

        # --------------------------------------------------
        # CALCULATE AI MATCH SCORE
        # --------------------------------------------------

        scores = calculate_match_score(
            found_item,
            lost_item
        )

        if scores is None:
            continue

        # --------------------------------------------------
        # MINIMUM SCORE FILTER
        # --------------------------------------------------

        if scores["final_score"] < MIN_MATCH_SCORE:
            continue

        results.append({
            "id": lost_item.get("id"),
            "report_id": lost_item.get("report_id"),
            "image": lost_item.get("image"),
            "image_url": lost_item.get("image_url"),
            "title": lost_item.get("title"),
            "location": lost_item.get("location", {}),
            "category": lost_item.get("category"),
            **scores
        })

    results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return results[:TOP_K]