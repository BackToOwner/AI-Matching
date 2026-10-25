#  Lost & Found AI Matching Service

AI-powered matching service for the Lost & Found application.

This service compares a user's **lost-item report** against registered **found-item reports** using multimodal embeddings. It combines information from the item's **image** and **text/title** to identify the most likely matches.


## Overview

The AI matching system receives information about a lost item and compares it with available found items.

### Matching flow

```text
                 Lost Item
                     │
          ┌──────────┴──────────┐
          │                     │
        Image            Structured Fields
          │                     │
          ▼                     ▼
       SigLIP              Text Input
       Image                ┌────┴────┐
      Embedding              │         │
          │                Title     Category
          │                Color     Location
          │                  │         │
          │                  └────┬────┘
          │                       ▼
          │                 SigLIP Text
          │                  Embedding
          │                       │
          └──────────┬────────────┘
                     ▼
             Hybrid Matching
                     │
          ┌──────────┴──────────┐
          │                     │
    Image Similarity      Text Similarity
          │                     │
          └──────────┬──────────┘
                     ▼
               Final Score
                     │
                     ▼
              Ranked Matches
```

The system uses both visual and textual information instead of relying only on one source.


# 🤖AI Model- SigLIP

The project uses:

```text
google/siglip-base-patch16-224
```

SigLIP is a multimodal vision-language model capable of representing both:

* Images
* Text

in a shared embedding space.

This allows the system to compare:

```text
Lost Item Image ↔ Found Item Image
```

and:

```text
Lost Item Title ↔ Found Item Title
```


# Matching Strategy

The system generates embeddings for both the lost item and found items.

### 1. Image similarity

The system converts the uploaded lost-item image into an image embedding.

Each found-item image is also converted into an image embedding.

Cosine similarity is then used to determine how visually similar the images are.

```text
Image Similarity
=
Cosine Similarity(
    Lost Image Embedding,
    Found Image Embedding
)
```


### 2. Text similarity

Convert the lost item's title into a text embedding.

For example:

```text
Silver HP Envy Laptop
```

Found-item titles can be compared against it:

```text
Black HP Envy Laptop
Silver HP Laptop
Gold HP Laptop
Maroon Backpack
```

The model calculates the semantic similarity between the titles.



### 3. Hybrid score

The final matching score combines image and text similarity.

Current weighting:

```text
Final Score =
    60% Image Similarity
    +
    40% Text Similarity
```

or:

```python
final_score = (
    0.6 * image_similarity
    +
    0.4 * text_similarity
)
```

This allows the system to consider both:

* What the item looks like
* What the user says the item is


# Project Structure

```text
ai_service/
│
├── main.py
├── matcher.py
├── lost_item_service.py
├── found_item_service.py
├── database.py
│
├── requirements.txt
├── .env
├── .gitignore
│
└── README.md
```


# 📁 File Responsibilities

## `main.py`

Main FastAPI application.

Responsible for:

* Starting the API
* Receiving matching requests
* Returning matching results
* Connecting the different services



## `matcher.py`

Core AI matching logic.

Responsible for:

* Loading the SigLIP model
* Creating image embeddings
* Creating text embeddings
* Normalizing embeddings
* Calculating similarity
* Combining image and text scores
* Ranking results



## `lost_item_service.py`

Handles lost-item processing.

Responsible for:

* Receiving lost-item information
* Processing the uploaded image
* Creating embeddings
* Preparing data for matching



## `found_item_service.py`

Handles found-item data.

Responsible for:

* Retrieving found items
* Processing found-item images
* Generating/storing embeddings
* Preparing found items for comparison



## `database.py`

Handles communication with the application's database/storage layer.

It can be used to:

* Retrieve found-item records
* Store embeddings
* Retrieve item metadata
* Connect the AI service with the main application backend


# ⚙️ Technologies

### Backend

* Python
* FastAPI

### AI / Machine Learning

* PyTorch
* Hugging Face Transformers
* SigLIP
* NumPy
* PIL

### Database

* Supabase
* PostgreSQL
* pgvector

### Application

* Flutter
* Firebase
* Node.js
* React


#  Installation

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ai_service
```



## 2. Create a Conda environment

```bash
conda create -n ai python=3.11
```

Activate it:

```bash
conda activate ai
```


## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Example dependencies:

```text
fastapi
uvicorn
torch
torchvision
transformers
pillow
numpy
python-dotenv
supabase
```


#  Environment Variables

Create a `.env` file:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

Do not commit `.env` to GitHub.

Add it to `.gitignore`:

```text
.env
__pycache__/
*.pyc
```

# Model Loading

The service uses:

```python
MODEL_NAME = "google/siglip-base-patch16-224"
```

The model is automatically downloaded from Hugging Face the first time it is loaded.

After downloading, the model can be reused locally.


# Embedding Generation

## Image embedding

The image is processed by SigLIP and converted into a vector representation.

Conceptually:

```text
Image
  ↓
SigLIP
  ↓
Image Embedding
  ↓
Normalization
```

## Text embedding

The item title is processed using the text side of SigLIP.

Example:

```text
"Silver HP Envy Laptop"
        ↓
      SigLIP
        ↓
 Text Embedding
```


#  Cosine Similarity

The system uses cosine similarity to compare embeddings.

Conceptually:

```text
Similarity(A, B)
=
(A · B)
/
(||A|| ||B||)
```

Higher values indicate greater similarity.

Example:

```text
0.90 → Very strong similarity
0.80 → Strong similarity
0.70 → Moderate similarity
0.50 → Weak similarity
```

These values should be treated as ranking scores rather than guaranteed probabilities.


#  Ranking

After calculating the hybrid score for each found item:

```text
Found Item A → 0.91
Found Item B → 0.83
Found Item C → 0.76
Found Item D → 0.62
Found Item E → 0.51
```

The system sorts the results in descending order:

```text
1. Found Item A
2. Found Item B
3. Found Item C
4. Found Item D
5. Found Item E
```

The application can display the top `K` matches to the user.




