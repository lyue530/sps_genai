
from pathlib import Path
import io

import spacy
import torch
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.cnn_model import AssignmentCNN
from app.gan_api import router as gan_router

# Initialize FastAPI
app = FastAPI()

# Register GAN image generation API
app.include_router(gan_router)

# =====================================
# Load spaCy model
# =====================================

nlp = spacy.load("en_core_web_lg")


# Word embedding function
def calculate_embedding(input_word):
    word = nlp(input_word)
    return word.vector


# =====================================
# Bigram text generation
# =====================================

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


# =====================================
# Request models
# =====================================

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


# =====================================
# CNN Image Classification
# =====================================

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

device = torch.device("cpu")

cnn_model = AssignmentCNN().to(device)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "assignment2_cnn.pth"
)

cnn_model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

cnn_model.eval()


# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# =====================================
# Root endpoint
# =====================================

@app.get("/")
def read_root():
    return {"Hello": "World"}


# =====================================
# Bigram text generation endpoint
# =====================================

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}


# =====================================
# Word embedding endpoint
# =====================================

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)
    return {
        "word": request.word,
        "embedding": embedding.tolist()
    }


# =====================================
# CNN image classification endpoint
# =====================================

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:
        image_bytes = await file.read()
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    image_tensor = (
        image_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        outputs = cnn_model(image_tensor)

        probabilities = torch.softmax(
            outputs, dim=1
        )

        confidence, predicted = torch.max(
            probabilities, 1
        )

    return {
        "predicted_class": CLASSES[predicted.item()],
        "confidence": round(confidence.item(), 4)
    }
