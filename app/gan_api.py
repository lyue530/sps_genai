import io
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
from torchvision.utils import make_grid
from app.gan_model import Generator, LATENT_DIM

router = APIRouter(prefix="/gan", tags=["GAN"])
WEIGHTS_PATH = Path(__file__).resolve().parent / "generator.pt"

@router.get("/status")
def status():
    return {"model": "MNIST GAN", "weights_available": WEIGHTS_PATH.exists()}

@router.get("/generate")
def generate(count: int = Query(default=4, ge=1, le=16)):
    if not WEIGHTS_PATH.is_file():
        raise HTTPException(503, "Train your GAN first to create generator.pt")
    generator = Generator().to("cpu")
    generator.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    generator.eval()
    with torch.no_grad():
        images = (generator(torch.randn(count, LATENT_DIM)) + 1) / 2
        grid = make_grid(images.clamp(0, 1), nrow=min(count, 4))
    array = (grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    output = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(output, format="PNG")
    return Response(output.getvalue(), media_type="image/png")
