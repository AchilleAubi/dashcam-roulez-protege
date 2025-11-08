from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

def analyser_driver_mediapipe(image_path):
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    actions = [
        "une personne qui conduit",
        "une personne qui dort",
        "une personne qui mange",
        "une personne qui téléphone",
        "une personne qui lit",
        "une personne qui tape sur un clavier",
    ]

    img = Image.open(image_path).convert("RGB")
    inputs = processor(text=actions, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
        probs = logits.softmax(dim=-1).squeeze()

    best = int(probs.argmax())
    return best

if __name__ == "__main__":
    path = "driver.jpg"  # remplace par ton image
    out = analyser_driver_mediapipe(path)
    print("~~~analyser_driver_mediapipe~~~", out)
