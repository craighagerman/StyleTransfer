import torch
import clip
from PIL import Image



class ClipImageEncoder:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # model, preprocess = clip.load("ViT-B/32", device=device)
        model, preprocess = clip.load("ViT-L/14", device=device)
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)




def load_image(path):
    return preprocess(Image.open(path)).unsqueeze(0).to(device)


image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


def main():
    pass

# ------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()

