import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SamProcessor, SamModel
from dataset import get_bounding_box
from utils import show_mask
from datasets import load_dataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"


    dataset = load_dataset("nielsr/breast-cancer", split="train")

    idx = np.random.randint(0, len(dataset) -1 )
    image = dataset[idx]["image"]
    ground_truth_mask = np.array(dataset[idx]["label"])
    prompt = get_bounding_box(ground_truth_mask)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("./FineTuned_SAM").to(device)

    inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    prediction_mask = torch.sigmoid(outputs.pred_masks.squeeze(1))
    prediction_mask = (prediction_mask.cpu().numpy().squeeze() > 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.array(image))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.array(image))
    show_mask(prediction_mask, axes[1])
    axes[1].set_title("Predicted mask")
    axes[1].axis("off")

    axes[2].imshow(np.array(image))
    show_mask(ground_truth_mask, axes[2])
    axes[2].set_title("Ground truth")
    axes[2].axis("off")

    plt.savefig("testing.png")
    plt.show()

if __name__ == "__main__":
    main()