import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
from dataset import SAMDataset
from datasets import load_dataset
import monai
from torch.optim import Adam
from tqdm import tqdm
from statistics import mean
import yaml


def main():

    with open("config.yaml") as f:
        syaml = yaml.safe_load(f)


    print("Loading dataset...")

    dataset = load_dataset(syaml["dataset"]["name"], split=syaml["dataset"]["split"])

    print(f"Loaded {len(dataset)} samples.")

    processor = SamProcessor.from_pretrained(syaml["model"]["name"])

    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.resample = 2

    train_dataset = SAMDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=syaml["training"]["batch_size"], shuffle=True)

    model = SamModel.from_pretrained(syaml["model"]["name"])

    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = Adam(model.mask_decoder.parameters(), lr=syaml["training"]["lr"], weight_decay=syaml["training"]["weight_decay"])
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    all_epoch_losses = []

    model.train()
    for epoch in range(syaml["training"]["epochs"]):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False
            )
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            epoch_losses.append(loss_item)
            all_epoch_losses.append(loss_item)

        print(f'EPOCH: {epoch} | Mean loss: {mean(epoch_losses):.4f}')

    plt.figure(figsize=(8, 4))
    plt.plot(all_epoch_losses, label="Batch Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(syaml["output"]["loss_plot_path"], dpi=150)
    plt.show()

    model.save_pretrained("./FineTuned_SAM")

if __name__ == "__main__":
    main()


