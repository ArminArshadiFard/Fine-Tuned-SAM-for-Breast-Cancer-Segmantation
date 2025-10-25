## Fine-Tuned-SAM-for-Breast-Cancer-Segmantation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

This project investigates the fine-tuning of the Segment Anything Model (SAM) on a small-scale medical imaging dataset—specifically, histopathological images of breast cancer—to improve its segmentation performance.

## Why Fine-Tune SAM?
Although SAM demonstrates strong zero-shot segmentation capabilities across diverse natural image domains, its performance on medical images can be suboptimal due to:
- Domain Shift: Since SAM was Trained on general RGB images it will be challengning for it to segment medical images with high accuracy.
- Relying to Prompt: SAM relies on its Prompting capablities in order to segment everything which results into lower accuacy in fine structures.
- Even training for Few Epochs like 50 shows improvements
- **Goal**: Enhance the accuracy of SAM for medical image segmentation through minimal, fine-tuning while adapting to domain-specific requirements.

## 📌  Features
-  The goal is to segment the Tumor in the given picture
-  **the dataset has a total of 130 images which is divided test and train
-  **To Test** – Run the interface.py app to segment random images
-  **Low Training epochs**: I Trained the model only for 20 epochs and here are the results!


## 📈 Results (ResNet18, 15 epochs)
<img width="1200" height="600" alt="TRAIN_LOSS" src="https://github.com/user-attachments/assets/323eb4ae-3fc1-4474-81b1-cdb2668d7025" />


- Since SAM takes so much time Training i only used 20 epochs in this case.
- It would be best to train for longer to improve the results.

  
## Some Test Examples
<img width="1500" height="500" alt="testing" src="https://github.com/user-attachments/assets/c1351e38-2ee3-400f-b8be-45a0ee94b0cb" />
