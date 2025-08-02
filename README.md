# 🧠 Learning-Not-to-Learn
Implementation of the "Learning Not to Learn" Paper on Colored MNIST Dataset

## 📑 Table of Contents

- [Problem Definition](#problem-definition)
- [Project Structure](#project-structure)
- [Procedure](#procedure)
- [How to Use](#how-to-use)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)

---

## Problem Definition
This project implements the "Learning Not to Learn" paper, focusing on mitigating bias in the colored MNIST dataset. The goal is to train a model that unlearns color-based biases present in the training data, improving accuracy on an unbiased test dataset. The baseline model achieves 68% accuracy, while the proposed method increases it to 90%.
<img width="368" height="372" alt="image" src="https://github.com/user-attachments/assets/26e28905-abaf-45cd-a8d8-4cff219c8a32" />

  
---


## Project Structure
The project is organized as follows:
```
Learning-Not-to-Learn/
├── config/                          # Configuration files
│   └── train_hyperparameters.yaml   # Hyperparameter configurations
├── models/                          # Model definitions and checkpoints
│   ├── baseline_model.py            # Baseline model definition
│   ├── model-saved/                 # Saved model checkpoints
│   │   ├── model_base.pt           # Baseline model checkpoint
│   │   └── model_paper.pt          # Paper model checkpoint
│   ├── paper_model.py               # Paper model definition
│   └── __pycache__/                 # Compiled Python files
├── notebook/                        # Jupyter notebooks for exploration
│   └── colored-mnist-new.ipynb      # Main notebook for analysis
├── scripts/                         # Python scripts for project tasks
│   ├── confusion_matrix_base.py     # Confusion matrix for baseline model
│   ├── confusion_matrix_paper.py    # Confusion matrix for paper model
│   ├── dataloaders/                 # Data loading scripts
│   │   ├── test_loader.py           # Test dataset loader
│   │   ├── train_loader.py          # Train dataset loader
│   │   └── __pycache__/             # Compiled Python files
│   ├── evaluate.py                  # Evaluation script
│   ├── mnist_download.py            # Script to download MNIST dataset
│   ├── train_base.py                # Baseline model training script
│   └── train_paper_model.py         # Paper model training script
├── README.md                        # Project documentation
├── .gitignore                       # Git ignore file
└── __init__.py                      # Python package initialization
```

---

## Procedure

1. Dataset Creation  

        - Training Dataset: Each digit is assigned a mean color to introduce bias.  
        - Test Dataset: Colors are assigned randomly to ensure no bias.


2. Model Development  

        - Implemented the baseline and paper models as described in the "Learning Not to Learn" paper.


3. Training and Hyperparameter Tuning  

        - Defined custom loss functions to unlearn bias.  
        - Tuned hyperparameters to optimize model performance.


4. Evaluation  

        - Evaluated model performance using accuracy and confusion matrices.  
        - Visualized results to compare baseline (68% accuracy) and paper model (90% accuracy).

---

## How to Use
1. Clone the repository
```bash
git clone https://github.com/your-username/learning-not-to-learn.git
cd learning-not-to-learn
```
2. Download the MNIST dataset
```bash
python scripts/mnist_download.py
```
3. Install dependencies
Ensure you have Python installed, then run:
```
pip install -r requirements.txt
```
4. Train the models
Train the baseline model:
```bash
python scripts/train_base.py
```

Train the paper model:
```bash
python scripts/train_paper_model.py
```
5. Evaluate and visualize results

- Generate confusion matrix for the baseline model:
```bash
python scripts/confusion_matrix_base.py
```

- Generate confusion matrix for the paper model:
```bash
python scripts/confusion_matrix_paper.py
```


## Results and Visualizations
### Confusion Matrix of Baseline Model (68% Accuracy)
<img width="800" height="717" alt="image" src="https://github.com/user-attachments/assets/406d3fdb-72b3-4840-b690-b4f5adeeacdd" />

### Confusion Matrix of Paper Model (90% Accuracy)
<img width="813" height="696" alt="image" src="https://github.com/user-attachments/assets/72fc1d94-70bd-4801-a8de-221ea9594f12" />

---

## Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure your code follows the project’s coding standards and includes appropriate tests.
