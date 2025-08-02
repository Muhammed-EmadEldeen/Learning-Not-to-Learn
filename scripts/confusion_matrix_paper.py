import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scripts.dataloaders.test_loader import get_test_dataloader
from models.paper_model import Convy
import yaml

with open("./config/train_hyperparameters.yaml","r") as f:
    hyperparameters = yaml.safe_load(f)


model = Convy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(hyperparameters["model_paper_path"],map_location=device))
test_dataloader = get_test_dataloader()


all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        images, labels, _, _, _ = batch  
        images, labels = images.to(device), labels.to(device)
        
        # Get predictions using predict_number
        outputs = model.predict_number(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


cm = confusion_matrix(all_labels, all_preds)


plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
