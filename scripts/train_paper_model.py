import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from scripts.dataloaders.train_loader import get_train_dataloader
from scripts.dataloaders.test_loader import get_test_dataloader
from scripts.evaluate import evaluate
from models.paper_model import Convy
import yaml

with open("../config/train_hyperparameters.yaml","r") as f:
    hyperparameters = yaml.safe_load(f)


model = Convy()
max_acc = 0
max_acc = 0
optimizer_f = optim.Adam(model.f_parameters(), lr=hyperparameters["learning_rate"],weight_decay=1e-3)
optimizer_g = optim.Adam(model.g_parameters(), lr=hyperparameters["learning_rate"],weight_decay=1e-3)
optimizer_h = optim.Adam(model.h_parameters(), lr=hyperparameters["learning_rate"],weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = get_train_dataloader()
test_dataloader = get_test_dataloader()

def train(train_dataloader,lambda_,mu):
    number_of_samples = 0
    batch_loss = 0
    batch_loss_pred=0
    model.to(device)
    model.train()

    for batch_idx, (input, target_label, bias_red, bias_green, bias_blue) in enumerate(train_dataloader):
        input, target_label, bias_red, bias_green, bias_blue = input.to(device), target_label.to(device), bias_red.to(device), bias_green.to(device), bias_blue.to(device)

        optimizer_h.zero_grad()
        optimizer_f.zero_grad()
        bias_preds_red = model.predict_bias_inv_r(input)
        bias_preds_green = model.predict_bias_inv_g(input)
        bias_preds_blue = model.predict_bias_inv_b(input)
        loss_bias_h_r = criterion(bias_preds_red, bias_red)
        loss_bias_h_g = criterion(bias_preds_green, bias_green)
        loss_bias_h_b = criterion(bias_preds_blue, bias_blue)
        loss_bias_h = (loss_bias_h_r + loss_bias_h_g + loss_bias_h_b)*(mu/3)
        loss_bias_h.backward()
        optimizer_h.step()
        optimizer_f.step()

        
        optimizer_f.zero_grad()
        bias_probs_r = F.softmax(model.predict_bias_r(input), dim=1)
        entropy_r = torch.sum(bias_probs_r * torch.log(bias_probs_r + 1e-8), dim=1).mean()
        loss_entropy_r = lambda_ * entropy_r

        bias_probs_g = F.softmax(model.predict_bias_g(input), dim=1)
        entropy_g = torch.sum(bias_probs_g * torch.log(bias_probs_g + 1e-8), dim=1).mean()
        loss_entropy_g = lambda_ * entropy_g

        bias_probs_b = F.softmax(model.predict_bias_b(input), dim=1)
        entropy_b = torch.sum(bias_probs_b * torch.log(bias_probs_b+ 1e-8), dim=1).mean()
        loss_entropy_b = lambda_ * entropy_b
 
        loss_entropy = (loss_entropy_r + loss_entropy_g + loss_entropy_b) / 3
        loss_entropy.backward()
        optimizer_f.step()


        optimizer_f.zero_grad()
        optimizer_g.zero_grad()
        outputs = model.predict_number(input)
        loss_main = criterion(outputs, target_label)
        loss_main.backward()
        optimizer_f.step()
        optimizer_g.step()

        number_of_samples += 1
        batch_loss += loss_main.item() + loss_bias_h.item() + loss_entropy.item()
        batch_loss_pred += loss_main.item()

    return batch_loss, batch_loss_pred


for _ in range(20):
    train_batches_error = []
    train(train_dataloader,hyperparameters["lambda"] ,hyperparameters["mu"])
    test_acc, train_f1= evaluate(model, test_dataloader)
    print(test_acc)
    if test_acc > max_acc:
        max_acc = test_acc
        torch.save(model.state_dict(), hyperparameters["model_paper_path"])
