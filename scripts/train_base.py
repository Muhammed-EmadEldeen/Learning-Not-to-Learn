import torch
import torch.nn as nn
import torch.optim as optim
from models.baseline_model import Convy_base
from scripts.dataloaders.train_loader import get_train_dataloader
from scripts.dataloaders.test_loader import get_test_dataloader
from scripts.evaluate import evaluate

model_base = Convy_base()
optimizer_base = optim.Adam(model_base.parameters(), lr=1e-3,weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
max_acc_base = 0
device = torch.device("cuda")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def train(train_dataloader):
    number_of_samples = 0
    batch_loss = 0
    batch_loss_pred=0
    model_base.to(device)
    model_base.train()

    for batch_idx, (input, target_label, _, _, _) in enumerate(train_dataloader):
        input, target_label = input.to(device), target_label.to(device)


        optimizer_base.zero_grad()
        outputs = model_base(input)
        loss_main = criterion(outputs, target_label)
        loss_main.backward()
        optimizer_base.step()

        number_of_samples += 1
        batch_loss += loss_main.item()

    return batch_loss

if __name__ == "__main__":
    train_dataloader = get_train_dataloader()
    test_dataloader = get_test_dataloader()
    for _ in range(20):
        train_batches_error = []
        train(train_dataloader)
        test_acc, train_f1= evaluate(model_base, test_dataloader)
        print(test_acc)
        if test_acc > max_acc_base:
            max_acc_base = test_acc
            torch.save(model_base.state_dict(), "best_model_mnist_base.pt")
