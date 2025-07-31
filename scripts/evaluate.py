import torch
from sklearn.metrics import f1_score


def evaluate(model, dataloader, device='cuda'):
    model.eval()
    correct = total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch, _, _, _ in dataloader:
            x_batch = x_batch.to(device, torch.float32)
            y_batch = y_batch.to(device)

            
            logits = model.predict_number(x_batch)  
            probs = torch.softmax(logits, dim=1)  
            preds = torch.argmax(probs, dim=1)  


            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)


            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())


    accuracy = 100.0 * correct / total 
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, f1
