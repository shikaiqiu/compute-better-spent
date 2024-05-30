import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_meter = AverageMeter()
    correct, total_n = 0, 0

    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), x.shape[0])
        preds = torch.argmax(logits, dim=1)
        correct += preds.eq(y).sum().item()
        total_n += y.shape[0]

    return {"loss": loss_meter.avg, "acc": 100 * (correct / total_n)}


def eval_model(model, val_loader, device):
    model.eval()
    correct, total_n = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(y).sum().item()
            total_n += y.shape[0]
    return {"acc": 100 * (correct / total_n)}
