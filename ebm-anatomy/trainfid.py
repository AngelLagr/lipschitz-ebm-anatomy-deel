import os
import torch as t
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utilfid import SimpleFeatureExtractor


class MNISTClassifier(nn.Module):
    def __init__(self, device):
        super(MNISTClassifier, self).__init__()
        self.backbone = SimpleFeatureExtractor(device)
        self.head = nn.Linear(256, 10)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    t.manual_seed(0)

    out_dir = './fidcheckpoints'
    out_path = os.path.join(out_dir, 'mnist_feature_extractor.pth')
    os.makedirs(out_dir, exist_ok=True)

    model = MNISTClassifier(device).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_set = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transforms.ToTensor())
    train_loader = t.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

    epochs = 50
    print('Training MNIST feature extractor on {}'.format(device))
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        print('Epoch {}/{} - loss {:.4f} - acc {:.2f}%'.format(
            epoch + 1, epochs, running_loss / len(train_loader), 100.0 * correct / total
        ))

    t.save(model.backbone.state_dict(), out_path)
    print('Saved pretrained feature extractor to {}'.format(out_path))


if __name__ == '__main__':
    main()
