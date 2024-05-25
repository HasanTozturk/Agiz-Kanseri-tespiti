import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

# GPU kullanılabilirse CUDA'yı seç, aksi halde CPU kullan
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri setinin yolu
data_dir = 'dataset/'

# Veri ön işleme ve artırma işlemleri tanımlama
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Veri setini yükleme
image_datasets = {x: datasets.ImageFolder(data_dir + x, data_transforms[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val', 'test']}

# EfficientNetB5 modelini yükleme
efficientnet_model = EfficientNet.from_pretrained('efficientnet-b5')

# Modelin çıktı katmanını değiştirme
num_features = efficientnet_model._fc.in_features
efficientnet_model._fc = nn.Sequential(
    nn.Dropout(0.5),  # Daha önceki çıktı katmanının dropout oranı
    nn.Linear(num_features, 2)  # İki sınıf için yeni çıktı katmanı
)

# Modeli CUDA'ya taşıma (GPU)
efficientnet_model = efficientnet_model.to(device)

# Optimizasyon algoritmasını tanımlama
optimizer = optim.SGD(efficientnet_model.parameters(), lr=0.001, momentum=0.9)

# Kayıp fonksiyonunu tanımlama (binary cross-entropy loss)
criterion = nn.CrossEntropyLoss()

# Eğitim fonksiyonunu tanımlama
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Her bir eğitim ve doğrulama aşaması için
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modeli eğitim modunda ayarlama
            else:
                model.eval()   # Modeli değerlendirme modunda ayarlama

            running_loss = 0.0
            running_corrects = 0

            # Veri kümesindeki her veri yığını için
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):  # tqdm ile döngüyü sar
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Gradyanları sıfırla
                optimizer.zero_grad()

                # İleri yayılım
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Eğitim aşamasında geriye yayılım ve optimize etme
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # İstatistikleri topla
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

# Modeli eğitme (epoch sayısı 10 olarak ayarlandı)
train_model(efficientnet_model, criterion, optimizer, num_epochs=10)

# Test fonksiyonunu tanımlama
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print('Test accuracy: {:.2f}%'.format(100 * correct / total))

    # Sınıflandırma raporunu oluşturma
    target_names = ['normal', 'oscc']
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

    # Sınıflandırma matrisini oluşturma
    cm = confusion_matrix(true_labels, predicted_labels)
    print('Confusion Matrix:')
    print(cm)

# Test veri kümesi için modeli değerlendirme
evaluate_model(efficientnet_model, dataloaders['test'])
