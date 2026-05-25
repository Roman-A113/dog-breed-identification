import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

TRAIN_DIR = 'train'
TEST_DIR = 'test'
LABELS_CSV = 'labels.csv'
SAMPLE_SUB_CSV = 'sample_submission.csv'
OUTPUT_CSV = 'my_submission.csv'

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224


class DogDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id']
        # Если в датафрейме id без расширения, добавляем '.jpg'
        img_name = f"{img_id}.jpg" if not str(img_id).endswith('.jpg') else img_id
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_id
        else:
            label = self.df.iloc[idx]['target']
            return image, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")

    df_labels = pd.read_csv(LABELS_CSV)
    df_sample = pd.read_csv(SAMPLE_SUB_CSV)

    breed_cols = list(df_sample.columns[1:])
    breed_to_idx = {breed: i for i, breed in enumerate(breed_cols)}

    df_labels['target'] = df_labels['breed'].map(breed_to_idx)

    train_df, val_df = train_test_split(df_labels, test_size=0.2, random_state=42, stratify=df_labels['target'])

    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DogDataset(train_df, TRAIN_DIR, transform=train_transforms)
    val_dataset = DogDataset(val_df, TRAIN_DIR, transform=val_transforms)
    test_dataset = DogDataset(df_sample, TEST_DIR, transform=val_transforms, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(breed_cols))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    epoch_bar = tqdm(range(EPOCHS), desc="Всего эпох", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{EPOCHS} [Обучение]", leave=False, unit="batch")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            train_bar.set_postfix(current_loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0

        val_bar = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{EPOCHS} [Валидация]", leave=False, unit="batch")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        epoch_bar.set_postfix(Loss=f"{epoch_loss:.4f}", Val_Acc=f"{accuracy:.2f}%")

    print("Обучение завершено успешно!")

    model.eval()
    test_ids = []
    all_preds = []

    print("Начинаем предсказание для тестовой выборки...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            all_preds.append(probabilities.cpu().numpy())
            test_ids.extend(ids)

    all_preds = np.vstack(all_preds)

    submission_df = pd.DataFrame(all_preds, columns=breed_cols)
    submission_df.insert(0, 'id', [os.path.splitext(f)[0] for f in test_ids])
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Файл ответов сохранен как '{OUTPUT_CSV}'. Готов к отправке!")
