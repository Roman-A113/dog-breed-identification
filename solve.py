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
BEST_MODEL_PATH = 'best_model.pth'

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMAGE_SIZE = 300


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
        img_name = f"{img_id}.jpg" if not str(img_id).endswith('.jpg') else img_id
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image
        else:
            label = self.df.iloc[idx]['target']
            return image, torch.tensor(label, dtype=torch.long)


def prepare_dataframes():
    df_labels = pd.read_csv(LABELS_CSV)
    df_sample = pd.read_csv(SAMPLE_SUB_CSV)

    breed_columns = list(df_sample.columns[1:])
    breed_to_idx = {breed: i for i, breed in enumerate(breed_columns)}
    df_labels['target'] = df_labels['breed'].map(breed_to_idx)

    train_df, valid_df = train_test_split(
        df_labels,
        test_size=0.2,
        random_state=42,
        stratify=df_labels['target']
    )
    return train_df, valid_df, df_sample


def prepare_transforms():
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
    return train_transforms, val_transforms


def prepare_data_loaders(train_df, valid_df, test_df, train_transforms, val_transforms):
    train_dataset = DogDataset(train_df, TRAIN_DIR, transform=train_transforms)
    valid_dataset = DogDataset(valid_df, TRAIN_DIR, transform=val_transforms)
    test_dataset = DogDataset(test_df, TEST_DIR, transform=val_transforms, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader


def prepare_model(device, test_df):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    num_classes = len(test_df.columns) - 1
    model.classifier[1] = nn.Linear(num_features, num_classes)

    model = model.to(device)
    return model


def valid_epoch(model, valid_loader, device, criterion, epoch, current_stage_epochs, stage_name):
    model.eval()
    correct = 0
    total = 0
    val_running_loss = 0.0

    desc_str = f"[{stage_name}] Эпоха {epoch+1}/{current_stage_epochs} [Валидация]"
    valid_progress_bar = tqdm(valid_loader, desc=desc_str, leave=False, unit="batch")
    with torch.no_grad():
        for images, labels in valid_progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(valid_loader.dataset)
    accuracy = 100 * correct / total
    return val_loss, accuracy


def train_epoch(model, train_loader, device, criterion, optimizer, epoch, current_stage_epochs, stage_name):
    model.train()
    running_loss = 0.0

    desc_str = f"[{stage_name}] Эпоха {epoch+1}/{current_stage_epochs} [Обучение]"
    train_progress_bar = tqdm(train_loader, desc=desc_str, leave=False, unit="batch")
    for images, labels in train_progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def train_model(model, train_loader, valid_loader, device, optimizer, epochs, stage_name, scheduler=None):
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = train_epoch(model, train_loader, device, criterion, optimizer, epoch, epochs, stage_name)
        valid_loss, accuracy = valid_epoch(model, valid_loader, device, criterion, epoch, epochs, stage_name)

        if scheduler:
            scheduler.step()

        tqdm.write(
            f"[{stage_name}] Эпоха {epoch+1}/{epochs} - Train Loss: {epoch_loss:.5f}, Val Loss: {valid_loss:.5f}, Val Acc: {accuracy:.2f}%"
        )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    tqdm.write(f"[{stage_name}] Лучший loss на валидации: {best_val_loss:.5f}\n")


def test_model(device, test_loader, model):
    model.eval()
    all_preds = []
    test_bar = tqdm(test_loader, desc="Тестирование", leave=True, unit="batch")
    with torch.no_grad():
        for images in test_bar:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            all_preds.append(probabilities.cpu().numpy())

    all_preds = np.vstack(all_preds)
    return all_preds


def prepare_submission_file(all_preds, test_df):
    submission_df = pd.DataFrame(all_preds, columns=test_df.columns[1:])
    submission_df.insert(0, 'id', test_df['id'].values)
    submission_df.to_csv(OUTPUT_CSV, index=False)
    tqdm.write(f"Файл предсказаний сохранен в '{OUTPUT_CSV}'")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tqdm.write(f"Используемое устройство: {device}")

    train_df, valid_df, test_df = prepare_dataframes()
    train_transforms, valid_transforms = prepare_transforms()
    train_loader, valid_loader, test_loader = prepare_data_loaders(
        train_df, valid_df, test_df, train_transforms, valid_transforms)

    model = prepare_model(device, test_df)

    optimizer_stage1 = optim.AdamW(model.classifier.parameters(), lr=1e-3)

    train_model(model, train_loader, valid_loader, device,
                optimizer=optimizer_stage1, epochs=2, stage_name="Этап 1")

    for param in model.parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=4)
    train_model(model, train_loader, valid_loader, device,
                optimizer=optimizer_stage2, epochs=4, stage_name="Этап 2", scheduler=scheduler)

    tqdm.write("Загрузка лучшей сохраненной модели...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))

    all_preds = test_model(device, test_loader, model)
    prepare_submission_file(all_preds, test_df)
