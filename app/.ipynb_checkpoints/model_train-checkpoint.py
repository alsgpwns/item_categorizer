import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import time
import argparse
import os

# 모델 클래스 정의
class CategoryModel(nn.Module):
    """
    Desc : BERT 기반 분류 모델 클래스.
    Args :
        num_classes (int): 출력 클래스 수.
        num_hidden_layers (int): 사용할 BERT의 히든 레이어 수. 기본값 12.
        dropout_rate (float): 드롭아웃 비율. 기본값 0.3.
    """
    def __init__(self, num_classes, num_hidden_layers=12, dropout_rate=0.3):
        super(CategoryModel, self).__init__()
        self.bert = AutoModel.from_pretrained("klue/roberta-base")
        
        # BERT의 레이어 수 조정
        if num_hidden_layers < len(self.bert.encoder.layer):
            self.bert.encoder.layer = nn.ModuleList(self.bert.encoder.layer[:num_hidden_layers])
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Desc : BERT 모델의 순전파.
        Args :
            input_ids (torch.Tensor): 입력 토큰 ID.
            attention_mask (torch.Tensor): 어텐션 마스크.
        Returns :
            torch.Tensor: 분류 결과.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

# 데이터셋 클래스 정의
class CategoryDataset(Dataset):
    """
    Desc : 분류 모델 학습을 위한 데이터셋 클래스.
    Args :
        texts (list): 텍스트 데이터 리스트.
        labels (list): 라벨 데이터 리스트.
        tokenizer (AutoTokenizer): HuggingFace 토크나이저 객체.
        max_len (int): 입력 텍스트의 최대 길이.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Desc : 데이터셋의 총 샘플 수 반환.
        Returns :
            int: 데이터셋 크기.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Desc : 데이터셋의 특정 샘플 반환.
        Args :
            idx (int): 샘플 인덱스.
        Returns :
            dict: 토큰화된 입력 및 라벨 정보.
        """
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Optimizer 설정 함수
def get_optimizer(optimizer_name, model, learning_rate):
    """
    Desc : 최적화 알고리즘 설정.
    Args :
        optimizer_name (str): 최적화 알고리즘 이름 ('adam', 'adamw', 'sgd' 등).
        model (nn.Module): 학습할 모델.
        learning_rate (float): 학습률.
    Returns :
        torch.optim.Optimizer: 설정된 최적화 알고리즘.
    """
    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# 모델 정보 업데이트 함수
def update_model_info(class_type, model_save_path, label_encoder_path, model_info_path="./config/model_info.json"):
    """
    Desc : model_info.json 파일에 학습된 모델 경로 정보 업데이트.
    Args :
        class_type (str): 분류 유형 ('big_cate', 'mid_cate', 'small_cate').
        model_save_path (str): 저장된 모델 경로.
        label_encoder_path (str): 저장된 라벨 인코더 경로.
        model_info_path (str): 모델 정보 JSON 파일 경로.
    """
    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
    else:
        model_info = {}

    model_info[class_type] = {
        "model_path": model_save_path,
        "label_path": label_encoder_path
    }

    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    print(f"Model info updated in {model_info_path}")

# 모델 학습 함수
def train_category_model(config_path, dataset_path, text_column, class_type, model_save_path, label_encoder_path):
    """
    Desc : 분류 모델 학습 실행.
    Args :
        config_path (str): 학습 설정 JSON 파일 경로.
        dataset_path (str): 학습 데이터 CSV 파일 경로.
        text_column (str): 텍스트 데이터 열 이름.
        class_type (str): 학습할 카테고리 유형.
        model_save_path (str): 학습된 모델 저장 경로.
        label_encoder_path (str): 라벨 인코더 저장 경로.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    df = pd.read_csv(dataset_path)

    label_encoder = LabelEncoder()
    df[f'{class_type}_encoded'] = label_encoder.fit_transform(df[class_type])
    joblib.dump(label_encoder, label_encoder_path)

    texts = df[text_column].values
    labels = df[f'{class_type}_encoded'].values

    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')

    dataset = CategoryDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=data_config["max_len"]
    )
    data_loader = DataLoader(dataset, batch_size=training_config["batch_size"], shuffle=True)

    num_classes = len(label_encoder.classes_)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CategoryModel(
        num_classes=num_classes,
        num_hidden_layers=model_config.get("num_hidden_layers", 12),
        dropout_rate=model_config.get("dropout_rate", 0.3)
    )
    model = model.to(device)

    optimizer = get_optimizer(
        optimizer_name=training_config.get("optimizer", "Adam"),
        model=model,
        learning_rate=training_config["learning_rate"]
    )
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()
    model.train()
    for epoch in range(training_config["epochs"]):
        total_loss = 0
        correct_predictions = 0
        n_examples = len(dataset)

        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{training_config['epochs']}, Loss: {total_loss / n_examples}, Accuracy: {correct_predictions.double() / n_examples}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    update_model_info(class_type, model_save_path, label_encoder_path)


# argparse 설정
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="카테고리 분류 모델 학습 실행 스크립트")
    parser.add_argument('--config_path', type=str, required=True, help="설정 파일 경로 (JSON 형식)")
    parser.add_argument('--dataset_path', type=str, required=True, help="학습 데이터셋 경로 (CSV 파일)")
    parser.add_argument('--text_column', type=str, required=True, help="텍스트 데이터가 있는 열 이름")
    parser.add_argument('--class_type', type=str, required=True, help="학습할 카테고리 유형 (예: 'big_cate', 'mid_cate', 'small_cate')")
    parser.add_argument('--model_save_path', type=str, required=True, help="학습된 모델을 저장할 경로")
    parser.add_argument('--label_encoder_path', type=str, required=True, help="라벨 인코더를 저장할 경로")
    
    args = parser.parse_args()

    train_category_model(
        config_path=args.config_path,
        dataset_path=args.dataset_path,
        text_column=args.text_column,
        class_type=args.class_type,
        model_save_path=args.model_save_path,
        label_encoder_path=args.label_encoder_path
    )
