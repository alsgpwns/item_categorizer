from .model.roberta_classifier import CategoryModel
from .model.moe_classifier import MoEClassifier
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import joblib
from tqdm import tqdm


# 텍스트 데이터 분류
def classify_category(df, text_column, category_type, pipeline_info, max_len=128):
    """
    Desc : 입력 데이터프레임의 텍스트 열을 KLUE-RoBERTa 모델로 분류.
    Args :
        df (pd.DataFrame): 분류할 데이터가 포함된 데이터프레임.
        text_column (str): 텍스트 데이터가 포함된 열 이름.
        category_type (str): 분류할 카테고리 유형 (예: 'big_cate', 'mid_cate', 'small_cate').
        pipeline_info (dict): 모델 경로와 라벨 인코더 경로가 포함된 설정 정보.
        max_len (int): 입력 텍스트의 최대 토큰 길이. 기본값 128.
    Returns :
        pd.DataFrame: 분류 결과가 추가된 데이터프레임.
    """
    # 모델 및 라벨 인코더 경로 설정
    model_path = pipeline_info[category_type]['model_path']
    label_encoder_path = pipeline_info[category_type]['label_path']
    bert_path = pipeline_info[category_type]['bert_path']

    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Connect torch Device info : {device}")
    
    # 라벨 인코더 로드
    label_encoder = joblib.load(label_encoder_path)
    num_classes = len(label_encoder.classes_)
    print(f"class 개수 : {num_classes}")
    print(f"{label_encoder.classes_}")
    # 모델 로드 및 설정
    model = CategoryModel(num_classes=num_classes, bert_path=bert_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    # 예측 결과와 확률 저장 리스트 초기화
    predictions, probabilities = [], []

    # 데이터프레임의 텍스트 열에 대해 분류 수행
    for text in tqdm(df[text_column], desc=f"Processing {category_type} categories"):
        # 텍스트 인코딩
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # 모델 추론
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output, dim=1)  # 소프트맥스 함수로 확률 계산
            pred = torch.argmax(probs, dim=1).cpu().numpy()[0]  # 예측 클래스
            prob = probs[0][pred].cpu().item()  # 예측 클래스의 확률

        # 라벨 인코더를 통해 클래스 이름 변환
        predicted_label = label_encoder.inverse_transform([pred])[0]
        predictions.append(predicted_label)  # 예측 라벨 저장
        probabilities.append(prob)  # 예측 확률 저장

    # 데이터프레임에 예측 결과와 확률 추가
    df[f'{category_type}_roberta'] = predictions
    df[f'{category_type}_roberta_prob'] = probabilities

    return df

def classify_small_fine_category(df, text_column, pipeline_info='./model', moe_type='hard'):
    """
    Desc : 입력 데이터프레임의 텍스트 열을 ROBERTA모델을 베이스로 한 MOE 기법을 통한 소분류 및 세분류 분류 + 각 카테고리별 score 제공
    Args :
        df (pd.DataFrame): 분류할 데이터가 포함된 데이터프레임.
        text_column (str): 텍스트 데이터가 포함된 열 이름.
        pipeline_info (dict): 모델 경로와 라벨 인코더 경로가 포함된 설정 정보.
        moe_type (str): MOE 모델 분류 시 사용되는 가중치 설정에 따른 모델 분기. 기본값 hard
    Returns :
        pd.DataFrame: 분류 결과 및 신뢰도(score)가 추가된 데이터프레임.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Connection device info : {device}")

    # path info
    small_cate_le_path = pipeline_info['small_fine_cate']['small_cate_le']
    fine_cate_le_dict_path = pipeline_info['small_fine_cate']['fine_cate_le_dict']
    label2_dict_path = pipeline_info['small_fine_cate']['label2_dict']
    model_state_path = pipeline_info['small_fine_cate']['model_path']
    bert_path = pipeline_info['small_fine_cate']['bert_path']

    # Load encoders
    small_cate_le = joblib.load(small_cate_le_path)
    fine_cate_le_dict = joblib.load(fine_cate_le_dict_path)
    label2_dict = joblib.load(label2_dict_path)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    # Model
    model = MoEClassifier(
        backbone_model_name=bert_path,
        hidden_dim=256,
        label2_dict=label2_dict,
        moe_type=moe_type
    ).to(device)
    model.load_state_dict(torch.load(model_state_path, map_location=device), strict=False)
    model.eval()

    # Tokenize input
    class InferDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe):
            self.df = dataframe.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            encoding = tokenizer(
                str(row[text_column]),
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt"
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }

    dataset = InferDataset(df)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    all_small, all_fine = [], []
    all_small_score, all_fine_score = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            if moe_type == 'hard':
                logits1, logits2_list = model(input_ids, attention_mask)
                probs1 = F.softmax(logits1, dim=1)
                pred_label1 = torch.argmax(probs1, dim=1)

                probs2 = [F.softmax(logit, dim=0) for logit in logits2_list]
                pred_label2 = torch.tensor([torch.argmax(logit).item() for logit in logits2_list])

                for i in range(len(pred_label1)):
                    small = small_cate_le.inverse_transform([pred_label1[i].item()])[0]
                    score1 = probs1[i][pred_label1[i]].item()

                    fine_le = fine_cate_le_dict[pred_label1[i].item()]
                    fine = fine_le.inverse_transform([pred_label2[i].item()])[0]
                    score2 = probs2[i][pred_label2[i].item()].item()

                    all_small.append(small)
                    all_fine.append(fine)
                    all_small_score.append(score1)
                    all_fine_score.append(score2)

            else:
                gate_probs, logits2 = model(input_ids, attention_mask)
                pred_label1 = torch.argmax(gate_probs, dim=1)
                probs2 = F.softmax(logits2, dim=1)
                pred_label2 = torch.argmax(probs2, dim=1)

                for i in range(len(pred_label1)):
                    small = small_cate_le.inverse_transform([pred_label1[i].item()])[0]
                    score1 = gate_probs[i][pred_label1[i]].item()

                    fine_le = fine_cate_le_dict[pred_label1[i].item()]
                    fine = fine_le.inverse_transform([pred_label2[i].item()])[0]
                    score2 = probs2[i][pred_label2[i]].item()

                    all_small.append(small)
                    all_fine.append(fine)
                    all_small_score.append(score1)
                    all_fine_score.append(score2)

    df['small_category'] = all_small
    df['fine_category'] = all_fine
    df['small_category_score'] = all_small_score
    df['fine_category_score'] = all_fine_score

    return df
