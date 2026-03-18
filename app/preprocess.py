import re
import pandas as pd
import os

def load_valid_words(base_path, target_small):
    """
    DESC:
        지정된 경로 내 폴더들에서 txt 파일명을 기반으로 유효한 단어(valid_words)를 추출합니다.
        파일명에서 특수문자를 제거하고 언더바(_)로 분리된 단어들을 수집합니다.

    Args:
        base_path (str): 폴더의 최상단 경로
        target_small (list): 세부 폴더 리스트

    Returns:
        list: 중복 제거된 유효 단어 리스트 (valid_words)
    """
    special_char_pattern = r'[()\[\]{}<>_\-]'
    valid_words = []

    for folder_name in target_small:
        real_folder_name = folder_name.replace('/', '_')
        folder_path = os.path.join(base_path, real_folder_name)

        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith('.txt'):
                filename = os.path.splitext(file)[0]
                split_words = filename.split('_')
                for word in split_words:
                    clean_word = re.sub(special_char_pattern, ' ', word)
                    clean_word = re.sub(r'\s+', ' ', clean_word).strip()
                    if clean_word:
                        valid_words.append(clean_word)

    valid_words = list(set(valid_words))
    return valid_words

def clean_text(text, valid_words):
    """
    DESC:
        괄호 안의 단어 중 valid_words에 포함된 단어만 유지하고, 나머지 특수문자 등을 제거합니다.

    Args:
        text (str): 입력 텍스트
        valid_words (list): 유효 단어 리스트

    Returns:
        str: 괄호 처리 및 정제된 텍스트
    """
    if not isinstance(text, str):
        return text

    valid_words_set = set(valid_words)

    def bracket_replacer(match):
        inner_text = match.group(1)
        inner_text_cleaned = re.sub(r"[^\w가-힣]", "", inner_text).strip()
        if not inner_text_cleaned:
            return ''
        if inner_text_cleaned in valid_words_set:
            return inner_text_cleaned
        else:
            return ''

    text = re.sub(r"[\(\[\{]([^\)\]\}]*)[\)\]\}]", bracket_replacer, text)
    text = re.sub(r"[^\w가-힣\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_file(file_path, file_type='txt'):
    """
    DESC:
        txt 파일을 줄 단위로 읽어 리스트로 반환합니다.

    Args:
        file_path (str): 파일 경로
        file_type (str): 지원 파일 형식 (기본값은 'txt')

    Returns:
        list: 줄 단위 텍스트 리스트
    """
    if file_type != 'txt':
        raise ValueError("Only 'txt' file type is supported.")
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def apply_regex_from_file(text, regex_file_path):
    """
    DESC:
        정규표현식 파일 내 패턴들을 적용하여 텍스트에서 제거합니다.

    Args:
        text (str): 입력 텍스트
        regex_file_path (str): 정규표현식 파일 경로

    Returns:
        str: 패턴 적용 후의 텍스트
    """
    regex_patterns = load_file(regex_file_path, 'txt')
    for pattern in regex_patterns:
        text = re.sub(pattern, '', text)
    return text

def remove_stopwords(text, stopword_file_path):
    """
    DESC:
        불용어 파일에 포함된 단어들을 텍스트에서 제거합니다.

    Args:
        text (str): 입력 텍스트
        stopword_file_path (str): 불용어 리스트가 담긴 파일 경로

    Returns:
        str: 불용어가 제거된 텍스트
    """
    stopwords = load_file(stopword_file_path, 'txt')
    pattern = '|'.join(re.escape(word) for word in stopwords)
    return re.sub(pattern, '', text)

def remove_units_iterative(text):
    """
    DESC:
        숫자 다음에 오는 단위(예: 500ml, 2개입 등)를 반복적으로 제거합니다.
        단위가 여러 겹으로 붙은 경우에도 완전히 제거될 때까지 반복합니다.

    Args:
        text (str): 입력 텍스트

    Returns:
        str: 단위 제거가 완료된 텍스트
    """
    units = [
        'kg', 'g', 'mg', 'L', 'mL', 'cm', 'mm', 'km', 'm', 'ft', 'inch', 'cc', '시간', '시', '분', '가지', '번길', '번', '본','리터','길','회','갑','겹',
        'ml', 'l', '만', '도씨', '도', '개입', '개', '병', '팩', '박스', '캔', '캔입', '봉지', '장', '판', '호', '줄', '종', '접시','구', '번가',
        '조각', '마리', '입', '롤', '잔', '인치', '인상', '인', '원', '층', '인분', '스푼', '년대', '년','알', '단계','년산', '피스', '주', '곡'
    ]
    sorted_units = sorted(units, key=lambda x: len(x), reverse=True)
    pattern = r'(\d+)\s*(' + '|'.join(map(re.escape, sorted_units)) + r')'

    prev_text = None
    while prev_text != text:
        prev_text = text
        text = re.sub(pattern, r'\1', text)
    return text

def is_english_dominant(text, threshold=0.7):
    """
    DESC:
        텍스트 내 영어 비율이 지정된 임계치 이상인지 판단합니다.

    Args:
        text (str): 입력 텍스트
        threshold (float): 영어 비율 기준 (기본 0.7)

    Returns:
        bool: 영어 위주인지 여부
    """
    total_chars = len(re.sub(r"\s+", "", text))
    if total_chars == 0:
        return False
    eng_chars = len(re.findall(r"[a-zA-Z]", text))
    return (eng_chars / total_chars) >= threshold

def __clean_once(text, regex_file_path, stopword_file_path):
    """
    DESC:
        한 번의 텍스트 정제 작업을 수행합니다.
        괄호 제거 → 단위 제거 → 특수문자 제거 → 정규식 적용 → 한글 외 문자 제거 → 불용어 제거 순으로 수행됩니다.

    Args:
        text (str): 입력 텍스트
        regex_file_path (str): 정규표현식 파일 경로
        stopword_file_path (str): 불용어 파일 경로

    Returns:
        str: 정제된 텍스트
    """
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", "", text)
    text = remove_units_iterative(text)
    text = re.sub(r"[^\w\s]", "", text)
    text = apply_regex_from_file(text, regex_file_path)
    text = re.sub(r"[^가-힣\s]", "", text)
    text = remove_stopwords(text, stopword_file_path)
    return text.strip()

def __iterate_until_clean(text, regex_file_path, stopword_file_path, max_iter=10):
    """
    DESC:
        텍스트 정제를 반복 적용하여 결과가 더 이상 변하지 않을 때까지 반복합니다.

    Args:
        text (str): 입력 텍스트
        regex_file_path (str): 정규표현식 파일 경로
        stopword_file_path (str): 불용어 파일 경로
        max_iter (int): 최대 반복 횟수

    Returns:
        str: 최종 정제된 텍스트
    """
    for _ in range(max_iter):
        cleaned = __clean_once(text, regex_file_path, stopword_file_path)
        if cleaned == text:
            return cleaned
        text = cleaned
    return text

def process_text_row(text, valid_words, regex_file_path, stopword_file_path, max_iter=10):
    """
    DESC:
        한 줄의 텍스트를 전처리합니다. (영문 위주일 경우 빈 문자열 반환)

    Args:
        text (str): 입력 텍스트
        valid_words (list): 유효 단어 리스트
        regex_file_path (str): 정규표현식 경로
        stopword_file_path (str): 불용어 경로
        max_iter (int): 반복 횟수

    Returns:
        str: 전처리 결과 문자열 (빈 문자열 포함 가능)
    """
    if not isinstance(text, str):
        return ""
    if is_english_dominant(text):
        return ""

    try:
        text_cleaned = clean_text(text, valid_words)
        final_cleaned = __iterate_until_clean(text_cleaned, regex_file_path, stopword_file_path, max_iter)
        return final_cleaned
    except Exception as e:
        print(f"전처리 오류 - 텍스트: {text}, 에러: {e}")
        return ""

def preprocess_text_column(df, column_name, regex_file_path, stopword_file_path, max_iter=10):
    """
    DESC:
        원본 데이터 건수를 유지하며 전처리 결과 컬럼을 추가합니다.

    Returns:
        pd.DataFrame: 전처리 결과 컬럼이 포함된 원본 row 그대로 반환
    """
    base_path = './data/rulebased/'
    target_small = ['구이', '기타', '디저트류', '면/만두류', '밥류', '볶음', '빵/과자', '생채/무침류', '쌈류', 
                    '전/부침', '젓갈/장아찌', '조림', '주류', '죽/스프류', '찌개/탕/국', '찜', 
                    '커피/음료', '튀김', '회'] 

    df = df.copy()
    df[column_name] = df[column_name].fillna("").astype(str)

    valid_words = load_valid_words(base_path, target_small)

    df["item_name_eda"] = df[column_name].apply(
        lambda x: process_text_row(x, valid_words, regex_file_path, stopword_file_path, max_iter)
    )

    return df
