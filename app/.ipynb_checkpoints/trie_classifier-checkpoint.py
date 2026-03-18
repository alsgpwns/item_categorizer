import os
import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import faiss
import json
import joblib

# ──────────────────────────────── #
# Trie 클래스 정의
# ──────────────────────────────── #

class TrieNode:
    def __init__(self):
        self.children = {}
        self.categories = set()

class Trie:
    def __init__(self, rulebased_base="./data/rulebased", model_name="nlpai-lab/KURE-v1"):
        self.root = TrieNode()
        self.words = []
        self.rulebased_base = rulebased_base
        self.model = SentenceTransformer(model_name)
        self.faiss_cache = {}

    def insert(self, word, category):
        self.words.append(word)
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.categories.add(category)

    def search(self, word):
        results = set()
        node = self.root
        for char in word:
            if char not in node.children:
                return results
            node = node.children[char]
        results.update(node.categories)
        return results

    def partial_substring_search(self, text):
        results = set()
        for word in self.words:
            if text in word:
                results.update(self.search(word))
        return results

    def similarity_search(self, keyword, small_cate, threshold=0.6, top_k=5):
        if not small_cate:
            return set(), None, 0
        small_cate = preprocess_text(small_cate.replace("/", "_"))

        if small_cate not in self.faiss_cache:
            vectordb_path = os.path.join(self.rulebased_base, small_cate, "vectordb")
            index_path = os.path.join(vectordb_path, f"{small_cate}.index")
            meta_path = os.path.join(vectordb_path, f"{small_cate}.json")

            if not os.path.exists(index_path) or not os.path.exists(meta_path):
                print(f"⚠️ {small_cate} vectordb 파일 없음. 스킵.")
                return set(), None, 0

            faiss_index = faiss.read_index(index_path)
            metadata = joblib.load(open(meta_path, 'rb'))
            self.faiss_cache[small_cate] = (faiss_index, metadata)

        faiss_index, metadata = self.faiss_cache[small_cate]

        keyword = preprocess_text(keyword)
        if not keyword:
            return set(), None, 0

        query_vec = self.model.encode([keyword], normalize_embeddings=True).astype("float32")
        D, I = faiss_index.search(query_vec, top_k)

        best_idx = I[0][0]
        best_score = D[0][0]

        if best_idx == -1 or best_score < threshold:
            return set(), None, 0

        matched_info = metadata[best_idx]
        return {matched_info.get('fine_cate', 'Unknown')}, matched_info.get('text', ''), float(best_score)

# ──────────────────────────────── #
# 유틸 함수
# ──────────────────────────────── #

def preprocess_text(text):
    """
    DESC : 문자열 전처리 함수
           - 공백 제거 및 소문자 변환으로 검색 정확도 향상
    Args :
        text (str): 원본 문자열
    Output :
        str: 전처리된 문자열
    """
    return text.replace(" ", "").strip().lower()

def load_text_files_into_trie(folder_path):
    """
    DESC : 폴더 내 텍스트 파일들을 Trie에 로드
           - 각 파일명을 카테고리로, 각 라인을 키워드로 간주
    Args :
        folder_path (str): 텍스트 파일들이 존재하는 폴더 경로
    Output :
        Trie: 키워드-카테고리 맵이 로드된 Trie 객체
    """
    trie = Trie()
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        category = file_name.split('.')[0]
        with open(file_path, 'r', encoding='utf-8') as file:
            contents = set(preprocess_text(line) for line in file if line.strip())
        for word in contents:
            trie.insert(word, category)
    return trie

def preload_tries(trie_folder_base):
    """
    DESC : 모든 소분류 폴더에 대해 Trie 객체 생성 및 캐싱
    Args :
        trie_folder_base (str): 상위 폴더 경로 (소분류 폴더들이 위치)
    Output :
        dict: {sub_category(str): Trie 객체}
    """
    trie_cache = {}
    for sub_category in os.listdir(trie_folder_base):
        folder_path = os.path.join(trie_folder_base, sub_category)
        if os.path.isdir(folder_path):
            trie_cache[sub_category.replace("/", "_")] = load_text_files_into_trie(folder_path)
    return trie_cache

def include_fine_category(row, folder_path):
    """
    DESC : 파일명 기반으로 텍스트에 포함된 미세 카테고리 탐색
    Args :
        row (pd.Series): 데이터프레임 한 행 (item_name_eda, small_cate_roberta 필수)
        folder_path (str): 텍스트 기반 카테고리 정보가 담긴 폴더
    Output :
        str: 탐색된 fine_category 또는 'Unknown'
    """
    small_cate = row['small_cate_roberta'].replace('/', '_')
    fine_category_folder = os.path.join(folder_path, small_cate)
    if not os.path.isdir(fine_category_folder):
        return 'Unknown'
    for file_name in os.listdir(fine_category_folder):
        if file_name.endswith('.txt'):
            fine_categories = file_name.replace('.txt', '').split('_')
            if any(category in row['item_name_eda'] for category in fine_categories):
                return file_name.replace('.txt', '')
    return 'Unknown'

# ──────────────────────────────── #
# 분류 함수들
# ──────────────────────────────── #

def classify_simple_search(row, trie_cache, text_column):
    """
    DESC : 전체 문자열이 Trie 키워드와 정확히 일치하는 경우 분류
    Args :
        row (pd.Series): 분류할 한 줄의 데이터
        trie_cache (dict): 사전 로딩된 Trie 객체들
        text_column (str): 텍스트 컬럼 이름
    Output :
        tuple(str, str): (fine_category, classification_step)
    """
    keyword = preprocess_text(row[text_column])
    sub_category = row['small_cate_roberta']
    if pd.isna(sub_category): return "Unknown", "Unknown"
    trie = trie_cache.get(preprocess_text(sub_category.replace("/", "_")))
    if not trie: return "Unknown", "Unknown"
    categories = trie.search(keyword)
    if categories: return list(categories)[0], "Simple Search"
    return "Unknown", "Unknown"

def classify_partial_substring_search(row, trie_cache, text_column):
    """
    DESC : 텍스트 전체가 Trie 내 키워드 일부에 포함되는지 검색
    Args :
        row (pd.Series): 한 줄 데이터
        trie_cache (dict): Trie 캐시
        text_column (str): 텍스트 컬럼 이름
    Output :
        tuple(str, str): (fine_category, classification_step)
    """

    keyword = preprocess_text(row[text_column])
    sub_category = row['small_cate_roberta']
    if pd.isna(sub_category): return "Unknown", "Unknown"
    trie = trie_cache.get(preprocess_text(sub_category.replace("/", "_")))
    if not trie: return "Unknown", "Unknown"
    categories = trie.partial_substring_search(keyword)
    if categories: return list(categories)[0], "Word Inclusion"
    return "Unknown", "Unknown"

def classify_word_inclusion(row, folder_path):
    """
    DESC : 텍스트가 특정 파일명(카테고리명)에 포함되는지 검색
    Args :
        row (pd.Series): 한 줄 데이터
        folder_path (str): 텍스트 기반 카테고리 경로
    Output :
        tuple(str, str): (fine_category, classification_step)
    """
    fine_category = include_fine_category(row, folder_path)
    return (fine_category, "File Inclusion") if fine_category != "Unknown" else ("Unknown", "Unknown")

def classify_similarity_search(row, text_column, trie, threshold):
    """
    DESC : 임베딩 유사도를 기반으로 FAISS index에서 가장 가까운 텍스트 탐색
    Args :
        row (pd.Series): 한 줄 데이터
        text_column (str): 텍스트 컬럼
        trie (Trie): 검색용 Trie 객체 (임베딩 포함)
        threshold (float): 유사도 임계값
    Output :
        tuple(str, str, str, float):
            fine_category, classification_step, best_match_text, similarity_score
    """
    keyword = preprocess_text(row[text_column])
    sub_category = row['small_cate_roberta']
    if pd.isna(sub_category): return "Unknown", "Unknown", None, None
    categories, best_match, similarity = trie.similarity_search(keyword, sub_category, threshold)
    if categories: return list(categories)[0], "Similarity Search", best_match, similarity
    return "Unknown", "Unknown", best_match, similarity

# ──────────────────────────────── #
# 전체 분류 파이프라인
# ──────────────────────────────── #

def classify_with_trie_pipeline(df, text_column, trie_folder_base, similarity_threshold=0.6, trie=None):
    """
    Desc : 전체 Trie 기반 분류 파이프라인 (ngram 제외)
    Args :
        df (pd.DataFrame): 입력 데이터프레임
        text_column (str): 텍스트 컬럼명
        trie_folder_base (str): Trie 폴더 경로
        similarity_threshold (float): 유사도 임계값
        trie (Trie): 사전 생성된 Trie 객체 (default: None)
    Returns :
        pd.DataFrame: 분류 결과가 포함된 데이터프레임
    """
    timings = {}

    if trie is None:
        print("Initializing Trie and loading model...")
        trie = Trie(rulebased_base=trie_folder_base)
    else:
        print("Using externally provided Trie instance.")

    print("Preloading tries...")
    start_time = time.time()
    trie_cache = preload_tries(trie_folder_base)
    timings['Preload Tries'] = time.time() - start_time
    print("Fine category classification started.")

    df['fine_category'] = "Unknown"
    df['classification_step'] = "Unknown"
    df['similar_word'] = None
    df['similarity_score'] = None
    total_rows = len(df)

    # Step 1: Simple Search
    start_time = time.time()
    results = df.apply(lambda row: classify_simple_search(row, trie_cache, text_column), axis=1)
    df[['fine_category', 'classification_step']] = pd.DataFrame(results.tolist(), columns=['fine_category', 'classification_step'], index=df.index)
    count = (df['classification_step'] == "Simple Search").sum()
    print(f"[Step 1: Simple Search] 처리된 데이터: {count}/{total_rows}")
    timings['Simple Search'] = time.time() - start_time
    
    # Step 2: Word Inclusion
    start_time = time.time()
    unknown_rows = df['fine_category'] == "Unknown"
    results = df.loc[unknown_rows].apply(lambda row: classify_partial_substring_search(row, trie_cache, text_column), axis=1)
    results_df = pd.DataFrame(results.tolist(), columns=['fine_category', 'classification_step'], index=df[unknown_rows].index)
    df.loc[unknown_rows, ['fine_category', 'classification_step']] = results_df
    count = (df['classification_step'] == "Word Inclusion").sum()
    print(f"[Step 2: Word Inclusion] 처리된 데이터: {count}/{total_rows}")
    timings['Word Inclusion'] = time.time() - start_time
    
    # Step 3: File Inclusion
    start_time = time.time()
    unknown_rows = df['fine_category'] == "Unknown"
    results = df.loc[unknown_rows].apply(lambda row: classify_word_inclusion(row, trie_folder_base), axis=1)
    results_df = pd.DataFrame(results.tolist(), columns=['fine_category', 'classification_step'], index=df[unknown_rows].index)
    df.loc[unknown_rows, ['fine_category', 'classification_step']] = results_df
    count = (df['classification_step'] == "File Inclusion").sum()
    print(f"[Step 3: File Inclusion] 처리된 데이터: {count}/{total_rows}")
    timings['File Inclusion'] = time.time() - start_time
    
    # Step 4: Similarity Search
    start_time = time.time()
    unknown_rows = df['fine_category'] == "Unknown"
    results = df.loc[unknown_rows].apply(
        lambda row: classify_similarity_search(row, text_column, trie, threshold=similarity_threshold), axis=1
    )
    results_df = pd.DataFrame(
        results.tolist(),
        columns=['fine_category', 'classification_step', 'similar_word', 'similarity_score'],
        index=df[unknown_rows].index
    )
    df.loc[unknown_rows, ['fine_category', 'classification_step', 'similar_word', 'similarity_score']] = results_df
    count = (df['classification_step'] == "Similarity Search").sum()
    print(f"[Step 4: Similarity Search] 처리된 데이터: {count}/{total_rows}")
    timings['Similarity Search'] = time.time() - start_time


    # Summary 출력
    print("\n[단계별 소요 시간]")
    for step, elapsed in timings.items():
        print(f"{step}: {elapsed:.2f}초")

    print("\n[최종 classification_step 분포]")
    print(df['classification_step'].value_counts(dropna=False))

    return df


