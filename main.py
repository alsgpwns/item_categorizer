import argparse
import json
import os
import re
from datetime import datetime
from io import StringIO
from time import perf_counter
from urllib.parse import urlparse
import warnings

import pandas as pd

from app.model_classifier import classify_category, classify_small_fine_category
from app.preprocess import preprocess_text_column

SOURCE_CODE_COLUMNS = ["본부코드", "매장코드", "상품코드"]
OUTPUT_COLUMNS = SOURCE_CODE_COLUMNS + [
    "item_name",
    "item_clean_name",
    "big_cate_roberta",
    "mid_cate_roberta",
    "small_category",
    "fine_category",
]
SNOWFLAKE_OUTPUT_COLUMNS = [
    "HEAD_OFFICE_CODE",
    "SHOP_CODE",
    "ITEM_CODE",
    "ITEM_NAME",
    "ITEM_CLEAN_NAME",
    "BIG_CATE_ROBERTA",
    "MID_CATE_ROBERTA",
    "SMALL_CATEGORY",
    "FINE_CATEGORY",
]


def log_info(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [INFO] {message}")


try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# JSON 파일에서 pipeline 정보를 로드하는 함수
def load_pipeline_info(config_path, base_dir):
    with open(config_path, 'r', encoding='utf-8') as file:
        pipeline_info = json.load(file)

    # 카테고리 종류 (big_cate, mid_cate, small_cate)
    for cate_type, cate_info in pipeline_info.items():
        for key, rel_path in cate_info.items():
            # 경로가 문자열이고 절대경로가 아닐 경우에만 base_dir 기준으로 변환
            if isinstance(rel_path, str) and not os.path.isabs(rel_path):
                abs_path = os.path.normpath(os.path.join(base_dir, rel_path))
                pipeline_info[cate_type][key] = abs_path

    return pipeline_info


def load_json_config(config_path):
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_s3_path(path):
    return isinstance(path, str) and path.startswith("s3://")


def parse_s3_path(s3_path):
    parsed = urlparse(s3_path)
    return parsed.netloc, parsed.path.lstrip("/")


def validate_s3_input_path(s3_path):
    import boto3

    bucket, key = parse_s3_path(s3_path)
    prefix = key.rstrip("/") + "/"
    s3 = boto3.client("s3")

    log_info(f"Validating S3 input path: s3://{bucket}/{prefix}")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = response.get("Contents", [])

    if not contents:
        raise FileNotFoundError(f"X 폴더가 존재하지 않습니다: s3://{bucket}/{prefix}")

    parquet_files = [obj for obj in contents if obj["Key"].endswith(".parquet")]
    if len(parquet_files) != len(contents):
        raise ValueError("X Parquet 외 다른 파일이 포함되어 있습니다.")

    total_size = sum(obj["Size"] for obj in parquet_files)
    if total_size <= 0:
        raise ValueError("X 모든 Parquet 파일의 사이즈가 0입니다.")

    log_info(f"S3 input validation completed: files={len(parquet_files)}, size_kb={total_size / 1024:.2f}")


def load_input_dataframe(input_path):
    if is_s3_path(input_path):
        import pyarrow.dataset as ds

        validate_s3_input_path(input_path)
        log_info(f"Loading parquet dataset: {input_path}")
        dataset = ds.dataset(input_path, format="parquet", partitioning="hive")
        df = dataset.to_table().to_pandas()
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        log_info(f"Input load completed: rows={len(df)}")
        return df

    if isinstance(input_path, str) and input_path.lower().endswith(".parquet"):
        import pyarrow.parquet as pq

        log_info(f"Loading parquet file: {input_path}")
        df = pq.read_table(input_path).to_pandas()
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        log_info(f"Input load completed: rows={len(df)}")
        return df

    try:
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="cp949")

    log_info(f"Loading CSV file: {input_path}")
    log_info(f"Input load completed: rows={len(df)}")
    return df


def build_output_path(input_path, output_path):
    if not is_s3_path(output_path):
        return output_path

    if output_path.endswith(".csv"):
        return output_path

    if not is_s3_path(input_path):
        return output_path.rstrip("/") + "/output_result.csv"

    input_bucket, input_key = parse_s3_path(input_path)
    input_parts = input_key.strip("/").split("/")

    try:
        year = next(part.split("=", 1)[1] for part in input_parts if part.startswith("year="))
        month = next(part.split("=", 1)[1] for part in input_parts if part.startswith("month="))
        day = next(part.split("=", 1)[1] for part in input_parts if part.startswith("day="))
    except StopIteration:
        return output_path.rstrip("/") + "/output_result.csv"

    bucket, key = parse_s3_path(output_path)
    base_path = key.rstrip("/")
    return f"s3://{bucket}/{base_path}/year={year}/month={month}/day={day}/output_result.csv"


def save_output_dataframe(df, output_path):
    if is_s3_path(output_path):
        import boto3

        bucket, key = parse_s3_path(output_path)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
        log_info(f"Output saved to S3: s3://{bucket}/{key}")
        return

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log_info(f"Output saved locally: {output_path}")


def get_snowflake_settings(base_dir):
    config_path = os.path.join(base_dir, "config", "snowflake_info.json")
    config = load_json_config(config_path)

    enabled = str(
        os.getenv("ENABLE_SNOWFLAKE_LOAD", config.get("enabled", "false"))
    ).lower() in {"1", "true", "y", "yes"}

    settings = {
        "enabled": enabled,
        "account": os.getenv("SNOWFLAKE_ACCOUNT", config.get("account", "")),
        "user": os.getenv("SNOWFLAKE_USER", config.get("user", "")),
        "password": os.getenv(
            config.get("password_env_var", "SNOWFLAKE_PASSWORD"),
            config.get("password", ""),
        ),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", config.get("warehouse", "")),
        "database": os.getenv("SNOWFLAKE_DATABASE", config.get("database", "")),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", config.get("schema", "")),
        "table": os.getenv("SNOWFLAKE_TABLE", config.get("table", "")),
        "role": os.getenv("SNOWFLAKE_ROLE", config.get("role", "")),
    }

    required_keys = ["account", "user", "password", "warehouse", "database", "schema", "table"]
    missing_keys = [key for key in required_keys if not settings[key]]
    settings["missing_keys"] = missing_keys
    return settings


def build_snowflake_dataframe(df):
    rename_map = {
        "본부코드": "HEAD_OFFICE_CODE",
        "매장코드": "SHOP_CODE",
        "상품코드": "ITEM_CODE",
        "item_name": "ITEM_NAME",
        "item_clean_name": "ITEM_CLEAN_NAME",
        "big_cate_roberta": "BIG_CATE_ROBERTA",
        "mid_cate_roberta": "MID_CATE_ROBERTA",
        "small_category": "SMALL_CATEGORY",
        "fine_category": "FINE_CATEGORY",
    }
    snowflake_df = df.rename(columns=rename_map).copy()
    return snowflake_df[SNOWFLAKE_OUTPUT_COLUMNS]


def save_output_to_snowflake(df, base_dir):
    settings = get_snowflake_settings(base_dir)
    if not settings["enabled"]:
        log_info("Snowflake load skipped: disabled")
        return

    if settings["missing_keys"]:
        missing = ", ".join(settings["missing_keys"])
        raise ValueError(f"Snowflake load enabled but missing settings: {missing}")

    log_info(
        "Snowflake load started: "
        f"{settings['database']}.{settings['schema']}.{settings['table']}"
    )

    snowflake_df = build_snowflake_dataframe(df)

    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas

    connection_kwargs = {
        "account": settings["account"],
        "user": settings["user"],
        "password": settings["password"],
        "warehouse": settings["warehouse"],
        "database": settings["database"],
        "schema": settings["schema"],
    }
    if settings["role"]:
        connection_kwargs["role"] = settings["role"]

    conn = snowflake.connector.connect(**connection_kwargs)
    try:
        success, chunk_count, row_count, _ = write_pandas(
            conn=conn,
            df=snowflake_df,
            table_name=settings["table"],
            database=settings["database"],
            schema=settings["schema"],
            auto_create_table=False,
            overwrite=False,
            quote_identifiers=False,
        )
    finally:
        conn.close()

    if not success:
        raise RuntimeError("Snowflake load failed during write_pandas execution")

    log_info(
        "Snowflake load completed: "
        f"rows={row_count}, chunks={chunk_count}, "
        f"target={settings['database']}.{settings['schema']}.{settings['table']}"
    )


def reorder_output_columns(df):
    leading_columns = [col for col in SOURCE_CODE_COLUMNS if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in leading_columns]
    return df[leading_columns + remaining_columns]


def select_output_columns(df, target_column):
    rename_map = {}
    if target_column in df.columns and target_column != "item_name":
        rename_map[target_column] = "item_name"

    output_df = df.rename(columns=rename_map).copy()
    selected_columns = [col for col in OUTPUT_COLUMNS if col in output_df.columns]
    return output_df[selected_columns]

# 메인 함수: input_path, output_path, target_column을 인자로 받음
def main(input_path, output_path, target_column, step='all'):
    pipeline_start = perf_counter()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_info(f"Pipeline started: step={step}, target_column={target_column}")
    
    # 설정 파일 경로를 현재 디렉토리 기준으로 안전하게 설정
    regex_file_path = os.path.join(base_dir, 'config', 'reg_pattern_list.txt')
    stopword_file_path = os.path.join(base_dir, 'config', 'stopwords_list.txt')
    config_path = os.path.join(base_dir, 'config', 'model_info.json')
    trie_folder_base = os.path.join(base_dir, 'data', 'rulebased')
    
    # JSON 파일에서 pipeline 정보 로드
    pipeline_info = load_pipeline_info(config_path,base_dir)
    output_path = build_output_path(input_path, output_path)
    if is_s3_path(output_path):
        log_info(f"Planned S3 output path: {output_path}")

    df = load_input_dataframe(input_path)
    item_df_unique = df.iloc[:, :]
    
    ## [전처리 시작] 텍스트 전처리 수행
    if target_column not in df.columns:
        print(f"{target_column} is not in DataFrame")
        return
    if step in ['all', 'preprocess']:
        preprocess_start = perf_counter()
        pre_df = preprocess_text_column(df, target_column, regex_file_path, stopword_file_path)
    
        # 1. 빈 값 / 비어 있지 않은 상품명 기준 분리
        non_item_df = pre_df[pre_df['item_name_eda'] == ''].reset_index(drop=True)
        item_df = pre_df[pre_df['item_name_eda'] != ''].reset_index(drop=True)
        
        log_info(
            f"Preprocess completed: valid_rows={item_df.shape[0]}, "
            f"empty_rows={non_item_df.shape[0]}, elapsed_sec={perf_counter() - preprocess_start:.2f}"
        )
    else:
        item_df = df
        if 'item_clean_name' in item_df.columns:
            item_df.rename(columns={'item_clean_name': 'item_name_eda'}, inplace=True)
            item_df['item_name_eda'].fillna(item_df[target_column])
        else:
            item_df['item_name_eda'] = item_df[target_column]
        non_item_df = pd.DataFrame(columns=item_df.columns)
    ## [전처리 종료] ##

    ## [분류 시작] 모델을 통한 분류
    if step in ['all', 'model']:
        # 2. 중복 제거된 분류 대상 준비
        item_df_unique = item_df[[target_column,'item_name_eda']].drop_duplicates().reset_index(drop=True)
        log_info(f"Classification input prepared: unique_rows={item_df_unique.shape[0]}")
        
        # 3. 대분류 / 중분류 분류 수행
        for category_type in ["big_cate", "mid_cate"]:
            category_start = perf_counter()
            item_df_unique = classify_category(
                item_df_unique, 
                text_column='item_name_eda', 
                category_type=category_type, 
                pipeline_info=pipeline_info
            )
            log_info(f"{category_type} classification elapsed_sec={perf_counter() - category_start:.2f}")
            
        # # Trie 기반 분류
        # shared_trie = Trie(rulebased_base=trie_folder_base)
        # item_df_unique = classify_with_trie_pipeline(item_df_unique, text_column='item_name_eda', trie_folder_base=trie_folder_base, trie=shared_trie)
        
        # # 육회_타다끼 와 같이 _로 명시된 애들 전처리 통한 육회/타다끼로 변경
        # item_df_unique["fine_category"] = item_df_unique["fine_category"].str.replace("_", "/", regex=False)
        # item_df_unique.drop(columns=['classification_step','similar_word','similarity_score'],inplace=True)
        # print(item_df_unique.columns)
        # display(item_df_unique.head(10))
        # item_df_unique = item_df_unique.rename(columns={'small_cate_roberta' : 'small_category'})
        # print(item_df_unique.columns)
        # display(item_df_unique.head(10))
        
        # 4. 소분류 / 세분류 분류 수행
        small_fine_start = perf_counter()
        item_df_unique = classify_small_fine_category(
            df=item_df_unique,
            text_column='item_name_eda',
            pipeline_info=pipeline_info,
            moe_type='hard'
        )
        log_info(f"small/fine classification elapsed_sec={perf_counter() - small_fine_start:.2f}")
    ## [분류 종료] ##

    ## [후처리 시작] 상품명 기준 후처리 시작
    if step in ['all', 'postprocess']:
        # 5. 정규표현식 기반 수동 소분류 보정
        # temperature_pattern = re.compile(r'(?:[\[\(\{]?)(H|I|HOT|ICE)(?:[\]\)\}\.\]]?)', re.IGNORECASE)
        # coffee_fine_category = ['과실차', '기타 커피류', '기타', '녹차', '미숫가루', '밀크티', '버블티', '수정과', '스무디', '아이스티', '주스', '카페라떼', '탄산음료', '프라푸치노', '홍차']
        # small_cate_postprocessing_cond = (item_df_unique['fine_category'].isin(coffee_fine_category)) & (item_df_unique[target_column].str.contains(temperature_pattern))
        # item_df_unique.loc[
        #     small_cate_postprocessing_cond, 
        #     'small_category'
        # ] = '커피/음료'
    
        # 피자 정의
        pizza_pattern = re.compile(r'piz', re.IGNORECASE)
        pizza_mask = item_df_unique[target_column].str.contains(pizza_pattern, na=False)
    
        item_df_unique.loc[pizza_mask, 'big_cate_roberta'] = '양식'
        item_df_unique.loc[pizza_mask, 'mid_cate_roberta'] = '빵/과자'
        item_df_unique.loc[pizza_mask, 'small_category'] = '빵/과자'
        item_df_unique.loc[pizza_mask, 'fine_category'] = '피자'
    
        # IPA 맥주 정의
        ipa_beer_pattern = re.compile(r'IPA', re.IGNORECASE)
        
        # 조건에 맞는 행 필터링
        beer_parts_mask = item_df_unique[target_column].str.contains(ipa_beer_pattern, na=False)
        
        item_df_unique.loc[beer_parts_mask, 'big_cate_roberta'] = '주류'
        item_df_unique.loc[beer_parts_mask, 'mid_cate_roberta'] = '기타곡류'
        item_df_unique.loc[beer_parts_mask, 'small_category'] = '주류'
        item_df_unique.loc[beer_parts_mask, 'fine_category'] = '맥주'
    
    
        # 쌀국수 정의
        rice_noodles_pattern = re.compile(r'PHO', re.IGNORECASE)
        beef_parts_pattern = re.compile(r'^P\d+.*(양지|차돌)', re.IGNORECASE)
        
        # 조건에 맞는 행 필터링
        beef_parts_mask = item_df_unique[target_column].str.contains(beef_parts_pattern, na=False)
        
        item_df_unique.loc[beef_parts_mask, 'big_cate_roberta'] = '동남아식'
        item_df_unique.loc[beef_parts_mask, 'mid_cate_roberta'] = '면/만두류'
        item_df_unique.loc[beef_parts_mask, 'small_category'] = '면/만두류'
        item_df_unique.loc[beef_parts_mask, 'fine_category'] = '쌀국수'
        
        # 쌀국수 포함된 행 필터링
        rice_noodles_mask = item_df_unique[target_column].str.contains(rice_noodles_pattern, na=False)
        
        item_df_unique.loc[rice_noodles_mask, 'big_cate_roberta'] = '동남아식'
        item_df_unique.loc[rice_noodles_mask, 'mid_cate_roberta'] = '면/만두류'
        item_df_unique.loc[rice_noodles_mask, 'small_category'] = '면/만두류'
        item_df_unique.loc[rice_noodles_mask, 'fine_category'] = '쌀국수'
        				
        # 케이크 또는 숫자+호 패턴 정규표현식
        cake_or_ho_pattern = re.compile(r'cake|\d+호', re.IGNORECASE)
        
        # 조건에 맞는 행에 카테고리 지정
        cake_or_ho_mask = item_df_unique[target_column].str.contains(cake_or_ho_pattern, na=False)
        
        item_df_unique.loc[cake_or_ho_mask, 'big_cate_roberta'] = '베이커리'
        item_df_unique.loc[cake_or_ho_mask, 'mid_cate_roberta'] = '빵/과자'
        item_df_unique.loc[cake_or_ho_mask, 'small_category'] = '빵/과자'
        item_df_unique.loc[cake_or_ho_mask, 'fine_category'] = '케이크'
    
        # soup 패턴 정의 (대소문자 구분 없음)
        soup_pattern = re.compile(r'soup', re.IGNORECASE)
    
        # soup 포함된 행 필터링
        soup_mask = item_df_unique[target_column].str.contains(soup_pattern, na=False)
        
        # 카테고리 값 설정
        item_df_unique.loc[soup_mask, 'big_cate_roberta'] = '양식'
        item_df_unique.loc[soup_mask, 'small_category'] = '죽/스프류'
        item_df_unique.loc[soup_mask, 'fine_category'] = '스프'
    
        # XO 또는 X.O 패턴 정의 (대소문자 구분 없음)
        xo_pattern = re.compile(r'\bX\.?O\b', re.IGNORECASE)
        
        # XO 또는 X.O 포함된 행 필터링
        xo_mask = item_df_unique[target_column].str.contains(xo_pattern, na=False)
        
        # big_category_roberta 값 수정
        item_df_unique.loc[xo_mask, 'big_cate_roberta'] = '중식'
    ## [후처리 종료] ##

    ## [매핑 시작] 모델 분류, 후처리로 만들어진 매핑 사전을 사용하여 대, 중, 소, 세분류 매핑
    if step in ['all', 'model', 'postprocess']:
        # 6. 분류 결과 item_df에 매핑
        mapping_columns  = ['big_cate_roberta', 'mid_cate_roberta','small_category', 'fine_category']
        category_mapping = {
            col: item_df_unique.set_index('item_name_eda')[col].to_dict()
            for col in mapping_columns
        }
        for col in mapping_columns:
            item_df[col] = item_df['item_name_eda'].map(category_mapping[col])
            
        # [TEMP] 기타해산물 라벨링 -> 기타 해산물로 변경
        item_df.loc[item_df['mid_cate_roberta']=='기타해산물', 'mid_cate_roberta'] = '기타 해산물'
    ## [매핑 종료] ##

    ## [산출물 저장]
    # 7. 최종 결과 생성
    result_df = pd.concat([item_df, non_item_df]).reset_index(drop=True)
    result_df = result_df.rename(columns={'item_name_eda' : 'item_clean_name'})
    result_df = reorder_output_columns(result_df)
    result_df = select_output_columns(result_df, target_column)
    log_info(f"Output dataframe prepared: rows={result_df.shape[0]}")

    # 결과를 CSV 파일로 저장
    save_output_dataframe(result_df, output_path)
    save_output_to_snowflake(result_df, base_dir)
    log_info(f"Pipeline completed: total_elapsed_sec={perf_counter() - pipeline_start:.2f}")
    return result_df

# 스크립트 실행을 위한 argparse 설정
if __name__ == "__main__":

    # 명령어 인자 파서 생성
    parser = argparse.ArgumentParser(description="카테고리 분류 파이프라인 실행")
    parser.add_argument('--input_path', type=str, required=True, help="입력 데이터 CSV 파일 경로")
    parser.add_argument('--output_path', type=str, required=True, help="결과를 저장할 CSV 파일 경로")
    parser.add_argument('--target_column', type=str, required=True, help="상품명 컬럼")
    parser.add_argument('--step', type=str, required=True, help="실행할 모듈명 입력 [all, preprocess, model, postprocess]")
    
    # 인자 파싱
    args = parser.parse_args()
    
    #메인 함수 실행
    main(args.input_path, args.output_path, args.target_column, args.step)
