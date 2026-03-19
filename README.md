# item_categorizer

KICC POS 상품명 데이터를 분류하는 배치 프로젝트입니다.

입력 데이터는 S3 parquet 또는 로컬 CSV/parquet 파일을 사용할 수 있고, 결과는 S3 CSV 또는 로컬 CSV로 저장됩니다. 현재 운영 기준으로는 EC2 cron 배치에서 S3 입력을 읽어 상품명 분류를 수행한 뒤, S3와 Snowflake에 결과를 적재하는 구조입니다.

## 주요 기능

- 상품명 전처리
- 대분류 / 중분류 분류
- 소분류 / 세분류 분류
- 원본 키 유지
  - `본부코드`
  - `매장코드`
  - `상품코드`
- S3 입력 경로 검증
  - 경로 존재 여부
  - parquet 확장자 여부
  - 총 파일 용량 확인
- S3 결과 CSV 저장
- Snowflake append 적재 옵션 지원

## 디렉터리 구조

```text
.
├── app/
├── config/
├── input/
├── main.py
├── requirements.txt
├── run_daily_batch.sh
├── run_model.sh
├── run_pipeline.sh
├── run_postprocess.sh
└── run_preprocess.sh
```

## 주요 파일

- [`main.py`](/Users/heajunmin/Downloads/20251217_kicc_release/main.py)
  - 전체 파이프라인 진입점
- [`app/preprocess.py`](/Users/heajunmin/Downloads/20251217_kicc_release/app/preprocess.py)
  - 상품명 전처리 로직
- [`app/model_classifier.py`](/Users/heajunmin/Downloads/20251217_kicc_release/app/model_classifier.py)
  - 대/중/소/세분류 분류 로직
- [`config/model_info.json`](/Users/heajunmin/Downloads/20251217_kicc_release/config/model_info.json)
  - 모델 경로 설정
- [`config/snowflake_info.json`](/Users/heajunmin/Downloads/20251217_kicc_release/config/snowflake_info.json)
  - Snowflake 적재 설정
- [`run_daily_batch.sh`](/Users/heajunmin/Downloads/20251217_kicc_release/run_daily_batch.sh)
  - 일배치 실행 스크립트

## 입력 / 출력 스키마

### 입력

운영 기준 원본 입력 경로:

```text
s3://zone-1-raw-data/kicc/pos/master/shop_item_daily_update/year=YYYY/month=MM/day=DD/
```

핵심 컬럼:

- `본부코드`
- `매장코드`
- `상품코드`
- `상품명`

### 출력

운영 기준 결과 경로:

```text
s3://zone-1-raw-data/kicc/pos/master/categorized_shop_item_daily_update/year=YYYY/month=MM/day=DD/output_result.csv
```

최종 결과 컬럼:

- `본부코드`
- `매장코드`
- `상품코드`
- `item_name`
- `item_clean_name`
- `big_cate_roberta`
- `mid_cate_roberta`
- `small_category`
- `fine_category`

Snowflake 적재 시 컬럼명은 아래처럼 영문으로 변환됩니다.

- `HEAD_OFFICE_CODE`
- `SHOP_CODE`
- `ITEM_CODE`
- `ITEM_NAME`
- `ITEM_CLEAN_NAME`
- `BIG_CATE_ROBERTA`
- `MID_CATE_ROBERTA`
- `SMALL_CATEGORY`
- `FINE_CATEGORY`

## 로컬 실행

### 1. 가상환경 준비

```sh
python3 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

### 2. 샘플 실행

```sh
python main.py \
  --input_path ./input/sample_input.csv \
  --output_path ./output/sample_output.csv \
  --target_column item_name \
  --step all
```

### 3. 로컬 parquet 실행

```sh
python main.py \
  --input_path ./input/sample.parquet \
  --output_path ./output/sample_output.csv \
  --target_column 상품명 \
  --step all
```

## EC2 배치 실행

운영 경로 예시:

```text
/home/datalake/apps/kicc_item_categorizer
```

수동 실행:

```sh
cd /home/datalake/apps/kicc_item_categorizer
PYTHON_BIN=/home/datalake/anaconda3/bin/python3 RUN_DATE=2026-03-17 ./run_daily_batch.sh
```

로그 경로:

```text
/home/datalake/apps/kicc_item_categorizer/logs/daily_job_YYYY-MM-DD.log
```

## Snowflake 적재

Snowflake 적재는 선택형입니다. 결과 CSV를 S3에 저장한 뒤, 같은 실행에서 Snowflake 테이블로 append 적재할 수 있습니다.

대상 테이블 예시:

```text
DATALAKE.KICC.I_POS_ITEM_CLASSIFY_RESULT
```

### 설정 파일

[`config/snowflake_info.json`](/Users/heajunmin/Downloads/20251217_kicc_release/config/snowflake_info.json)

예시:

```json
{
  "enabled": true,
  "account": "HV92036",
  "user": "hjmin",
  "password_env_var": "SNOWFLAKE_PASSWORD",
  "warehouse": "COMPUTE_WH",
  "database": "DATALAKE",
  "schema": "KICC",
  "table": "I_POS_ITEM_CLASSIFY_RESULT",
  "role": "ACCOUNTADMIN"
}
```

### 환경변수

```sh
export SNOWFLAKE_PASSWORD='your_password'
```

### 주의

- `SNOWFLAKE_PASSWORD`는 파일에 저장하지 않고 환경변수로 주입합니다.
- cron 환경에서는 별도 env 파일 또는 안전한 주입 방식이 필요합니다.

## 운영 메모

- 배치 기준 일자는 기본적으로 `yesterday` 입니다.
- 따라서 `2026-03-18`에 실행된 cron은 `day=17` 데이터를 처리합니다.
- 원본 복구 키는 `본부코드 + 매장코드 + 상품코드` 조합을 사용합니다.
- `상품코드` 단독은 전역 유니크 키로 가정하지 않습니다.

## 참고

- 모델 파일은 용량 문제로 Git 관리 대상에서 제외할 수 있습니다.
- `output/`, `logs/`, `cache/`, `.venv*`, `*.tar.gz` 는 `.gitignore`에 의해 제외됩니다.
