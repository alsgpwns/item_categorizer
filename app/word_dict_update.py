import os

def update_txt_file(small_category, fine_category, additional_words, base_path="./data/rulebased", mode="update"):
    """
    Desc : 소분류와 세분류에 해당하는 텍스트 파일을 업데이트, 생성, 또는 삭제.
    Args :
        small_category (str): 소분류 이름 (예: '구이').
        fine_category (str): 세분류 이름 (예: '삼겹살').
        additional_words (list): 추가하거나 삭제할 단어 리스트 (예: ['양념삼겹살', '칼집삼겹살']).
        base_path (str): 기본 경로. 기본값은 './data/rulebased'.
        mode (str): 동작 모드 ('update', 'create', 또는 'delete'). 기본값은 'update'.
    """
    # 파일 경로 설정
    folder_path = os.path.join(base_path, small_category)
    file_path = os.path.join(folder_path, f"{fine_category}.txt")
    
    # 폴더가 없는 경우 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: '{folder_path}'")

    # 'create' 모드: 새 파일 생성 및 사전 구축
    if mode == "create":
        if os.path.exists(file_path):
            print(f"File '{file_path}' already exists. Switching to 'update' mode.")
        else:
            with open(file_path, 'w', encoding='utf-8') as file:
                for word in sorted(set(additional_words)):  # 중복 제거 및 정렬 후 저장
                    file.write(f"{word}\n")
            print(f"File '{file_path}' created successfully with {len(additional_words)} words.")
            return

    # 'update' 모드: 기존 파일 업데이트
    if mode == "update":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist. Use 'create' mode to create a new file.")
        
        # 기존 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_words = set(line.strip() for line in file if line.strip())
        
        # 추가 단어 업데이트
        updated_words = existing_words.union(additional_words)
        
        # 업데이트된 단어 리스트 저장
        with open(file_path, 'w', encoding='utf-8') as file:
            for word in sorted(updated_words):  # 정렬된 상태로 저장
                file.write(f"{word}\n")
        
        print(f"File '{file_path}' updated successfully with {len(additional_words)} new words.")
        print(f"Total words in the file: {len(updated_words)}")
        return

    # 'delete' 모드: 특정 단어 삭제
    if mode == "delete":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist. Cannot delete words from a non-existent file.")
        
        # 기존 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_words = set(line.strip() for line in file if line.strip())
        
        # 삭제할 단어 제거
        updated_words = existing_words - set(additional_words)
        deleted_words = existing_words & set(additional_words)
        
        # 업데이트된 단어 리스트 저장
        with open(file_path, 'w', encoding='utf-8') as file:
            for word in sorted(updated_words):  # 정렬된 상태로 저장
                file.write(f"{word}\n")
        
        print(f"File '{file_path}' updated successfully. Deleted {len(deleted_words)} words.")
        print(f"Remaining words in the file: {len(updated_words)}")
        return

    # 유효하지 않은 모드 처리
    raise ValueError("Invalid mode. Please use 'update', 'create', or 'delete'.")
