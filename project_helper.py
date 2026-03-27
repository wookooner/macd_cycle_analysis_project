import os
from pathlib import Path

def find_project_structure():
    """현재 프로젝트 구조를 자세히 시각적으로 표시합니다."""
    
    # 스크립트가 실행된 현재 작업 디렉토리를 프로젝트 루트로 고정
    project_root = Path.cwd()  
    print(f"현재 위치이자 🔍 프로젝트 루트: {project_root}\n")

    # 출력에서 제외할 불필요한 폴더 목록 (필요에 따라 추가/삭제 가능)
    ignore_dirs = {'.git', '__pycache__', 'venv', '.venv', 'env', '.idea', '.vscode', 'node_modules'}

    def display_structure(path, prefix=''):
        """재귀적으로 파일 및 폴더 구조를 출력합니다."""
        
        # 무시할 폴더를 제외하고 목록 생성
        valid_items = [item for item in path.iterdir() if item.name not in ignore_dirs]
        # 폴더를 먼저 보여주고, 그 다음 파일을 알파벳 순으로 정렬
        items = sorted(valid_items, key=lambda x: (x.is_file(), x.name))
        
        for i, item in enumerate(items):
            # 마지막 아이템인지 확인
            is_last = (i == len(items) - 1)
            # 현재 아이템에 대한 접두사
            new_prefix = prefix + ('└── ' if is_last else '├── ')
            # 다음 재귀 호출에 대한 접두사
            next_prefix = prefix + ('    ' if is_last else '│   ')

            if item.is_dir():
                print(f"{new_prefix}📁 {item.name}/")
                # 서브 디렉토리 재귀 탐색
                try:
                    display_structure(item, next_prefix)
                except PermissionError:
                    print(f"{next_prefix}    ❌ 접근 권한 없음")
            else:
                print(f"{new_prefix}📄 {item.name}")

    print("📁 프로젝트 구조:")
    display_structure(project_root)

if __name__ == "__main__":
    find_project_structure()