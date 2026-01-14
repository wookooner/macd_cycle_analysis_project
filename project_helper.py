import os
from pathlib import Path

def find_project_structure():
    """현재 프로젝트 구조를 자세히 시각적으로 표시합니다."""
    
    current_path = Path.cwd()  # 현재 작업 디렉토리
    print(f"현재 위치: {current_path}\n")
    
    # 상위 디렉토리까지 탐색하여 프로젝트 루트 찾기
    project_root = current_path
    for _ in range(5):  # 최대 5단계 상위까지
        if any(folder.name in ['cycle_algorithm', 'cycle_detect', 'cycle_detection'] 
               for folder in project_root.iterdir() if folder.is_dir()):
            break
        parent = project_root.parent
        if parent == project_root: # 루트 디렉토리에 도달
            break
        project_root = parent

    print(f"🔍 프로젝트 루트: {project_root}\n")

    def display_structure(path, prefix=''):
        """재귀적으로 파일 및 폴더 구조를 출력합니다."""
        items = sorted(list(path.iterdir()), key=lambda x: (x.is_file(), x.name))
        
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