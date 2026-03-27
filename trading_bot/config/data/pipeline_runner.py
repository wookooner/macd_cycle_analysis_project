"""
trading_bot/data/pipeline_runner.py
====================================
기존 update_pipeline.py를 subprocess로 호출하는 래퍼.
"""

import subprocess
import sys
import logging
from pathlib import Path

logger = logging.getLogger("bot.pipeline")


class PipelineRunner:
    """update_pipeline.py를 실행하여 데이터 갱신"""

    def __init__(self, project_root: Path, python_path: str = None):
        self.project_root = project_root
        self.pipeline_script = project_root / "update_pipeline.py"
        self.python_path = python_path or sys.executable

    def run(self, steps: list[int] = None, no_futures: bool = True,
            timeout: int = 300) -> bool:
        """
        update_pipeline.py 실행.
        
        Args:
            steps: 실행할 스텝 (기본: [1,2,3,4] 전체)
            no_futures: 선물 데이터 수집 제외 (봇에서는 불필요)
            timeout: 최대 실행 시간 (초)
            
        Returns:
            성공 여부
        """
        if not self.pipeline_script.exists():
            logger.error(f"Pipeline script not found: {self.pipeline_script}")
            return False

        cmd = [self.python_path, str(self.pipeline_script)]

        if steps:
            cmd.extend(["--steps"] + [str(s) for s in steps])

        if no_futures:
            cmd.append("--no-futures")

        logger.info(f"Running pipeline: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                logger.info("Pipeline completed successfully")
                # 마지막 몇 줄만 로그
                for line in result.stdout.strip().split("\n")[-5:]:
                    logger.debug(f"  {line}")
                return True
            else:
                logger.error(f"Pipeline failed (exit code {result.returncode})")
                for line in result.stderr.strip().split("\n")[-10:]:
                    logger.error(f"  {line}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Pipeline timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}", exc_info=True)
            return False