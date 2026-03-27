"""
trading_bot/data/pipeline_runner.py
====================================
기존 update_pipeline.py를 subprocess로 호출.
"""

import subprocess
import sys
import logging
from pathlib import Path

logger = logging.getLogger("bot.pipeline")


class PipelineRunner:
    def __init__(self, project_root: Path, python_path: str = None):
        self.project_root = project_root
        self.pipeline_script = project_root / "update_pipeline.py"
        self.python_path = python_path or sys.executable

    def run(self, steps: list[int] = None, no_futures: bool = True,
            timeout: int = 300) -> bool:
        if not self.pipeline_script.exists():
            logger.error(f"Pipeline not found: {self.pipeline_script}")
            return False

        cmd = [self.python_path, str(self.pipeline_script)]
        if steps:
            cmd.extend(["--steps"] + [str(s) for s in steps])
        if no_futures:
            cmd.append("--no-futures")

        logger.info(f"Pipeline: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, cwd=str(self.project_root),
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                logger.info("Pipeline OK")
                for line in result.stdout.strip().split("\n")[-5:]:
                    logger.debug(f"  {line}")
                return True
            else:
                logger.error(f"Pipeline FAIL (exit {result.returncode})")
                for line in result.stderr.strip().split("\n")[-10:]:
                    logger.error(f"  {line}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Pipeline timeout ({timeout}s)")
            return False
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return False
