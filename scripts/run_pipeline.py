from data_pipeline.pipeline_runner import run_pipeline


if __name__ == "__main__":
    run_pipeline(asset="btc", steps=[1, 2, 3, 4], force=False, collect_futures=True)
