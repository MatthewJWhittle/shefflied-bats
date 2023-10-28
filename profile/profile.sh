python -m cProfile -o profile/output.pstats profile/load_dataset.py
snakeviz profile/output.pstats
