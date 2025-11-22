python experiments/sample.py \
  --full_tsv data/dataset_small_2k_normalized.tsv \
  --n_runs 3 \
  --out_dir experiments/multi_runs \
  --train_script new_src/train_supervised.py \
  --window_eval_script new_src/window_eval_plot.py \
  --find_zfp_script new_src/find_zero_fp.py \
  --vocab checkpoints/vocab_n.json \
  --epochs 6 \
  --batch_size 64