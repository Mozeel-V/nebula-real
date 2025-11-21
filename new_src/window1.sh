python new_src/window_eval_plot.py \
  --data_file data/dataset_small_2k_normalized.tsv \
  --ckpt checkpoints/run_weighted_w0_8/best.pt \
  --vocab checkpoints/vocab_n.json \
  --out_dir new_results/window_eval_before \
  --events_per_window 16 --stride_events 4 --batch_size 256 --max_len 256
