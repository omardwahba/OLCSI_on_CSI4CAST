eval "$(mamba shell hook --shell bash)"
mamba activate ol-csi-env

python -m src.utils.inspect_data --cm A --ds 30e-9 --ms 10 --is_train --is_U2D --n_samples 2 --batch_size 2 --no_shuffle