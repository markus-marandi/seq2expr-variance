mamba install -c conda-forge \
    pytorch pytorch-cuda=11.8 torchvision torchaudio \
    pandas numpy scipy scikit-learn tqdm

mamba install \
    "pytorch=2.7=*cuda121*" \
    torchvision torchaudio \
    pandas numpy scipy scikit-learn tqdm \
    -c conda-forge

mkdir -p logs
sbatch flashzoi_array.sbatch

tail -f logs/fz_test_13649675.out