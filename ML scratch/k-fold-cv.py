import numpy as np

def k_fold_split(n_samples,k=5,shuffle=True,random_state=42):
    if not 2<=k <=n_samples:
        raise ValueError("k must be between 2 and n_samples")
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    folds = np.array_split(indices,k)

    for i in range(k):

        # Current fold is used for validation
        val_indices = folds[i]

        #all remaining for training
        train_indices = np.concatenate(
            [folds[j] for j in range(k) if j!=i]
        )

        yield train_indices , val_indices
