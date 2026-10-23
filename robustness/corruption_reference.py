import numpy as np



def generate_random_index(length, seed=2024):
    """
    Generate fixed random order.

    Used for:
    - fixed missing nodes
    - fixed missing time steps

    """

    rng = np.random.RandomState(seed)

    return rng.permutation(length)



# ==========================================================
# Node Missing
# ==========================================================

def node_missing(
        data,
        rate=0.1,
        seed=2024):

    """
    Random Node Missing

    Args:
        data:
            [sample, time, node, feature]

        rate:
            0.1 / 0.2 / 0.3

        seed:
            random seed

    Example:
        10% nodes missing
        20% nodes missing
        30% nodes missing

    """

    data = data.copy()


    N = data.shape[2]


    # number of missing nodes
    missing_num = int(N * rate)


    # fixed random nodes
    nodes = generate_random_index(
        N,
        seed
    )


    missing_nodes = nodes[:missing_num]


    # selected nodes are missing
    # all time steps become zero

    data[:, :, missing_nodes, 0] = 0


    return data



# ==========================================================
# Gaussian Noise
# ==========================================================


def gaussian_noise(
        data,
        rate=0.1,
        noise_mean=10,
        noise_std=500,
        seed=2024):

    """
    Random Gaussian Noise Injection

    Noise:
        epsilon ~ N(noise_mean, noise_std^2)

    Args:
        data:
            [sample,time,node,feature]

        rate:
            ratio of corrupted data points

            0.1 / 0.2 / 0.3

        noise_mean:
            Gaussian noise mean

        noise_std:
            Gaussian noise std

    """

    data = data.copy()

    rng = np.random.RandomState(seed)


    # traffic flow feature
    target = data[..., 0]


    # total number of data points
    total_num = target.size


    # number of noisy points
    noise_num = int(total_num * rate)


    # randomly select corrupted positions
    index = rng.choice(
        total_num,
        noise_num,
        replace=False
    )


    # generate Gaussian noise
    noise = rng.normal(
        loc=noise_mean,
        scale=noise_std,
        size=noise_num
    )


    # flatten for random position assignment
    target_flat = target.reshape(-1)


    # add noise
    target_flat[index] += noise


    # restore shape
    target = target_flat.reshape(
        target.shape
    )


    # ===============================
    # Non-negative constraint
    # ===============================

    target = np.maximum(
        target,
        0
    )


    data[...,0] = target


    return data



def node_contiguous_gap(
        data,
        rate=0.1,
        block_length=3,
        seed=2024):
    """
    Node-wise contiguous temporal missingness.

    Args:
        data:
            [sample, time, node, feature]

        rate:
            Missing ratio for EACH node.
            e.g. 0.1 / 0.2 / 0.3

        block_length:
            Length of each contiguous missing block.
            Default = 3 time steps.

        seed:
            Random seed.

    Description:
        Each node independently selects contiguous temporal
        blocks of length 3.

        Every node has the SAME overall missing ratio,
        but different nodes have different missing positions.

        Only traffic flow (feature 0) is set to zero.
    """

    if data.ndim != 4:
        raise ValueError(
            "Expected data shape [sample, time, node, feature]."
        )

    if not 0 <= rate <= 1:
        raise ValueError(
            "rate must be between 0 and 1."
        )

    data = data.copy()

    S, T, N, F = data.shape

    if T % block_length != 0:
        raise ValueError(
            f"T={T} must be divisible by block_length={block_length}."
        )

    if F < 1:
        raise ValueError(
            "At least one feature is required."
        )

    rng = np.random.RandomState(seed)

    # Number of candidate 3-step blocks in each sample
    blocks_per_sample = T // block_length

    # Total number of candidate blocks for each node
    total_blocks_per_node = S * blocks_per_sample

    # Every node uses the same number of missing blocks
    num_missing_blocks = int(
        total_blocks_per_node * rate
    )

    # Process each node independently
    for node in range(N):

        # Randomly select blocks for this node
        selected_blocks = rng.choice(
            total_blocks_per_node,
            size=num_missing_blocks,
            replace=False
        )

        # Convert global block index to sample index
        sample_indices = (
            selected_blocks // blocks_per_sample
        )

        # Block position inside each sample
        block_indices = (
            selected_blocks % blocks_per_sample
        )

        # Starting time of each 3-step block
        starts = block_indices * block_length

        # Mask the complete contiguous block
        for offset in range(block_length):

            data[
                sample_indices,
                starts + offset,
                node,
                0
            ] = 0

    return data