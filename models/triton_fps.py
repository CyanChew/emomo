import torch
import triton
import triton.language as tl

# We'll assume C is 3 (pointclouds in 3D). Use a buffer size that's a power of 2.
MAX_C = tl.constexpr(4)


@triton.jit
def update_distance_kernel(
    xyz_ptr,  # pointer to input point cloud, shape [B, N, 3]
    distances_ptr,  # pointer to per-point distances, shape [B, N]
    current_farthest_ptr,  # pointer to current farthest index for each batch, shape [B]
    B: tl.constexpr,  # batch size
    N: tl.constexpr,  # total number of points
    C: tl.constexpr,  # actual number of channels (e.g. 3)
    stride_xyz_b: tl.constexpr,
    stride_xyz_n: tl.constexpr,
    stride_xyz_c: tl.constexpr,
    stride_dist: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # chunk size (e.g. 1024)
    NUM_CHUNKS: tl.constexpr,  # number of chunks = ceil(N / BLOCK_SIZE)
):
    b = tl.program_id(0)
    current_index = tl.load(current_farthest_ptr + b)
    ch_idx = tl.arange(0, MAX_C)
    # Vectorized load for centroid, masking out indices >= C.
    centroid = tl.load(
        xyz_ptr
        + b * stride_xyz_b
        + current_index * stride_xyz_n
        + ch_idx * stride_xyz_c,
        mask=ch_idx < C,
        other=0.0,
    )

    # Loop over all chunks of points.
    for i in tl.range(NUM_CHUNKS):
        start = i * BLOCK_SIZE
        pid = start + tl.arange(0, BLOCK_SIZE)
        mask = pid < N
        # Load a block of points: shape [BLOCK_SIZE, MAX_C]
        points = tl.load(
            xyz_ptr
            + b * stride_xyz_b
            + pid[:, None] * stride_xyz_n
            + ch_idx[None, :] * stride_xyz_c,
            mask=(pid[:, None] < N) & (ch_idx[None, :] < C),
            other=0.0,
        )
        diff = points - centroid[None, :]
        dist2 = tl.sum(diff * diff, axis=1)
        old_dist = tl.load(distances_ptr + b * stride_dist + pid, mask=mask, other=1e10)
        new_dist = tl.minimum(old_dist, dist2)
        tl.store(distances_ptr + b * stride_dist + pid, new_dist, mask=mask)


@triton.jit
def argmax_kernel(
    distances_ptr,  # pointer to distances, shape [B, N]
    result_ptr,  # pointer to output farthest index per batch, shape [B]
    B: tl.constexpr,
    N: tl.constexpr,
    stride_dist: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # e.g. 1024
    NUM_CHUNKS: tl.constexpr,  # computed as ceil(N / BLOCK_SIZE)
):
    b = tl.program_id(0)
    max_val = -1.0
    max_idx = 0

    for i in tl.range(NUM_CHUNKS):
        start = i * BLOCK_SIZE
        pid = start + tl.arange(0, BLOCK_SIZE)
        mask = pid < N
        d = tl.load(distances_ptr + b * stride_dist + pid, mask=mask, other=-1.0)
        chunk_max = tl.max(d, axis=0)
        chunk_argmax = tl.argmax(d, axis=0)
        cond = chunk_max > max_val
        max_val = tl.where(cond, chunk_max, max_val)
        new_idx = start + chunk_argmax
        max_idx = tl.where(cond, new_idx, max_idx)
    tl.store(result_ptr + b, max_idx)


def triton_farthest_point_sample(xyz: torch.Tensor, npoint: int):
    """
    Args:
       xyz: Tensor of shape [B, N, 3]
       npoint: number of points to sample
    Returns:
       centroids: Tensor of shape [B, npoint] containing indices of sampled points.
    """
    B, N, C = xyz.shape
    device = xyz.device
    distances = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    centroids = torch.empty((B, npoint), device=device, dtype=torch.long)
    current_farthest = torch.randint(0, N, (B,), device=device, dtype=torch.long)

    # Choose a chunk size and compute number of chunks.
    BLOCK_SIZE = 1024  # Must be a power of 2.
    NUM_CHUNKS = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = (B,)

    for i in range(npoint):
        centroids[:, i] = current_farthest
        update_distance_kernel[grid](
            xyz,
            distances,
            current_farthest,
            B,
            N,
            C,
            xyz.stride(0),
            xyz.stride(1),
            xyz.stride(2),
            distances.stride(0),
            BLOCK_SIZE,
            NUM_CHUNKS,
        )
        argmax_out = torch.empty((B,), device=device, dtype=torch.long)
        argmax_kernel[grid](
            distances, argmax_out, B, N, distances.stride(0), BLOCK_SIZE, NUM_CHUNKS
        )
        current_farthest = argmax_out

    return centroids


def test_speed():
    import common_utils
    from pointnet2_utils import farthest_point_sample  # reference implementation

    B, N, C = 64, 107026, 3
    npoint = 1024
    xyz = torch.randn(B, N, C, device="cuda").to(torch.bfloat16)

    stopwatch = common_utils.Stopwatch()
    with stopwatch.time("compile"):
        _ = triton_farthest_point_sample(xyz, npoint)

    for _ in range(10):
        with stopwatch.time("triton"):
            centroids = triton_farthest_point_sample(xyz, npoint)

    xyz = torch.randn(B, 107026, C, device="cuda")
    for _ in range(10):
        with stopwatch.time("torch"):
            centroids = farthest_point_sample(xyz, npoint)
    stopwatch.summary()

    # compare result
    B, N, C = 1, 2500, 3
    npoint = 32
    xyz = torch.rand(B, N, C, device="cuda").to(torch.bfloat16)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    centroids = triton_farthest_point_sample(xyz, npoint)
    print(centroids.sum())

    ###
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    centroids = farthest_point_sample(xyz, npoint)
    print(centroids.sum())


if __name__ == "__main__":
    test_speed()
