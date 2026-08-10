

### 1
```
@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    offset = tl.arange(0, B0)
    mask = offset < N0
    x = tl.load(x_ptr + offset, mask)
    tl.store(z_ptr + offset, x + 10, mask)
```

### 2
```
@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    pid = tl.program_id(axis=0)
    blk_start = pid * B0
    offset = blk_start + tl.arange(0, B0)
    mask = offset < N0
    x = tl.load(x_ptr + offset, mask)
    tl.store(z_ptr + offset, x + 10, mask)
```

### 3
```
@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):

    pid = tl.program_id(axis=0)
    x_offset = tl.arange(0, B0)
    y_offset = tl.arange(0, B1)
    x_mask = x_offset < N0
    y_mask = y_offset < N1
    
    x = tl.load(x_ptr + x_offset, x_mask)
    y = tl.load(y_ptr + y_offset, y_mask)
    
    # out = y[:, None] + x[None, :]
    out = y[:, None] + x

    # z_mask = y_mask[:, None] & x_mask[None, :]
    z_mask = y_mask[:, None] & x_mask
    # z_offset = y_offset[:, None] * N0 + x_offset[None, :]
    z_offset = y_offset[:, None] * N0 + x_offset

    tl.store(z_ptr + z_offset, out, z_mask)
```

### 4
```
@triton.jit
def add_vec_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    # not used, but helpful to get block dims
    # x_n_blk = tl.num_programs(0)
    # y_n_blk = tl.num_programs(1)
    # print("n blocks", x_n_blk, y_n_blk)

    x_offset = pid_0*B0 + tl.arange(0, B0)
    y_offset = pid_1*B1 + tl.arange(0, B1)

    x_mask = x_offset < N0
    y_mask = y_offset < N1
    
    x = tl.load(x_ptr + x_offset, x_mask)
    y = tl.load(y_ptr + y_offset, y_mask)
    
    out = y[:, None] + x

    z_mask = y_mask[:, None] & x_mask

    # mental model and memo, y and x offsets are already absolute.
    # so linear index lookup will boil down to the classic: y_idx*(x_size) + x_idx
    z_offset = y_offset[:, None] * N0 + x_offset

    tl.store(z_ptr + z_offset, out, z_mask)
```

### 5
```
@triton.jit
def mul_relu_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    x_offset = pid_0*B0 + tl.arange(0, B0)
    y_offset = pid_1*B1 + tl.arange(0, B1)

    x_mask = x_offset < N0
    y_mask = y_offset < N1
    
    x = tl.load(x_ptr + x_offset, x_mask)
    y = tl.load(y_ptr + y_offset, y_mask)
    
    temp = y[:, None] * x
    out = tl.where(temp > 0, temp, 0)
    
    z_mask = y_mask[:, None] & x_mask
    z_offset = y_offset[:, None] * N0 + x_offset

    tl.store(z_ptr + z_offset, out, z_mask)
```

### 6
```
@triton.jit
def mul_relu_block_back_kernel(x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    x_offset = pid_0*B0 + tl.arange(0, B0)
    y_offset = pid_1*B1 + tl.arange(0, B1)
    
    # 2D tile offset
    # a neat way to index a B1xB0 tile from the N1xN0 tensor
    mat_offset = y_offset[:, None] * N0 + x_offset[None, :]

    y_mask = y_offset < N1
    mask = (y_offset[:, None] < N1) & (x_offset[None, :] < N0)

    x = tl.load(x_ptr + mat_offset, mask)
    y = tl.load(y_ptr + y_offset, y_mask)
    dz = tl.load(dz_ptr + mat_offset, mask)
    
    temp = x * y[:, None]
    out = tl.where(temp > 0, dz * y[:, None], 0)
    
    tl.store(dx_ptr + mat_offset, out, mask)
```

### 7
@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    row_id = tl.program_id(0)
    row_start = row_id * T

    total_sum = 0.0
    for chunk_idx in range(0, T, B1):
        col_start = chunk_idx + tl.arange(0, B1)
        mask = col_start < T
        x = tl.load(row_start + x_ptr + col_start, mask)
        total_sum += tl.sum(x, 0)
        
    tl.store(z_ptr + row_id, total_sum)

### 8
@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    row_start = pid_0 * T
    log2_e = 1.44269504

    # find max
    max_x = -float("inf")
    for chunk_idx in range(0, T, B1):
        col_start = chunk_idx + tl.arange(0, B1)
        mask = col_start < T
        x = tl.load(row_start + x_ptr + col_start, mask)
        max_x = tl.maximum(tl.max(x, axis=0), max_x)

    # find exp and sum
    x_sum_exp = 0.0
    for chunk_idx in range(0, T, B1):
        col_start = chunk_idx + tl.arange(0, B1)
        mask = col_start < T
        x = tl.load(row_start + x_ptr + col_start, mask)
        tmp = tl.exp2((x-max_x)*log2_e)
        tl.store(row_start + z_ptr + col_start, tmp, mask)
        x_sum_exp += tl.sum(tmp, axis=0)

    # normalize
    for chunk_idx in range(0, T, B1):
        col_start = chunk_idx + tl.arange(0, B1)
        mask = col_start < T
        x = tl.load(row_start + z_ptr + col_start, mask)
        tl.store(row_start + z_ptr + col_start, x / x_sum_exp, mask)
