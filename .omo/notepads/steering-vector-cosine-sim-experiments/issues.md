
# Issues

## 2026-03-13: CUDA Tensor to NumPy Conversion Error

**Location:** `experiments.py:compute_cosine_similarity_matrix()`

**Error:**
```
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

**Root Cause:** `torch.stack(vectors).numpy()` fails when tensors are on GPU.

**Fix:** Always use `.cpu().numpy()` to handle both CPU and CUDA tensors:
```python
stacked = torch.stack(vectors)
matrix = stacked.cpu().numpy()  # cpu() is no-op if already on CPU
```
