#!/usr/bin/env python3
"""
Verify MPS (Metal Performance Shaders) setup for Mac training.

Usage:
    python3 scripts/verify_mps.py
"""

import sys
import torch

print("=" * 60)
print("MPS VERIFICATION FOR PHASEGPT")
print("=" * 60)

# 1. PyTorch version
print(f"\n1. PyTorch Version:")
print(f"   {torch.__version__}")

min_version = "2.0.0"
current_version = torch.__version__.split('+')[0]  # Remove +cpu/+cu118 suffix
if current_version < min_version:
    print(f"   ⚠️  PyTorch {min_version}+ recommended for stable MPS support")
else:
    print(f"   ✓ Version is good")

# 2. MPS availability
print(f"\n2. MPS Availability:")
print(f"   MPS built: {torch.backends.mps.is_built()}")
print(f"   MPS available: {torch.backends.mps.is_available()}")

if not torch.backends.mps.is_available():
    print("\n❌ MPS NOT AVAILABLE")
    print("\n   Possible fixes:")
    print("   1. Upgrade PyTorch:")
    print("      pip install --upgrade torch torchvision")
    print("   2. Check macOS version:")
    print("      MPS requires macOS 12.3+ (Monterey or later)")
    print("   3. Verify you have Apple Silicon:")
    print("      This won't work on Intel Macs")
    print("   4. Try uninstalling and reinstalling PyTorch:")
    print("      pip uninstall torch torchvision")
    print("      pip install torch torchvision")
    sys.exit(1)

# 3. Test basic tensor operations
print(f"\n3. Testing MPS Operations:")
try:
    device = torch.device("mps")

    # Test 1: Basic tensor creation
    x = torch.randn(100, 100, device=device)
    print(f"   ✓ Tensor creation: {x.shape}")

    # Test 2: Matrix multiplication
    y = torch.randn(100, 100, device=device)
    z = torch.matmul(x, y)
    print(f"   ✓ Matrix multiplication: {z.shape}")

    # Test 3: Gradient computation
    x.requires_grad = True
    loss = (x ** 2).sum()
    loss.backward()
    print(f"   ✓ Gradient computation: {x.grad.shape}")

    # Test 4: Moving tensors between devices
    cpu_tensor = torch.randn(50, 50)
    mps_tensor = cpu_tensor.to(device)
    cpu_again = mps_tensor.cpu()
    print(f"   ✓ CPU ↔ MPS transfer")

    # Test 5: Common operations
    mean = z.mean()
    std = z.std()
    relu = torch.relu(z)
    print(f"   ✓ Common operations (mean, std, relu)")

except Exception as e:
    print(f"\n❌ MPS TEST FAILED!")
    print(f"   Error: {e}")
    print(f"\n   Fallback options:")
    print(f"   - Use CPU training: --device cpu")
    print(f"   - Report issue: https://github.com/pytorch/pytorch/issues")
    sys.exit(1)

# 4. Memory info
print(f"\n4. System Memory:")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"   Total: {mem.total / (1024**3):.1f} GB")
    print(f"   Available: {mem.available / (1024**3):.1f} GB")
    print(f"   Used: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")

    if mem.available / (1024**3) < 10:
        print(f"   ⚠️  Low available memory - consider closing other apps before training")
except ImportError:
    print("   (Install psutil for detailed memory info: pip install psutil)")

# 5. Performance test
print(f"\n5. Performance Benchmark:")
try:
    import time

    device = torch.device("mps")
    size = 1000

    # Warm-up
    _ = torch.randn(100, 100, device=device) @ torch.randn(100, 100, device=device)

    # Benchmark
    start = time.time()
    for _ in range(10):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        _ = c.sum()
    elapsed = time.time() - start

    gflops = (2 * size**3 * 10) / elapsed / 1e9
    print(f"   Matrix multiply (1000x1000): {elapsed:.2f}s for 10 iterations")
    print(f"   Throughput: ~{gflops:.1f} GFLOPS")

    if gflops < 50:
        print(f"   ⚠️  Performance seems low - check Activity Monitor for GPU usage")
    else:
        print(f"   ✓ Performance looks good")

except Exception as e:
    print(f"   ⚠️  Benchmark failed: {e}")

# 6. Final status
print("\n" + "=" * 60)
print("✅ MPS IS FULLY FUNCTIONAL!")
print("=" * 60)

print(f"\n📝 Next Steps:")
print(f"   1. Validate your dataset:")
print(f"      make validate-data")
print(f"\n   2. Run a quick test (10 steps):")
print(f"      python src/train.py \\")
print(f"        --config configs/v14/dpo_extended_100pairs.yaml \\")
print(f"        --device mps \\")
print(f"        --max-steps 10")
print(f"\n   3. Start full training:")
print(f"      python src/train.py \\")
print(f"        --config configs/v14/dpo_extended_100pairs.yaml \\")
print(f"        --device auto")
print(f"\n   4. Use 'caffeinate' to prevent sleep:")
print(f"      caffeinate -i python src/train.py --config <config> --device mps")

print(f"\n📚 For more info, see: MAC_STUDIO_TRAINING_GUIDE.md")
print(f"\n🍎 Happy training on Mac!\n")
