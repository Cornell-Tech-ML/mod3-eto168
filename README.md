# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: https://minitorch.github.io/

- Overview: https://minitorch.github.io/module3.html

You will need to modify `tensor_functions.py` slightly in this assignment.

- Tests:

```
python run_tests.py
```

- Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# 3.5

## XOR GPU

`!cd mod3-eto168; PYTHONPATH=/content/mod3-eto168 python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05`

```bash
usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 100 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch 0 | Loss: 7.2393 | Correct: 36 | Time: 3.45 sec
Epoch 10 | Loss: 3.6623 | Correct: 46 | Time: 1.44 sec
Epoch 20 | Loss: 3.5224 | Correct: 46 | Time: 1.64 sec
Epoch 30 | Loss: 3.7598 | Correct: 46 | Time: 1.43 sec
Epoch 40 | Loss: 1.4316 | Correct: 46 | Time: 1.49 sec
Epoch 50 | Loss: 1.2228 | Correct: 46 | Time: 1.97 sec
Epoch 60 | Loss: 2.4095 | Correct: 46 | Time: 1.43 sec
Epoch 70 | Loss: 2.8213 | Correct: 47 | Time: 1.44 sec
Epoch 80 | Loss: 2.4954 | Correct: 47 | Time: 1.99 sec
Epoch 90 | Loss: 1.1261 | Correct: 47 | Time: 1.44 sec
Epoch 100 | Loss: 2.1018 | Correct: 48 | Time: 1.42 sec
Epoch 110 | Loss: 1.1187 | Correct: 47 | Time: 1.76 sec
Epoch 120 | Loss: 2.5199 | Correct: 47 | Time: 1.53 sec
Epoch 130 | Loss: 1.5285 | Correct: 48 | Time: 1.42 sec
Epoch 140 | Loss: 1.6814 | Correct: 47 | Time: 1.44 sec
Epoch 150 | Loss: 1.2374 | Correct: 48 | Time: 1.42 sec
Epoch 160 | Loss: 1.9286 | Correct: 49 | Time: 1.48 sec
Epoch 170 | Loss: 1.8889 | Correct: 48 | Time: 1.43 sec
Epoch 180 | Loss: 0.8217 | Correct: 50 | Time: 1.71 sec
Epoch 190 | Loss: 0.7137 | Correct: 50 | Time: 1.43 sec
Epoch 200 | Loss: 1.4790 | Correct: 49 | Time: 1.49 sec
Epoch 210 | Loss: 1.3904 | Correct: 50 | Time: 2.10 sec
Epoch 220 | Loss: 0.8436 | Correct: 50 | Time: 1.43 sec
Epoch 230 | Loss: 1.0713 | Correct: 50 | Time: 1.41 sec
Epoch 240 | Loss: 0.3677 | Correct: 50 | Time: 1.58 sec
Epoch 250 | Loss: 0.7123 | Correct: 50 | Time: 1.47 sec
Epoch 260 | Loss: 0.4980 | Correct: 50 | Time: 1.44 sec
Epoch 270 | Loss: 0.4559 | Correct: 50 | Time: 1.44 sec
Epoch 280 | Loss: 0.3243 | Correct: 50 | Time: 1.42 sec
Epoch 290 | Loss: 0.7624 | Correct: 50 | Time: 1.51 sec
Epoch 300 | Loss: 0.2685 | Correct: 50 | Time: 1.42 sec
Epoch 310 | Loss: 0.7945 | Correct: 50 | Time: 1.60 sec
Epoch 320 | Loss: 0.3145 | Correct: 50 | Time: 1.42 sec
Epoch 330 | Loss: 0.5660 | Correct: 50 | Time: 1.49 sec
Epoch 340 | Loss: 0.3573 | Correct: 50 | Time: 2.08 sec
Epoch 350 | Loss: 0.6602 | Correct: 50 | Time: 1.46 sec
Epoch 360 | Loss: 0.1685 | Correct: 50 | Time: 1.43 sec
Epoch 370 | Loss: 0.6575 | Correct: 50 | Time: 1.67 sec
Epoch 380 | Loss: 0.2273 | Correct: 50 | Time: 1.50 sec
Epoch 390 | Loss: 0.1443 | Correct: 50 | Time: 1.44 sec
Epoch 400 | Loss: 0.1833 | Correct: 50 | Time: 1.42 sec
Epoch 410 | Loss: 0.4069 | Correct: 50 | Time: 1.45 sec
Epoch 420 | Loss: 0.1086 | Correct: 50 | Time: 1.49 sec
Epoch 430 | Loss: 0.3416 | Correct: 50 | Time: 1.44 sec
Epoch 440 | Loss: 0.5194 | Correct: 50 | Time: 1.98 sec
Epoch 450 | Loss: 0.2008 | Correct: 50 | Time: 1.46 sec
Epoch 460 | Loss: 0.2196 | Correct: 50 | Time: 1.41 sec
Epoch 470 | Loss: 0.6315 | Correct: 50 | Time: 2.11 sec
Epoch 480 | Loss: 0.1106 | Correct: 50 | Time: 1.44 sec
Epoch 490 | Loss: 0.1093 | Correct: 50 | Time: 1.43 sec
Average epoch time: 0.9157s
```

## Split GPU

`!cd mod3-eto168; PYTHONPATH=/content/mod3-eto168 python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05`

```bash
Epoch 0 | Loss: 7.5274 | Correct: 29 | Time: 4.21 sec
Epoch 10 | Loss: 6.7011 | Correct: 43 | Time: 1.43 sec
Epoch 20 | Loss: 3.7933 | Correct: 34 | Time: 1.44 sec
Epoch 30 | Loss: 3.8939 | Correct: 40 | Time: 1.84 sec
Epoch 40 | Loss: 4.9414 | Correct: 45 | Time: 1.50 sec
Epoch 50 | Loss: 3.8191 | Correct: 48 | Time: 1.43 sec
Epoch 60 | Loss: 2.5384 | Correct: 47 | Time: 1.54 sec
Epoch 70 | Loss: 2.1558 | Correct: 50 | Time: 1.48 sec
Epoch 80 | Loss: 1.4438 | Correct: 50 | Time: 1.48 sec
Epoch 90 | Loss: 3.8654 | Correct: 49 | Time: 1.43 sec
Epoch 100 | Loss: 1.3009 | Correct: 45 | Time: 1.81 sec
Epoch 110 | Loss: 1.5919 | Correct: 50 | Time: 1.43 sec
Epoch 120 | Loss: 1.5065 | Correct: 49 | Time: 1.48 sec
Epoch 130 | Loss: 1.2813 | Correct: 50 | Time: 2.06 sec
Epoch 140 | Loss: 2.0760 | Correct: 44 | Time: 1.44 sec
Epoch 150 | Loss: 2.6328 | Correct: 43 | Time: 1.42 sec
Epoch 160 | Loss: 1.4937 | Correct: 49 | Time: 1.72 sec
Epoch 170 | Loss: 2.2606 | Correct: 45 | Time: 1.43 sec
Epoch 180 | Loss: 0.7768 | Correct: 50 | Time: 1.42 sec
Epoch 190 | Loss: 2.1269 | Correct: 47 | Time: 1.51 sec
Epoch 200 | Loss: 3.3786 | Correct: 43 | Time: 1.51 sec
Epoch 210 | Loss: 0.7104 | Correct: 49 | Time: 1.43 sec
Epoch 220 | Loss: 0.4997 | Correct: 49 | Time: 1.41 sec
Epoch 230 | Loss: 0.3870 | Correct: 50 | Time: 1.69 sec
Epoch 240 | Loss: 0.5836 | Correct: 48 | Time: 1.45 sec
Epoch 250 | Loss: 0.7487 | Correct: 50 | Time: 1.49 sec
Epoch 260 | Loss: 1.3027 | Correct: 50 | Time: 1.73 sec
Epoch 270 | Loss: 0.0937 | Correct: 49 | Time: 1.41 sec
Epoch 280 | Loss: 0.3146 | Correct: 50 | Time: 1.43 sec
Epoch 290 | Loss: 0.2619 | Correct: 49 | Time: 1.49 sec
Epoch 300 | Loss: 1.3453 | Correct: 46 | Time: 1.43 sec
Epoch 310 | Loss: 0.5239 | Correct: 49 | Time: 1.45 sec
Epoch 320 | Loss: 1.2792 | Correct: 50 | Time: 1.47 sec
Epoch 330 | Loss: 1.9605 | Correct: 45 | Time: 1.84 sec
Epoch 340 | Loss: 0.4969 | Correct: 50 | Time: 1.42 sec
Epoch 350 | Loss: 1.2910 | Correct: 46 | Time: 1.42 sec
Epoch 360 | Loss: 0.5003 | Correct: 50 | Time: 2.04 sec
Epoch 370 | Loss: 0.3773 | Correct: 50 | Time: 1.41 sec
Epoch 380 | Loss: 0.2363 | Correct: 50 | Time: 1.48 sec
Epoch 390 | Loss: 0.0554 | Correct: 50 | Time: 1.52 sec
```

## Simple GPU

`!cd mod3-eto168; PYTHONPATH=/content/mod3-eto168 python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05`

```bash
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 100 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch 0 | Loss: 5.2070 | Correct: 32 | Time: 3.70 sec
Epoch 10 | Loss: 2.2722 | Correct: 46 | Time: 1.45 sec
Epoch 20 | Loss: 1.1583 | Correct: 47 | Time: 1.43 sec
Epoch 30 | Loss: 1.0787 | Correct: 49 | Time: 1.47 sec
Epoch 40 | Loss: 0.7134 | Correct: 47 | Time: 1.56 sec
Epoch 50 | Loss: 0.4919 | Correct: 48 | Time: 1.47 sec
Epoch 60 | Loss: 0.8574 | Correct: 50 | Time: 1.43 sec
Epoch 70 | Loss: 0.2293 | Correct: 48 | Time: 1.93 sec
Epoch 80 | Loss: 0.8328 | Correct: 50 | Time: 1.50 sec
Epoch 90 | Loss: 0.5719 | Correct: 49 | Time: 1.43 sec
Epoch 100 | Loss: 0.9033 | Correct: 50 | Time: 1.87 sec
Epoch 110 | Loss: 0.0464 | Correct: 50 | Time: 1.44 sec
Epoch 120 | Loss: 0.3215 | Correct: 49 | Time: 1.50 sec
Epoch 130 | Loss: 1.1869 | Correct: 50 | Time: 1.44 sec
Epoch 140 | Loss: 0.9640 | Correct: 50 | Time: 1.42 sec
Epoch 150 | Loss: 1.7024 | Correct: 50 | Time: 1.43 sec
Epoch 160 | Loss: 0.9292 | Correct: 49 | Time: 1.50 sec
Epoch 170 | Loss: 0.7149 | Correct: 50 | Time: 2.16 sec
Epoch 180 | Loss: 1.5306 | Correct: 49 | Time: 1.44 sec
Epoch 190 | Loss: 0.0516 | Correct: 50 | Time: 1.48 sec
Epoch 200 | Loss: 1.0632 | Correct: 49 | Time: 1.71 sec
Epoch 210 | Loss: 1.2737 | Correct: 49 | Time: 1.44 sec
Epoch 220 | Loss: 0.1431 | Correct: 49 | Time: 1.44 sec
Epoch 230 | Loss: 0.1490 | Correct: 50 | Time: 1.42 sec
Epoch 240 | Loss: 0.6401 | Correct: 49 | Time: 1.49 sec
Epoch 250 | Loss: 1.4383 | Correct: 49 | Time: 1.56 sec
Epoch 260 | Loss: 0.1857 | Correct: 49 | Time: 1.44 sec
Epoch 270 | Loss: 1.2399 | Correct: 49 | Time: 1.67 sec
Epoch 280 | Loss: 0.7741 | Correct: 49 | Time: 1.45 sec
Epoch 290 | Loss: 0.9416 | Correct: 50 | Time: 1.49 sec
Epoch 300 | Loss: 0.7611 | Correct: 49 | Time: 2.18 sec
Epoch 310 | Loss: 0.0341 | Correct: 50 | Time: 1.44 sec
Epoch 320 | Loss: 0.4940 | Correct: 49 | Time: 1.43 sec
Epoch 330 | Loss: 0.4021 | Correct: 50 | Time: 1.69 sec
Epoch 340 | Loss: 0.0282 | Correct: 49 | Time: 1.43 sec
Epoch 350 | Loss: 0.8678 | Correct: 49 | Time: 1.45 sec
Epoch 360 | Loss: 0.0496 | Correct: 49 | Time: 1.43 sec
Epoch 370 | Loss: 0.3844 | Correct: 50 | Time: 1.70 sec
Epoch 380 | Loss: 1.7254 | Correct: 49 | Time: 1.52 sec
Epoch 390 | Loss: 0.3084 | Correct: 50 | Time: 1.43 sec
```

## large model

`!cd mod3-eto168; PYTHONPATH=/content/mod3-eto168 python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET simple --RATE 0.05`

```bash
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 13 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 13 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 63 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 63 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 63 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 49 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 13 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 7 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 14 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
  warn(NumbaPerformanceWarning(msg))
/usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
Epoch 0 | Loss: 5.6445 | Correct: 38 | Time: 3.67 sec
Epoch 10 | Loss: 0.8079 | Correct: 50 | Time: 1.55 sec
Epoch 20 | Loss: 1.7720 | Correct: 49 | Time: 1.68 sec
Epoch 30 | Loss: 0.1722 | Correct: 50 | Time: 1.54 sec
Epoch 40 | Loss: 1.5156 | Correct: 49 | Time: 1.75 sec
Epoch 50 | Loss: 1.0121 | Correct: 46 | Time: 1.57 sec
Epoch 60 | Loss: 1.4272 | Correct: 49 | Time: 1.52 sec
Epoch 70 | Loss: 0.2807 | Correct: 48 | Time: 1.96 sec
Epoch 80 | Loss: 1.1317 | Correct: 49 | Time: 1.56 sec
Epoch 90 | Loss: 0.0078 | Correct: 49 | Time: 1.52 sec
Epoch 100 | Loss: 1.3598 | Correct: 50 | Time: 1.50 sec
Epoch 110 | Loss: 0.0556 | Correct: 49 | Time: 1.51 sec
Epoch 120 | Loss: 0.0104 | Correct: 50 | Time: 2.18 sec
Epoch 130 | Loss: 0.0198 | Correct: 49 | Time: 1.51 sec
Epoch 140 | Loss: 0.5957 | Correct: 49 | Time: 2.20 sec
Epoch 150 | Loss: 0.2980 | Correct: 48 | Time: 1.51 sec
Epoch 160 | Loss: 0.3393 | Correct: 49 | Time: 1.57 sec
Epoch 170 | Loss: 0.1561 | Correct: 49 | Time: 2.05 sec
Epoch 180 | Loss: 0.1475 | Correct: 49 | Time: 1.51 sec
Epoch 190 | Loss: 0.0347 | Correct: 49 | Time: 1.49 sec
Epoch 200 | Loss: 0.3907 | Correct: 49 | Time: 1.75 sec
Epoch 210 | Loss: 0.0142 | Correct: 48 | Time: 1.50 sec
Epoch 220 | Loss: 2.2647 | Correct: 49 | Time: 1.50 sec
Epoch 230 | Loss: 0.1540 | Correct: 49 | Time: 1.51 sec
Epoch 240 | Loss: 1.3396 | Correct: 50 | Time: 1.52 sec
Epoch 250 | Loss: 0.1520 | Correct: 49 | Time: 2.32 sec
Epoch 260 | Loss: 0.2754 | Correct: 50 | Time: 1.51 sec
Epoch 270 | Loss: 1.0876 | Correct: 49 | Time: 1.51 sec
Epoch 280 | Loss: 0.2038 | Correct: 49 | Time: 1.50 sec
Epoch 290 | Loss: 0.0281 | Correct: 49 | Time: 1.56 sec
Epoch 300 | Loss: 0.4403 | Correct: 49 | Time: 1.76 sec
Epoch 310 | Loss: 1.1733 | Correct: 50 | Time: 1.52 sec
Epoch 320 | Loss: 0.1317 | Correct: 49 | Time: 1.52 sec
Epoch 330 | Loss: 0.6031 | Correct: 50 | Time: 2.04 sec
Epoch 340 | Loss: 0.6283 | Correct: 49 | Time: 1.52 sec
Epoch 350 | Loss: 2.0752 | Correct: 48 | Time: 1.51 sec
Epoch 360 | Loss: 0.0011 | Correct: 49 | Time: 1.50 sec
Epoch 370 | Loss: 0.7200 | Correct: 49 | Time: 1.52 sec
Epoch 380 | Loss: 0.0612 | Correct: 49 | Time: 2.16 sec
Epoch 390 | Loss: 0.0297 | Correct: 49 | Time: 1.54 sec
Epoch 400 | Loss: 0.3833 | Correct: 49 | Time: 1.50 sec
Epoch 410 | Loss: 1.1039 | Correct: 49 | Time: 1.52 sec
Epoch 420 | Loss: 0.0547 | Correct: 48 | Time: 1.58 sec
Epoch 430 | Loss: 0.0306 | Correct: 49 | Time: 1.94 sec
Epoch 440 | Loss: 0.0755 | Correct: 50 | Time: 1.51 sec
Epoch 450 | Loss: 0.1563 | Correct: 50 | Time: 1.50 sec
Epoch 460 | Loss: 1.1643 | Correct: 49 | Time: 1.76 sec
Epoch 470 | Loss: 0.7633 | Correct: 49 | Time: 1.56 sec
Epoch 480 | Loss: 0.0001 | Correct: 48 | Time: 1.54 sec
Epoch 490 | Loss: 1.0544 | Correct: 49 | Time: 1.51 sec
Average epoch time: 0.9667s
```

## Simple CPU

`!cd mod3-eto168; PYTHONPATH=/content/mod3-eto168 python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05`

```bash
Epoch 0 | Loss: 6.8179 | Correct: 44 | Time: 16.66 sec
Epoch 10 | Loss: 2.6884 | Correct: 48 | Time: 0.10 sec
Epoch 20 | Loss: 1.0382 | Correct: 48 | Time: 0.10 sec
Epoch 30 | Loss: 2.4201 | Correct: 46 | Time: 0.11 sec
Epoch 40 | Loss: 0.8661 | Correct: 50 | Time: 0.10 sec
Epoch 50 | Loss: 0.3251 | Correct: 50 | Time: 0.19 sec
Epoch 60 | Loss: 2.3370 | Correct: 50 | Time: 0.10 sec
Epoch 70 | Loss: 1.5142 | Correct: 48 | Time: 0.11 sec
Epoch 80 | Loss: 0.2259 | Correct: 48 | Time: 0.11 sec
Epoch 90 | Loss: 1.1589 | Correct: 50 | Time: 0.10 sec
Epoch 100 | Loss: 0.8874 | Correct: 50 | Time: 0.10 sec
Epoch 110 | Loss: 0.9097 | Correct: 50 | Time: 0.11 sec
Epoch 120 | Loss: 1.1683 | Correct: 50 | Time: 0.11 sec
Epoch 130 | Loss: 0.0810 | Correct: 48 | Time: 0.11 sec
Epoch 140 | Loss: 1.9568 | Correct: 48 | Time: 0.11 sec
Epoch 150 | Loss: 1.1500 | Correct: 48 | Time: 0.17 sec
Epoch 160 | Loss: 0.6589 | Correct: 48 | Time: 0.10 sec
Epoch 170 | Loss: 1.0728 | Correct: 50 | Time: 0.10 sec
Epoch 180 | Loss: 0.0243 | Correct: 48 | Time: 0.10 sec
Epoch 190 | Loss: 1.0800 | Correct: 50 | Time: 0.11 sec
Epoch 200 | Loss: 0.4835 | Correct: 50 | Time: 0.10 sec
Epoch 210 | Loss: 0.7882 | Correct: 50 | Time: 0.10 sec
Epoch 220 | Loss: 0.4945 | Correct: 50 | Time: 0.11 sec
Epoch 230 | Loss: 0.7150 | Correct: 50 | Time: 0.12 sec
Epoch 240 | Loss: 0.0542 | Correct: 50 | Time: 0.11 sec
Epoch 250 | Loss: 0.3323 | Correct: 50 | Time: 0.18 sec
Epoch 260 | Loss: 0.3944 | Correct: 50 | Time: 0.11 sec
Epoch 270 | Loss: 0.0137 | Correct: 50 | Time: 0.11 sec
Epoch 280 | Loss: 0.0163 | Correct: 50 | Time: 0.11 sec
Epoch 290 | Loss: 1.4580 | Correct: 48 | Time: 0.11 sec
Epoch 300 | Loss: 0.4890 | Correct: 50 | Time: 0.11 sec
Epoch 310 | Loss: 0.1190 | Correct: 50 | Time: 0.12 sec
Epoch 320 | Loss: 0.7998 | Correct: 50 | Time: 0.11 sec
Epoch 330 | Loss: 0.5451 | Correct: 50 | Time: 0.11 sec
Epoch 340 | Loss: 0.7316 | Correct: 50 | Time: 0.15 sec
Epoch 350 | Loss: 0.0001 | Correct: 50 | Time: 0.17 sec
Epoch 360 | Loss: 0.4024 | Correct: 50 | Time: 0.11 sec
Epoch 370 | Loss: 0.4488 | Correct: 50 | Time: 0.11 sec
Epoch 380 | Loss: 0.3998 | Correct: 50 | Time: 0.11 sec
Epoch 390 | Loss: 0.0624 | Correct: 50 | Time: 0.10 sec
Epoch 400 | Loss: 1.2516 | Correct: 48 | Time: 0.11 sec
Epoch 410 | Loss: 0.2690 | Correct: 49 | Time: 0.11 sec
Epoch 420 | Loss: 0.0549 | Correct: 50 | Time: 0.11 sec
Epoch 430 | Loss: 0.5891 | Correct: 50 | Time: 0.11 sec
Epoch 440 | Loss: 0.1933 | Correct: 50 | Time: 0.11 sec
Epoch 450 | Loss: 0.8418 | Correct: 50 | Time: 0.21 sec
Epoch 460 | Loss: 0.5392 | Correct: 50 | Time: 0.10 sec
Epoch 470 | Loss: 0.7415 | Correct: 49 | Time: 0.10 sec
Epoch 480 | Loss: 0.4507 | Correct: 50 | Time: 0.11 sec
Epoch 490 | Loss: 0.3384 | Correct: 50 | Time: 0.12 sec
Average epoch time: 0.1028s
```

## XOR CPU

`!cd mod3-eto168; PYTHONPATH=/content/mod3-eto168 python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05`

```bash
Epoch 0 | Loss: 7.3272 | Correct: 35 | Time: 15.41 sec
Epoch 10 | Loss: 5.3579 | Correct: 44 | Time: 0.11 sec
Epoch 20 | Loss: 4.0895 | Correct: 45 | Time: 0.10 sec
Epoch 30 | Loss: 2.6610 | Correct: 46 | Time: 0.10 sec
Epoch 40 | Loss: 2.8831 | Correct: 45 | Time: 0.12 sec
Epoch 50 | Loss: 2.3906 | Correct: 47 | Time: 0.21 sec
Epoch 60 | Loss: 1.4001 | Correct: 46 | Time: 0.11 sec
Epoch 70 | Loss: 3.1907 | Correct: 47 | Time: 0.14 sec
Epoch 80 | Loss: 2.5756 | Correct: 47 | Time: 0.11 sec
Epoch 90 | Loss: 1.3728 | Correct: 47 | Time: 0.10 sec
Epoch 100 | Loss: 1.8827 | Correct: 47 | Time: 0.11 sec
Epoch 110 | Loss: 2.5816 | Correct: 47 | Time: 0.10 sec
Epoch 120 | Loss: 3.1861 | Correct: 49 | Time: 0.11 sec
Epoch 130 | Loss: 1.1003 | Correct: 48 | Time: 0.11 sec
Epoch 140 | Loss: 2.5688 | Correct: 47 | Time: 0.11 sec
Epoch 150 | Loss: 1.1396 | Correct: 48 | Time: 0.23 sec
Epoch 160 | Loss: 1.0946 | Correct: 48 | Time: 0.10 sec
Epoch 170 | Loss: 2.4977 | Correct: 49 | Time: 0.11 sec
Epoch 180 | Loss: 0.6026 | Correct: 48 | Time: 0.10 sec
Epoch 190 | Loss: 0.4682 | Correct: 48 | Time: 0.11 sec
Epoch 200 | Loss: 0.0696 | Correct: 48 | Time: 0.11 sec
Epoch 210 | Loss: 1.0539 | Correct: 48 | Time: 0.11 sec
Epoch 220 | Loss: 0.4703 | Correct: 50 | Time: 0.12 sec
Epoch 230 | Loss: 0.3496 | Correct: 48 | Time: 0.11 sec
Epoch 240 | Loss: 1.2055 | Correct: 48 | Time: 0.11 sec
Epoch 250 | Loss: 2.0133 | Correct: 48 | Time: 0.22 sec
Epoch 260 | Loss: 1.0792 | Correct: 48 | Time: 0.12 sec
Epoch 270 | Loss: 0.5275 | Correct: 49 | Time: 0.11 sec
Epoch 280 | Loss: 1.8731 | Correct: 49 | Time: 0.11 sec
Epoch 290 | Loss: 0.4676 | Correct: 49 | Time: 0.11 sec
Epoch 300 | Loss: 0.3257 | Correct: 49 | Time: 0.11 sec
Epoch 310 | Loss: 2.7255 | Correct: 49 | Time: 0.12 sec
Epoch 320 | Loss: 1.8066 | Correct: 49 | Time: 0.11 sec
Epoch 330 | Loss: 0.2363 | Correct: 50 | Time: 0.10 sec
Epoch 340 | Loss: 1.7601 | Correct: 49 | Time: 0.11 sec
Epoch 350 | Loss: 1.4444 | Correct: 49 | Time: 0.22 sec
Epoch 360 | Loss: 0.1805 | Correct: 49 | Time: 0.19 sec
Epoch 370 | Loss: 1.3794 | Correct: 49 | Time: 0.10 sec
Epoch 380 | Loss: 0.5119 | Correct: 49 | Time: 0.10 sec
Epoch 390 | Loss: 1.7671 | Correct: 49 | Time: 0.12 sec
Epoch 400 | Loss: 0.4233 | Correct: 50 | Time: 0.11 sec
Epoch 410 | Loss: 0.1237 | Correct: 49 | Time: 0.11 sec
Epoch 420 | Loss: 1.8250 | Correct: 49 | Time: 0.11 sec
Epoch 430 | Loss: 0.8935 | Correct: 50 | Time: 0.11 sec
Epoch 440 | Loss: 0.0174 | Correct: 49 | Time: 0.11 sec
Epoch 450 | Loss: 0.6805 | Correct: 49 | Time: 0.17 sec
Epoch 460 | Loss: 1.4067 | Correct: 50 | Time: 0.14 sec
Epoch 470 | Loss: 0.2010 | Correct: 49 | Time: 0.11 sec
Epoch 480 | Loss: 1.4177 | Correct: 49 | Time: 0.12 sec
Epoch 490 | Loss: 1.6543 | Correct: 49 | Time: 0.10 sec
Average epoch time: 0.1001s
```

## Split CPU

`!cd mod3-eto168; PYTHONPATH=/content/mod3-eto168 python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05`

```bash
Epoch 0 | Loss: 7.0357 | Correct: 18 | Time: 15.59 sec
Epoch 10 | Loss: 6.4060 | Correct: 30 | Time: 0.11 sec
Epoch 20 | Loss: 5.3767 | Correct: 38 | Time: 0.12 sec
Epoch 30 | Loss: 4.5670 | Correct: 31 | Time: 0.11 sec
Epoch 40 | Loss: 3.0645 | Correct: 47 | Time: 0.10 sec
Epoch 50 | Loss: 2.6063 | Correct: 40 | Time: 0.10 sec
Epoch 60 | Loss: 4.7169 | Correct: 47 | Time: 0.15 sec
Epoch 70 | Loss: 2.6934 | Correct: 45 | Time: 0.10 sec
Epoch 80 | Loss: 1.5674 | Correct: 48 | Time: 0.11 sec
Epoch 90 | Loss: 2.2910 | Correct: 48 | Time: 0.10 sec
Epoch 100 | Loss: 2.5645 | Correct: 48 | Time: 0.11 sec
Epoch 110 | Loss: 2.8466 | Correct: 48 | Time: 0.12 sec
Epoch 120 | Loss: 2.6119 | Correct: 46 | Time: 0.12 sec
Epoch 130 | Loss: 2.1451 | Correct: 48 | Time: 0.11 sec
Epoch 140 | Loss: 1.1187 | Correct: 48 | Time: 0.11 sec
Epoch 150 | Loss: 1.7776 | Correct: 46 | Time: 0.11 sec
Epoch 160 | Loss: 0.4829 | Correct: 43 | Time: 0.18 sec
Epoch 170 | Loss: 1.6293 | Correct: 48 | Time: 0.10 sec
Epoch 180 | Loss: 1.4402 | Correct: 48 | Time: 0.11 sec
Epoch 190 | Loss: 0.5884 | Correct: 45 | Time: 0.11 sec
Epoch 200 | Loss: 0.3210 | Correct: 48 | Time: 0.12 sec
Epoch 210 | Loss: 2.3595 | Correct: 48 | Time: 0.10 sec
Epoch 220 | Loss: 1.0473 | Correct: 48 | Time: 0.11 sec
Epoch 230 | Loss: 0.8810 | Correct: 48 | Time: 0.11 sec
Epoch 240 | Loss: 1.6813 | Correct: 48 | Time: 0.11 sec
Epoch 250 | Loss: 0.9195 | Correct: 48 | Time: 0.11 sec
Epoch 260 | Loss: 2.2647 | Correct: 47 | Time: 0.13 sec
Epoch 270 | Loss: 2.0137 | Correct: 47 | Time: 0.11 sec
Epoch 280 | Loss: 0.2098 | Correct: 45 | Time: 0.11 sec
Epoch 290 | Loss: 4.7196 | Correct: 41 | Time: 0.12 sec
Epoch 300 | Loss: 0.2144 | Correct: 48 | Time: 0.10 sec
Epoch 310 | Loss: 2.8379 | Correct: 46 | Time: 0.11 sec
Epoch 320 | Loss: 1.2692 | Correct: 49 | Time: 0.11 sec
Epoch 330 | Loss: 0.4169 | Correct: 49 | Time: 0.11 sec
Epoch 340 | Loss: 0.6024 | Correct: 46 | Time: 0.10 sec
Epoch 350 | Loss: 2.4662 | Correct: 46 | Time: 0.11 sec
Epoch 360 | Loss: 2.0997 | Correct: 45 | Time: 0.19 sec
Epoch 370 | Loss: 2.5766 | Correct: 46 | Time: 0.12 sec
Epoch 380 | Loss: 0.5330 | Correct: 48 | Time: 0.11 sec
Epoch 390 | Loss: 1.0231 | Correct: 48 | Time: 0.11 sec
Epoch 400 | Loss: 1.8795 | Correct: 47 | Time: 0.11 sec
Epoch 410 | Loss: 0.4668 | Correct: 47 | Time: 0.10 sec
Epoch 420 | Loss: 2.0683 | Correct: 49 | Time: 0.10 sec
Epoch 430 | Loss: 0.1304 | Correct: 49 | Time: 0.11 sec
Epoch 440 | Loss: 0.3414 | Correct: 48 | Time: 0.11 sec
Epoch 450 | Loss: 1.5551 | Correct: 48 | Time: 0.10 sec
Epoch 460 | Loss: 0.3916 | Correct: 48 | Time: 0.18 sec
Epoch 470 | Loss: 0.8351 | Correct: 48 | Time: 0.10 sec
Epoch 480 | Loss: 0.6480 | Correct: 50 | Time: 0.12 sec
Epoch 490 | Loss: 0.4236 | Correct: 49 | Time: 0.11 sec
Average epoch time: 0.1006s
```
