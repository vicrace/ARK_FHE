# Source Code

This repository contains the implementation of **ARK**, introduced in the paper:  **"ARK: Adaptive Rotation Key Management for Fully Homomorphic Encryption Targeting Memory-Efficient Deep Learning Inference."**

ARK enhances Fully Homomorphic Encryption (FHE)-based deep learning inference by introducing adaptive rotation key management to improve memory efficiency. The implementation builds upon the **ConvFHE** [1] and includes configurable options for memory- or speed-prioritized execution.

---

## Setup Requirements

1. **Install Go 1.16.6 or higher** from [https://go.dev/](https://go.dev/).

2. **Install Required Go Packages**  
   ```bash
   go get -u golang.org/x/crypto/...
   go get -u github.com/dwkim606/test_lattigo

**NOTE**: Ensure you install the package versions specified by the commands above, as compatibility issues may arise with newer versions.

## Dataset Preparation

**Download Dataset** at [datasetLink](https://drive.google.com/drive/folders/1zLTzJ58E_CDtqvnPv8t9YtgkDaHouWWn), and place all the downloaded folders into the same directory as the source code.

## Execution

1. **Single Convolution** - Perform single convolution tests for ConvFHE baseline and with ARK implementations on various configurations.  
  #### Arguments
  - `conv`: Selects the convolution test mode.
  - **Kernel width**: `3`, `5`, or `7`.
  - **Batch count**: `1`, `2`, or `3`, corresponding to `16`, `64`, and `256` batches, respectively.
  - **Number of test runs**: From `1` to `10`.
  - **Configuration**: `true` for memory-prioritized execution, `false` for speed-prioritized execution.
  - **Example Command** : Run a convolution with kernel width 3, 16 batches, 5 test runs, and memory-prioritized execution.
     ```bash
     go run *.go conv 3 1 5 true
  - **For Window**
    ```bash
     run.bat conv 3 1 5 true

2. **ResNet Evaluation** - Evaluate ResNet using ARK and ConvFHE with CIFAR10 datasets.
  #### Arguments
  - `resnet`: Specifies ResNet evaluation mode.
  - **Kernel width**: `3`, `5`, or `7`.
  - **Number of layers**: `8`, `14`, or `20`.
  - **Wideness Factor**: `1` and `3`.
  - **Number of tests**: From `1` to `100`.
  - **Configuration**: `true` for memory-prioritized execution, `false` for speed-prioritized execution.
  - **Example Command** : Run ResNet with kernel width 3, 20 layers, wideness factor 1, 10 test runs, and memory-prioritized execution.
     ```bash
     go run *.go resnet 3 20 1 10 true
  - **For Window**
    ```bash
     run.bat resnet 3 20 1 10 true
    
  - The following ResNet configurations are available for evaluation:
    ```bash
    resnet 3 20 1 (1 to 100) (true/false)
    resnet 5 20 1 (1 to 100) (true/false)
    resnet 7 20 1 (1 to 100) (true/false)
    resnet 5 8 3 (1 to 100) (true/false)
    resnet 3 14 3 (1 to 100) (true/false)
    resnet 3 20 3 (1 to 100) (true/false)

**NOTE**: ConvFHE baseline evaluation requires up to 100GB of memory.

[1] D. Kim and C. Guyot, “Optimized privacy-preserving cnn inference with fully homomorphic encryption,” IEEE Transactions on Information Forensics and Security, vol. 18, pp. 2175–2187, 2023. doi:10.1109/TIFS.2023.3263631
