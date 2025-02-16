# Assignment 3

## CNN Architecture Used

<img width="500" alt="image" src="https://github.com/user-attachments/assets/b508d3e5-23a0-430c-ab65-f3ea1dc8bcb0" />




<img width="604", height="200" alt="image" src="https://github.com/user-attachments/assets/8f72a97d-c298-49d1-a3aa-87ff630e36e6" />

### Conv Blocks:
- **Conv Block 1**: Conv (kernel size ( k x k )) → BatchNorm → Activation (ReLU or LeakyReLU) → MaxPool
- **Conv Block 2**: Conv (kernel size ( k x k )) → BatchNorm → Activation (ReLU or LeakyReLU) → MaxPool
- **Conv Block 3**: Conv (kernel size ( k x k )) → BatchNorm → Activation (ReLU or LeakyReLU) → MaxPool
- **GlobalAveragePooling2D + Fully Connected Layers (Dropout included)**
- **Softmax Activation for Classification**

## Experiment Details

We experimented with different kernel sizes (**k x k**)  in each Conv Block (**1x1, 3x3, 5x5**), different activation functions (**ReLU** or **LeakyReLU**), and different dropout ratios  (**0.25, 0.3, 0.5, 0.6, 0.75**) . The goal was to obtain the best model with the highest test accuracy for the **UCMerced Dataset** (**70% Training, 10% Validation, 20% Test**).


---

## Results
Training was conducted for **100 epochs**, and the model with the best validation accuracy was selected for testing.

---

## **Part 1: ReLU as Activation Function**

| Model     | Dropout | Kernel Sizes               | Best Val Accuracy | Best Test Accuracy |
|-----------|---------|----------------------------|------------------|------------------|
| **Model 1** | 0.25    | (3×3, 3×3, 3×3)           | -                | -                |
| **Model 2** | 0.5     | (3×3, 3×3, 3×3)           | -                | -                |
| **Model 3** | 0.75    | (3×3, 3×3, 3×3)           | -                | -                |
| **Model 4** | 0.25    | (5×5, 3×3, 3×3)           | -                | -                |
| **Model 5** | 0.25    | (3×3, 3×3, 5×5)           | -                | -                |
| **Model 6** | 0.25    | (1×1, 3×3, 3×3)           | -                | -                |
| **Model 7** | 0.25    | (3×3, 3×3, 1×1)           | **82.13%**       | **78.35%**       |
| **Model 8** | 0.25    | (1×1, 3×3, 5×5)           | -                | -                |


### **Best Model (ReLU Activation)**: Model 7

![Best Model (ReLU)](https://github.com/user-attachments/assets/0150ea28-8591-4eb5-96cd-46c04ba1fdce)

#### **Model Summary:**

| Layer (type)                         | Output Shape                | Param #      |
|--------------------------------------|-----------------------------|-------------|
| conv2d_33 (Conv2D)                   | (None, 128, 128, 32)        | 2,432       |
| batch_normalization_33 (BatchNormalization) | (None, 128, 128, 32) | 128         |
| activation_33 (Activation)           | (None, 128, 128, 32)        | 0           |
| max_pooling2d_33 (MaxPooling2D)      | (None, 64, 64, 32)          | 0           |
| conv2d_34 (Conv2D)                   | (None, 64, 64, 64)          | 18,496      |
| batch_normalization_34 (BatchNormalization) | (None, 64, 64, 64)  | 256         |
| activation_34 (Activation)           | (None, 64, 64, 64)          | 0           |
| max_pooling2d_34 (MaxPooling2D)      | (None, 32, 32, 64)          | 0           |
| conv2d_35 (Conv2D)                   | (None, 32, 32, 128)         | 8,320       |
| batch_normalization_35 (BatchNormalization) | (None, 32, 32, 128) | 512         |
| activation_35 (Activation)           | (None, 32, 32, 128)         | 0           |
| max_pooling2d_35 (MaxPooling2D)      | (None, 16, 16, 128)         | 0           |
| global_average_pooling2d_10 (GlobalAveragePooling2D) | (None, 128) | 0  |
| dense_22 (Dense)                     | (None, 128)                 | 16,512      |
| dropout_11 (Dropout)                 | (None, 128)                 | 0           |
| dense_23 (Dense)                     | (None, 21)                  | 2,709       |


```
Total params: 49,365 (192.83 KB)
Trainable params: 48,917 (191.08 KB)
Non-trainable params: 448 (1.75 KB)
```

### **Grad-CAM Visualization:**
![Grad-CAM](https://github.com/user-attachments/assets/db689dcd-8fab-442d-9aad-499eff41533d)

---

## **Part 2: LeakyReLU (0.01) as Activation Function**

| Model     | Dropout | Kernel Sizes               | Best Val Accuracy | Best Test Accuracy |
|-----------|---------|----------------------------|------------------|------------------|
| **Model 1** | 0.3     | (3×3, 3×3, 3×3)           | -                | -                |
| **Model 2** | 0.6     | (3×3, 3×3, 3×3)           | -                | -                |
| **Model 3** | 0.75    | (3×3, 3×3, 3×3)           | -                | -                |
| **Model 4** | 0.3     | (5×5, 3×3, 3×3)           | -                | -                |
| **Model 5** | 0.3     | (3×3, 3×3, 5×5)           | -                | -                |
| **Model 6** | 0.3     | (5×5, 5×5, 5×5)           | -                | -                |
| **Model 7** | 0.3     | (1×1, 3×3, 3×3)           | **83.15%**       | **81.73%**       |
| **Model 8** | 0.3     | (3×3, 3×3, 1×1)           | -                | -                |


### **Best Model (LeakyReLU Activation)**: Model 7

![Best Model (LeakyReLU)](https://github.com/user-attachments/assets/5a60e32e-6cdc-4eb4-9b7c-bd722aa6593e)

#### **Model Summary:**

| Layer (type)                         | Output Shape                | Param #      |
|--------------------------------------|-----------------------------|-------------|
| input_layer_4 (InputLayer)           | (None, 128, 128, 3)         | 0           |
| conv2d_12 (Conv2D)                   | (None, 128, 128, 32)        | 128         |
| batch_normalization_12 (BatchNormalization) | (None, 128, 128, 32) | 128         |
| leaky_re_lu_16 (LeakyReLU)           | (None, 128, 128, 32)        | 0           |
| max_pooling2d_12 (MaxPooling2D)      | (None, 64, 64, 32)          | 0           |
| conv2d_13 (Conv2D)                   | (None, 64, 64, 64)          | 18,496      |
| batch_normalization_13 (BatchNormalization) | (None, 64, 64, 64)  | 256         |
| leaky_re_lu_17 (LeakyReLU)           | (None, 64, 64, 64)          | 0           |
| max_pooling2d_13 (MaxPooling2D)      | (None, 32, 32, 64)          | 0           |
| conv2d_14 (Conv2D)                   | (None, 32, 32, 128)         | 73,856      |
| batch_normalization_14 (BatchNormalization) | (None, 32, 32, 128) | 512         |
| leaky_re_lu_18 (LeakyReLU)           | (None, 32, 32, 128)         | 0           |
| max_pooling2d_14 (MaxPooling2D)      | (None, 16, 16, 128)         | 0           |
| global_average_pooling2d_4 (GlobalAveragePooling2D) | (None, 128) | 0  |
| dense_8 (Dense)                      | (None, 128)                 | 16,512      |
| leaky_re_lu_19 (LeakyReLU)           | (None, 128)                 | 0           |
| dropout_4 (Dropout)                  | (None, 128)                 | 0           |
| dense_9 (Dense)                      | (None, 21)                  | 2,709       |


```
Total params: 112,597 (439.83 KB)
Trainable params: 112,149 (438.08 KB)
Non-trainable params: 448 (1.75 KB)
```

### **Grad-CAM Visualization:**
![Grad-CAM](https://github.com/user-attachments/assets/ba0d9f88-f065-43f2-955e-6900d8161b72)

---

## **Colab Links**
- **Part 1: CNN Architecture with ReLU Activation** → [Colab Notebook](https://colab.research.google.com/drive/13KHPbInQRBPFz_22U4aOXWz_WgNgXf_B#scrollTo=oWZiyaG-auD7)
- **Part 2: CNN Architecture with LeakyReLU (0.01) Activation** → [Colab Notebook](https://colab.research.google.com/drive/1fI3PFIwaAqyts_JlzwSLjfjLuoGZUoVQ?usp=sharing)
