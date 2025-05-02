### ⚠️ **Important** 
Avoid manual input normalization, as the TensorFlow model already includes a Rescaling(1./255) layer immediately after the input layer.
Applying additional normalization may result in near-zero pixel values, causing the model to consistently predict the same class.
I initially overlooked this and spent several hours troubleshooting the issue, mistakenly thinking the problem was related to the saved (Keras) or converted (TFLite) model.

### 🚨 **Warning**
This model is not highly accurate and is intended primarily for educational purposes.
To improve its performance, consider increasing the size of the training dataset or adjusting the neural network architecture.
