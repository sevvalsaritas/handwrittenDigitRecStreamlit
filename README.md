### ✍️ Handwritten Digit & Number Recognition with MNIST and Streamlit

<img width="3774" height="1722" alt="image" src="https://github.com/user-attachments/assets/51a65a22-c355-4e15-a2fb-35761470affe" />

This project is a deep learning web application that recognizes handwritten digits and multi-digit numbers.

Users can draw digits directly on a canvas or upload an image, and the system predicts the numbers using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

The application is built with TensorFlow/Keras for the model and Streamlit for the interactive web interface.

---

# 🚀 Features

- Draw digits on an interactive canvas
- Upload handwritten digit images
- Detect multiple digits written side by side
- Segment digits automatically
- Predict each digit using a trained CNN model
- Show confidence scores
- Display detected digit regions

---

# 🧠 How It Works

The system follows these steps:

1. Input acquisition  
   - Canvas drawing or uploaded image

2. Image preprocessing  
   - Convert image to grayscale  
   - Apply Gaussian blur  
   - Apply Otsu thresholding  
   - Remove noise

3. Digit segmentation  
   - Detect contours  
   - Extract individual digits  
   - Sort digits from left to right

4. Digit classification  
   - Resize each digit to 28x28  
   - Feed it into the trained MNIST CNN model

5. Prediction aggregation  
   - Combine predictions to form the final number

Example:

Input image:
325

Predicted output:
325

---

# 📁 Project Structure

handwrittenDigitRecStreamlit/

app.py  
utils.py  
train.py  
requirements.txt  

model/  
  mnist_cnn.h5  

README.md


app.py → Streamlit web application  
utils.py → Image preprocessing and prediction utilities  
train.py → Model training script  
model/mnist_cnn.h5 → Trained MNIST CNN model  
requirements.txt → Project dependencies  

---

# 🧠 Model Architecture

The project uses a Convolutional Neural Network designed for handwritten digit recognition.

Architecture:

Input (28x28 grayscale)

Conv2D (32 filters)  
Conv2D (32 filters)  
MaxPooling  
Dropout  

Conv2D (64 filters)  
Conv2D (64 filters)  
MaxPooling  
Dropout  

Flatten  
Dense (128)  
Dropout  
Dense (10 softmax)

Loss Function:
Sparse Categorical Crossentropy

Optimizer:
Adam

Dataset:
MNIST

Typical accuracy:
~99%

---

# ⚙️ Installation

Clone the repository:

git clone https://github.com/sevvalsaritas/handwrittenDigitRecStreamlit.git

cd handwrittenDigitRecStreamlit


Install dependencies:

pip install -r requirements.txt

---

# 🏋️ Train the Model

If you want to retrain the model:

python train.py

The trained model will be saved to:

model/mnist_cnn.h5

---

# ▶️ Run the Application

Start the Streamlit app:

streamlit run app.py

Open your browser:

http://localhost:8501

---

# ✏️ Usage

1. Draw a number on the canvas OR upload an image.
2. Click the **Predict** button.
3. The system will detect digits and predict the number.

The app also shows:

- detected digit regions
- individual digit predictions
- confidence scores
- probability distribution (for single digits)

---

# ⚠️ Limitations

The system works best when:

- digits are written clearly
- digits are not overlapping
- digits are written horizontally

Accuracy may decrease when:

- digits touch each other
- handwriting is extremely irregular
- digits are very small

---

# 🔮 Future Improvements

Possible future improvements:

- CRNN based sequence recognition
- object detection based digit localization
- mobile friendly UI
- real-time camera digit recognition
- support for more handwriting datasets

---

# 🛠️ Technologies Used

Python  
TensorFlow / Keras  
OpenCV  
NumPy  
Streamlit  
MNIST Dataset

---

# 📚 References

MNIST Dataset  
http://yann.lecun.com/exdb/mnist/

TensorFlow  
https://www.tensorflow.org/

Streamlit  
https://streamlit.io/

---

# 👩‍💻 Author

Şevval Özlem

GitHub:  
https://github.com/sevvalsaritas
