🎬 Movie Reviews Sentiment Analysis

This project is all about understanding how people feel about movies — whether they loved it or hated it — just by reading their reviews. I built this sentiment analysis model that reads a review and predicts if it’s positive or negative.

It’s a fun mix of NLP (Natural Language Processing) and deep learning using RNNs, embeddings, and Python.

🚀 What this project does

Reads a bunch of movie reviews

Cleans and preprocesses the text

Converts words into vectors using embeddings

Uses a SimpleRNN model to understand word sequences

Predicts if the sentiment is positive or negative

Lets you test your own custom review sentences

Basically, it teaches a model how to “feel” text the way humans do.

🧠 Tech Stack

Python

TensorFlow / Keras for building the neural network

NumPy, Pandas, and Matplotlib for data and visuals

Jupyter Notebook for experiments

📁 Project Structure
Movie-Reviews-Sentiment-Analysis/
│
├── embedding.ipynb        # Experiment with word embeddings
├── simplernn.ipynb        # RNN model for training and testing
├── prediction.ipynb       # Quick predictions on custom inputs
├── main.py                # Core script for training or running the model
├── requirements.txt       # All dependencies
└── README.md              # You’re reading it now :)

⚙️ How to run it

Clone the repo

git clone https://github.com/Suraj3155/Movie-Reviews-Sentiment-Analysis.git
cd Movie-Reviews-Sentiment-Analysis


Install the dependencies

pip install -r requirements.txt


(Optional) Download or prepare your movie reviews dataset
You can use the IMDb dataset or any CSV with text and sentiment labels.

Run the main script

python main.py


Or open the Jupyter notebooks (simplernn.ipynb, embedding.ipynb) to explore and train interactively.

🔍 What’s happening under the hood

Texts are tokenized and padded so that all reviews have equal length.

Words are turned into numerical vectors using embeddings.

A SimpleRNN model processes these sequences and learns patterns that define “positive” or “negative” tone.

The trained model can then predict new reviews you give it.

📊 Results

After training, the model achieved decent accuracy on test data.
It performs well at picking up obvious positive or negative tones like:

“This movie was a masterpiece!” → 👍 Positive

“I regret watching this…” → 👎 Negative

(You can experiment with your own reviews in the prediction.ipynb file.)

💡 Future Improvements

Use LSTM or GRU instead of SimpleRNN for better long-term memory

Try BERT or transformers for more context-aware predictions

Build a small web app using Flask or Streamlit for live predictions

Add more data and improve accuracy

🤝 Acknowledgements

Thanks to all the open-source resources and tutorials that helped shape this project, especially around Keras and sentiment analysis examples.
Also thanks to IMDb and similar datasets for providing free text data to experiment with.

🧾 License

Feel free to use, modify, or improve this project for learning purposes.
MIT License © 2025 Suraj Shivankar
