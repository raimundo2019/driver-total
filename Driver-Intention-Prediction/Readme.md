# Driver Intention Prediction in Unmarked Roundabouts

This repository contains the source code and experiments associated with the research paper:

**"Viability to Predict Driver Intentions in Unmarked Roundabouts using Vehicle and Contextual Information"**  
R. Vázquez, F. R. Masson – Universidad Tecnológica Nacional & Universidad Nacional del Sur-CONICET

---

## 📖 Abstract
This study introduces a predictive model based on Long Short-Term Memory (LSTM) and Convolutional Neural Networks (CNN) to anticipate driver intentions (Go Straight, Turn Right, Turn Left) in unsignalized roundabouts.  
The models integrate:
- **Kinematic data** (position, velocity, acceleration)  
- **Vehicle positioning** with fixed reference points  
- **Contextual texture features** extracted from the road environment  

The experimental results demonstrate promising predictive performance and robustness in real-world scenarios, highlighting the potential for integration into **Advanced Driver Assistance Systems (ADAS)** and autonomous driving pipelines.

---

## 📂 Repository Structure
---

## 🚀 Usage

1. **Clone this repository:**
   ```bash
   git clone https://github.com/your-username/Driver-Intention-Prediction.git
   cd Driver-Intention-Prediction
pip install -r requirements.txt
python src/trainers/trainer_cnn_rnn_cols_9.py
python src/trainers/trainer_cnn_rnn_cols_10.py
python src/trainers/trainer_9cols.py
python src/trainers/trainer_10cols.py
python src/evaluation/roc_rnn_cnn_cols_9.py
python src/evaluation/roc_rnn_cnn_cols_10.py
python src/evaluation/test_9cols.py
python src/evaluation/test_10cols.py
