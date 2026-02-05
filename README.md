# 🚢 Intelligent Monitoring and Control of Shipboard Power Systems

This repository presents a suite of machine learning and reinforcement learning approaches for:

- **Non-Intrusive Load Monitoring (NILM)**
- **Fault Detection**
- **Network Reconfiguration**

These methods are designed to enhance the monitoring, classification, and control of next-generation shipboard power systems using CNNs, GNNs, and RL.

---

## 📌 Project Modules

### 🔌 1. Non-Intrusive Load Monitoring (NILM)

Implements a 1D Convolutional Neural Network (CNN) to classify loads from time-series electrical data. Class imbalance is addressed using the SMOTE algorithm.

#### 🔧 Features

- **Data Handling**:
  - Loads `.mat` files simulating shipboard power system signals.
  - Combines multi-source data into a 3D tensor for CNN input.

- **Class Imbalance**:
  - Uses **SMOTE** (Synthetic Minority Over-sampling Technique) for balancing datasets.

- **Model Architecture**:
  - 1D CNN consisting of:
    - `Conv1D` layers for temporal feature extraction.
    - `MaxPooling` layers to reduce dimensionality.
    - Fully connected `Dense` layers and `Softmax` for multi-class output.

- **Training & Evaluation**:
  - Trained using the **Adam** optimizer with **categorical crossentropy** loss.
  - Accuracy and confusion matrix evaluation.

- **Visualization**:
  - Class distribution plots (before and after SMOTE).
  - Confusion matrix heatmaps.

#### 📖 Reference

Senemmar, S. and Zhang, J., *Wavelet-based Convolutional Neural Network for Non-Intrusive Load Monitoring of Next Generation Shipboard Power Systems*, **Measurement: Sensors**, Vol. 35, 2024, pp. 101298.  
[DOI Link](https://doi.org/10.1016/j.measen.2024.101298)

---

### ⚠️ 2. Non-Intrusive Fault Detection

Implements a **Graph Neural Network (GNN)** using the Spektral library for fault classification from tabular data encoded as graphs.

#### 🔧 Features

- **Data Preprocessing**:
  - Loads and normalizes CSV tabular data with `StandardScaler`.
  - Balances class distribution using `RandomOverSampler`.

- **Graph Construction**:
  - Constructs adjacency matrix.
  - Converts dataset into graphs using `Spektral`'s `Graph` and `Dataset` APIs.

- **Model Architecture**:
  - Two `GCNConv` layers with ReLU activation.
  - Dropout layers for regularization.
  - Final `Dense` + `Softmax` output layer.

- **Training & Evaluation**:
  - Trained with **Adam** optimizer and **categorical crossentropy**.
  - Accuracy/loss metrics, validation curves, and confusion matrix.

- **Visualization**:
  - Confusion matrix heatmap.
  - Training/validation performance curves.

#### 📖 Reference

Senemmar, S., Jacob, R. A., and Zhang, J., *Non-Intrusive Fault Detection in Shipboard Power Systems using Wavelet Graph Neural Networks*, **Measurement: Energy**, Vol. 3, 2024, pp. 100009.  
[DOI Link](https://doi.org/10.1016/j.mee.2024.100009)

---

### 🔄 3. Network Reconfiguration via Reinforcement Learning

Applies **Reinforcement Learning (RL)** using **Proximal Policy Optimization (PPO)** to dynamically reconfigure shipboard power systems modeled in Simulink.

#### 🔧 Key Features

- **Simulink Integration**:
  - Uses MATLAB Engine API for Python to interface with a four-zone MVDC Simulink model.

- **Custom Gym Environment**:
  - Implements an `OpenAI Gym`-compatible environment (`ShipEnvironment`) for simulation-based control.

- **PPO Agent**:
  - Actor-critic PPO agent.
  - Binary action space for controlling breakers, switches, and converters.

- **Reward Function**:
  - Designed to optimize generator power, bus voltage stability, and load service.

- **Visualization**:
  - Plots cumulative rewards and training losses.

#### 📁 Key Files

- `pyinterface.py`: Interface with MATLAB/Simulink and defines environment logic.
- `TrainingPPO.py`: Implements PPO training loop.

---

## 💻 Requirements

- Python 3.x  
- TensorFlow  
- Spektral  
- OpenAI Gym  
- Scikit-learn  
- NumPy, Pandas, Matplotlib  
- MATLAB Engine API for Python

---

## 👨‍🔬 Target Audience

This repository is intended for:

- Researchers in energy systems and smart ships.
- Engineers working on intelligent control and fault detection.
- Developers applying ML/RL to cyber-physical systems.

---

## 📬 Contact

For questions or collaboration, please contact:

**Dr. Jie Zhang**  
The University of Texas at Dallas  
[Research Website](https://personal.utdallas.edu/~jiezhang/)  

---

Let me know if you'd like badges, CI/CD instructions, example scripts, or installation steps added!
