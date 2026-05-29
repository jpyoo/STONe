# Model Card for STONe (Spatio-Temporal Operator Network)

STONe is a learned, non-autoregressive neural operator framework designed for virtual sensing across physically disjoint domains. It reconstructs complete global spatiotemporal cosmic radiation dose fields at aviation altitudes (10,000 m) over 180-day horizons directly from sparse, ground-based neutron monitor measurements.

## Model Details

### Model Description

STONe (Spatio-Temporal Operator Network) extends the Sequential Deep Operator Network (S-DeepONet) into a new regime of operator-based digital instrumentation. Rather than functioning as a standard forecasting surrogate, STONe formalizes sensing as a learned sequence-to-sequence mapping from accessible sparse data (12 ground stations) to an inaccessible target manifold (10,000 m altitude). The architecture features a temporal encoder branch (evaluating FCN, LSTM, GRU, and Transformer variants) that extracts historical system dynamics, and a spatiotemporal decoder trunk that maps spatial coordinates to basis functions. Synthesized via tensor contraction, STONe produces operational-scale predictions at sub-millisecond latencies, overcoming the domain mismatch and error accumulation typical of traditional autoregressive solvers.

- **Developed by:** Jay Phil Yoo, Kazuma Kobayashi, Souvik Chakraborty, Syed Bahauddin Alam
- **Funded by :** DOE Office of Nuclear Energy’s Nuclear Energy University Program (NEUP) (Award DOE DE-NE0009076) and the National Science Foundation (Awards OAC-2005572, OAC-2320345) via Delta/DeltaAI
- **Model type:** Sequence-to-Sequence Neural Operator / Spatio-Temporal Operator Network
- **Language(s) (NLP):** N/A (Physical Field Prediction)
- **License:** MIT
- **Finetuned from model :** N/A (Trained from scratch)

### Model Sources 

- **Repository:** [Link](https://github.com/jpyoo/STONe/)
- **Paper :** *Sensing Without Colocation: Operator-Based Virtual Instrumentation for Domains Beyond Physical Reach*

## Uses

### Direct Use

Direct prediction of global spatiotemporal cosmic radiation dose fields at 10,000 m altitude using input histories from sparse ground-based neutron monitor arrays over 180-day horizons. 

### Downstream Use 

Deployment on self-contained edge computing devices (e.g., NVIDIA Jetson Orin Nano) co-located at remote ground stations to create an active real-time global virtual sensing infrastructure. This enables operational closed-loop workflows, radiation-aware flight routing, and real-time occupational exposure management.

### Out-of-Scope Use

Predicting fields at altitudes outside of the calibrated 10,000 m target manifold without further retraining or evaluation. Furthermore, predicting acute dose spikes caused by rare solar energetic particle (SEP) events is currently out of scope and requires targeted physics-aware regularization. 

## Bias, Risks, and Limitations

STONe's reconstruction fidelity degrades more significantly if the ground station network density falls below 6 active stations. Feedforward architecture variants (e.g., FCN) may suffer from oversmoothing in high-latitude/polar regions where dose gradients are steepest. Additionally, the deterministic nature of the predictions currently lacks probabilistic uncertainty quantification bounds, which are essential for certifying occupational safety standards.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. For safety-critical monitoring, predictions should be supplemented with high-fidelity transport simulations during extreme SEP events. Future deployments should integrate conformal prediction or generative ensembling to establish calibrated uncertainty bounds.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import torch
import numpy as np
from utils import create_sliding_windows_m2m, SequentialDeepONetDataset
from s_deeponet import SequentialDeepONet

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming init_model handles the instantiation of branch and trunk networks
model_type = 'gru'
# model = init_model(model_type).to(device)

# Load pretrained FP32 weights
# model.load_state_dict(torch.load("path_to_stone_gru_weights.pth"))
# model.eval()

# Forward pass with historical sensor input (Y) and query coordinates (r)
# predicted_dose_field = model(Y_history, spatial_coords)

```

## Training Details

### Training Data

Input observations consist of daily time-series neutron count data from 12 globally distributed NMDB stations (2001-2023). Target reference data consists of effective radiation dose fields at 10,000 m altitude calculated using EXPACS at a 1°x1° latitude-longitude resolution.

### Training Procedure

#### Preprocessing

Data gaps from stations (e.g., ATHN and TERA) were imputed via polynomial interpolation. A sliding window of size $T=K=180$ days was applied to pair historical inputs with future state targets. The dataset was chronologically split into training (45%), validation (10%), and testing (45%) sets to ensure the model evaluates unseen future dynamics.

#### Training Hyperparameters

* **Training regime:** fp32 (Single-precision floating point)

#### Speeds, Sizes, Times

The optimal GRU-STONe variant contains 3.37M parameters. The model was trained using the Adam optimizer (MSE loss, initial LR $1 \times 10^{-3}$, ReduceLROnPlateau scheduler) on an NVIDIA H100 GPU, taking approximately 5.8 minutes per epoch for a maximum of 500 epochs with early stopping (patience of 10). Inference for a full 180-day global rollout completes in 0.048 ms on an H100 GPU and 43.5 ms on an embedded Jetson Orin Nano device.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The 45% chronological test set generated from EXPACS simulation data and NMDB measurements, strictly reserved to test generalizability to unseen temporal environments.

#### Factors

Evaluation disaggregated prediction performance across extended operational forecast horizons (Days 1, 30, 60, 90, 120, 150, 180), geographic high-risk structures (e.g., North America and polar regions), and station sparsity down to minimum viable limits (12 stations down to 2).

#### Metrics

The primary metric for full spatiotemporal reconstruction fidelity is Relative L2 Error. To measure local magnitude errors and physical structure preservation, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) were also analyzed.

### Results

The GRU branch variant demonstrated the lowest long-term drift and strongest reconstruction fidelity, achieving an average Relative L2 Error of 0.0415, RMSE of 0.1311, and MAE of 0.0995 across the full 180-day horizon.

#### Summary

The GRU-STONe architecture successfully converts extreme sparsity (12 ground stations) into operational global dose fields at 10,000 m with sub-millisecond server latencies.

## Model Examination

Ablation studies evaluated the operational stability limit for the number of active ground stations, demonstrating graceful degradation from 12 down to 8 stations (Relative L2 = 0.0905). Extreme 2-station sparsity still managed spatially coherent global fields (Relative L2 = 0.1685). A Ridge regression baseline completely failed to learn cross-domain mappings, validating that the spatiotemporal operator decomposition design is structurally essential rather than an architectural preference.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

* **Hardware Type:** NVIDIA H100 GPU
* **Hours used:** ~48 hours (estimated at max 500 epochs)
* **Cloud Provider:** NCSA Delta/DeltaAI
* **Compute Region:** Illinois, USA
* **Carbon Emitted:** 8.29 kg

## Technical Specifications 

### Model Architecture and Objective

STONe uses a modular S-DeepONet structure:

* **Branch Network (Temporal Encoder):** Uses sequence processing models (e.g., GRU, 3 hidden layers, 128 latent dim) to map the input history ($K_{hist} \times N$) into temporal dynamical state coefficients.
* **Trunk Network (Spatiotemporal Decoder):** Uses a Fully Connected Network (2 hidden layers of 128) to map spatial query coordinates into spatiotemporal basis matrices.
* **Objective:** Mean Squared Error (MSE) minimization via tensor contraction of branch and trunk vectors matching the sequence-to-sequence rollout.

### Compute Infrastructure

Trained on the NCSA Delta/DeltaAI advanced computing cluster.

#### Hardware

* **Training Server:** NVIDIA H100 (approx. 26.6 GB GPU memory footprint utilized)
* **Embedded Edge Deployment:** NVIDIA Jetson Orin Nano (MAXN configuration, ~7.3 W average system power, 143.3 MB peak GPU memory)

#### Software

PyTorch

## Glossary 

* **NMDB:** Neutron Monitor Database, providing worldwide surface-based neutron counts.
* **EXPACS:** EXcel-based Program for calculating Atmospheric Cosmic-ray Spectrum; utilized to calculate the target regulatory reference dose dataset.
* **GCR:** Galactic Cosmic Radiation.

## Model Card Authors

Jay Phil Yoo, Kazuma Kobayashi, Souvik Chakraborty, Syed Bahauddin Alam

## Model Card Contact

jayyoo2@illinois.edu
```
