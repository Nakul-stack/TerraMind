# TerraMind Federated Model (AdvisorNet)

> [!IMPORTANT]
> The TerraMind Federated Model represents our privacy-first approach to predictive agriculture. Utilizing the **AdvisorNet** architecture and the **Flower (flwr)** framework, this system trains across decentralized nodes (representing 28 agricultural states in India) without pooling raw data on a central server.

---

## 🔒 The Federated Learning Concept

In traditional machine learning, all local farm data (soil NPK, location, weather patterns) is uploaded to a central database for model training. This poses significant data privacy risks.

**Federated Learning (FL)** solves this by:
1. Sending the *model* down to the individual users (or state-level proxy servers).
2. Training the model locally on their raw, private data.
3. Sending only the calculated *weight updates* back to the central server.
4. Aggregating all updates to form a smarter global model.

*None of the local raw data is ever exposed to the central server or other states.*

---

## 🧠 AdvisorNet Architecture

Instead of managing 5 different independent sub-models like the Centralized Standard Pipeline, the Federated system uses `AdvisorNet`—a multi-task Neural Network written in **PyTorch**.

It consists of a **Shared Representation Layer** handling feature extraction, followed by 5 distinct task "heads":

1. **Crop Recommendation**: Multi-class classification head for selecting optimal crops.
2. **Yield Prediction**: Regression head estimating Ton/Hectare yield.
3. **Sunlight Advisory**: Regression head predicting optimal sunlight hours.
4. **Irrigation Type**: Multi-class classification head for determining the best watering method.
5. **Irrigation Quantity**: Regression head outputting daily water requirement (mm).

---

## 🚀 Running the FL Simulation

To recreate the production federated models, you can run the built-in simulation which spins up 28 virtual clients to emulate the Indian states.

### Basic Non-IID Run (Production Default)
By default, the simulation uses Non-Independent and Identically Distributed (Non-IID) data grouping. This correctly mimics real life, where, for instance, Kerala's data distribution looks vastly different from Punjab's.
```bash
python -m federated.run_simulation --rounds 20
```

### Advanced Flags
- `--iid`: Force an IID (uniform) split across clients for baseline debugging.
- `--dp`: Enable Differential Privacy tracking (reports `epsilon` spend during training).
- `--rounds N`: Adjust the number of server aggregation rounds (e.g., `--rounds 5` for a quick test).

Running the simulation automatically outputs serialized PyTorch weights, Scalers, Label Encoders, and a `federated_model_metadata.json` dict directly into `federated/results/`.

---

## ⚡ Production Inference

The `inference.py` module exposes the `FederatedAdvisor` class. This class is heavily utilized by TerraMind's FastAPI backend.

The `FederatedAdvisor` does **not** rely on the `Flower` runtime, enabling highly performant cold-starts. It loads the compiled artifacts from `federated/results/` and exposes a `.predict(N, P, K, ph, temperature, humidity, rainfall)` function mimicking the standard pipeline but guaranteeing absolute data privacy.

*When queried, the API explicitly annotates the inference response with `"privacy_guarantee": "Input data not used in training..."`.*

---

## 📊 Technical Specifications
- **Aggregation Strategy**: `FedAvg` (Federated Averaging).
- **Partition Engine**: `federated/data_partitioner.py` handles creating state-level isolated data shards.
- **Accuracy Gap**: Post-simulation, an automated breakdown of *Federated vs Centralized* accuracy is presented, highlighting the minimal tradeoffs encountered when enforcing absolute data privacy in precision agriculture.
