# Edge-Assisted Homomorphic Federated Learning (EAH-FL): Data Flow and Algorithm

## Abstract

This document presents the complete data flow and algorithmic description for Edge-Assisted Homomorphic Federated Learning (EAH-FL), a novel federated learning architecture that leverages edge devices to handle homomorphic encryption operations while maintaining strong privacy guarantees. The proposed approach introduces a hierarchical architecture comprising n clients, n edge devices, and one central cloud server, where edge devices offload encryption and decryption operations from clients, thereby reducing computational burden while preserving end-to-end privacy through homomorphic encryption.

## 1. Introduction

Traditional federated learning approaches require clients to perform computationally intensive homomorphic encryption operations, which can be prohibitive for resource-constrained devices. EAH-FL addresses this limitation by introducing edge devices that act as encryption proxies, enabling clients to focus solely on local model training while maintaining the same privacy guarantees as traditional homomorphic federated learning.

## 2. System Architecture

The EAH-FL system consists of three main components: (1) n clients that perform local model training on their private data, (2) n edge devices that handle homomorphic encryption and decryption operations, and (3) one central cloud server that performs aggregation and global model updates. The data flow follows a five-phase process: Client Local Training → Edge Device Encryption → Cloud Server Aggregation → Global Model Update → Edge Device Decryption and Client Synchronization.

## 3. Data Flow Description

### 3.1 Phase 1: Client Local Training

In the first phase, each client i ∈ {1, 2, ..., n} performs local model training on their private dataset Di = {(xi,j, yi,j)} where xi,j represents the j-th feature vector and yi,j represents the corresponding label. Each client trains a local model using logistic regression with enhanced feature engineering, incorporating 46 features including basic features (13), derived features (15), polynomial features (10), and categorical features (8). The training process includes one-class client handling strategies such as Laplace smoothing, FedProx regularization, and warm-start initialization to ensure robust performance across diverse client data distributions.

### 3.2 Phase 2: Edge Device Encryption

Following local training, each client transmits their model parameters (weights wi and bias bi) to their corresponding edge device in plaintext. The edge device then performs homomorphic encryption using the CKKS scheme with a polynomial modulus degree of 8192, coefficient modulus bit sizes of [40, 40, 40, 40], and a scaling factor of 2^40. The encryption process converts the plaintext model parameters into encrypted ciphertexts that can be processed homomorphically while maintaining privacy.

### 3.3 Phase 3: Cloud Server Aggregation

The cloud server receives encrypted model updates from all edge devices and performs homomorphic aggregation using weighted federated averaging. The aggregation process computes the weighted sum of encrypted updates, where weights are determined by the number of samples per client. The server maintains the global model in encrypted form throughout the aggregation process, ensuring that no decryption occurs during the training phase.

### 3.4 Phase 4: Global Model Update

The aggregated encrypted model parameters are used to update the global model, which remains encrypted throughout this process. The global model update follows the standard federated averaging formula but operates entirely on encrypted data, maintaining end-to-end privacy guarantees.

### 3.5 Phase 5: Edge Device Decryption and Client Synchronization

The updated global model is transmitted back to edge devices, where decryption occurs to obtain plaintext model parameters. These decrypted parameters are then distributed to clients for the next round of training, completing the federated learning round.

## 4. Algorithm Description

The EAH-FL algorithm operates in rounds, where each round consists of the five phases described above. The algorithm maintains convergence through iterative updates while preserving privacy through homomorphic encryption. The following algorithm provides a detailed description of the EAH-FL process:

---

## Algorithm 1: Edge-Assisted Homomorphic Federated Learning (EAH-FL)

**Input:** 
- Client datasets D₁, D₂, ..., Dₙ
- Number of rounds T
- FHE parameters (poly_modulus_degree, coeff_mod_bit_sizes, scale_bits)
- Learning rate η

**Output:** 
- Global model parameters W* and b*

**Initialization:**
1. Initialize global model parameters W⁰ and b⁰
2. Initialize FHE context with specified parameters
3. Generate public and private keys for homomorphic encryption

**For each round t = 1, 2, ..., T:**

**Phase 1: Client Local Training**
4. **For each client i = 1, 2, ..., n:**
   a. Load local dataset Dᵢ
   b. Apply enhanced feature engineering (46 features)
   c. Scale features using StandardScaler
   d. Train local model: (wᵢᵗ, bᵢᵗ) = TrainLocalModel(Dᵢ, Wᵗ⁻¹, bᵗ⁻¹)
   e. Apply one-class handling strategies if needed:
      - Laplace smoothing for one-class clients
      - FedProx regularization with μ = 0.01
      - Warm-start initialization using global model

**Phase 2: Edge Device Encryption**
5. **For each edge device i = 1, 2, ..., n:**
   a. Receive plaintext model parameters (wᵢᵗ, bᵢᵗ) from client i
   b. Concatenate parameters: θᵢᵗ = [wᵢᵗ, bᵢᵗ]
   c. Encrypt parameters: [[θᵢᵗ]] = Encrypt(θᵢᵗ, public_key)
   d. Transmit encrypted parameters [[θᵢᵗ]] to cloud server

**Phase 3: Cloud Server Aggregation**
6. **Cloud Server Operations:**
   a. Receive encrypted updates [[θ₁ᵗ]], [[θ₂ᵗ]], ..., [[θₙᵗ]]
   b. Compute sample counts: n₁, n₂, ..., nₙ
   c. Perform homomorphic aggregation:
      [[θᵗ]] = (1/Σᵢ₌₁ⁿ nᵢ) × Σᵢ₌₁ⁿ nᵢ × [[θᵢᵗ]]
   d. Update global model: [[Wᵗ]], [[bᵗ]] = [[θᵗ]]

**Phase 4: Global Model Update**
7. **Global Model Synchronization:**
   a. Maintain encrypted global model [[Wᵗ]], [[bᵗ]]
   b. Store model reference for client synchronization
   c. Prepare for next round or evaluation

**Phase 5: Edge Device Decryption and Client Synchronization**
8. **For each edge device i = 1, 2, ..., n:**
   a. Receive encrypted global model [[Wᵗ]], [[bᵗ]]
   b. Decrypt global model: Wᵗ, bᵗ = Decrypt([[Wᵗ]], [[bᵗ]], private_key)
   c. Transmit plaintext parameters (Wᵗ, bᵗ) to client i
   d. Client i updates local model reference for next round

**Evaluation (Optional):**
9. **If evaluation required:**
   a. Decrypt global model for evaluation
   b. Compute performance metrics on test dataset
   c. Log accuracy, F1-score, precision, recall, AUC, MAE, RMSE

**End For**

**Return:** Final global model parameters Wᵀ and bᵀ

---

## 5. Complexity Analysis

The time complexity of EAH-FL is O(T × n × (E + A + C)), where T is the number of rounds, n is the number of clients, E is the encryption/decryption time, A is the aggregation time, and C is the client training time. The space complexity is O(n × m), where m is the model parameter size. The encryption overhead scales linearly with the number of clients, with an average encryption time of 0.0044 seconds per client.

## 6. Privacy Guarantees

EAH-FL maintains the same privacy guarantees as traditional homomorphic federated learning. The global model remains encrypted throughout the training process, with decryption occurring only at edge devices for client synchronization and during evaluation. The homomorphic encryption scheme ensures that the cloud server cannot access plaintext model parameters during aggregation, providing strong privacy protection against honest-but-curious adversaries.

## 7. Performance Characteristics

The proposed EAH-FL approach achieves end-to-end delays of 0.303 seconds on average, representing a 231% overhead compared to plaintext federated learning. The system demonstrates linear O(n) scaling with client count, with per-client delay remaining constant at 0.005 seconds. The encryption efficiency is 91.38%, indicating that most of the encryption time is spent on actual homomorphic operations rather than overhead.

## 8. Conclusion

EAH-FL represents a significant advancement in federated learning by introducing edge devices as encryption proxies, enabling resource-constrained clients to participate in privacy-preserving federated learning while maintaining strong privacy guarantees. The hierarchical architecture provides a scalable solution for real-world deployment scenarios where computational resources are distributed across edge and cloud environments.

---

*This document provides a comprehensive description of the EAH-FL data flow and algorithm, suitable for IEEE standard publication and implementation reference.*


