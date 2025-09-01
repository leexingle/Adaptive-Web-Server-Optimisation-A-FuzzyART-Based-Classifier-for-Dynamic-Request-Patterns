# Adaptive-Web-Server-Optimisation-A-FuzzyART-Based-Classifier-for-Dynamic-Request-Patterns

## Overview
This project implements a real-time model for **adaptive web server optimisation**. It classifies incoming web request patterns using a **FuzzyART neural network** and applies a **reinforcement learning agent (UCB1)** to dynamically select the optimal server configuration.

The primary goal is to improve **web server response time**, since different web server configurations handle different traffic patterns with varying efficiency. This work extends previous research on adaptive web server, using a distributed web server and dataset provided by supervisor.

---

## Overall Architecture
- **Data processing** - Convert low level features to high level features
- **FuzzyART classifier** - Classify web request pattern  
- **Reinforcement learning (UCB1)** Select optimal web server configuration 

---

## Code Implementation
- `learning/ARTv2.py` → Main implementation of the FuzzyART classifier and reinforcement learning integration.  

---

## Input & Output
**Input:**  
- Request features:  
  - `contentLength` (numeric size of file)  
  - `type` (MIME type of requested file)  

**Output:**  
- Predicted request class (e.g., *Class A*)  

**UCB1 Integration**
- Each class then assigned to a UCB1 agent, choose optimal server cofiguration (e.g., *Server Type 2*)  

---

## Results
- Achieved **38.69%** improvement in performance compared to without classifier.  
- Demonstrates the effectiveness of combining **FuzzyART classifier** with **reinforcement learning** for adaptive optimisation.  

---
