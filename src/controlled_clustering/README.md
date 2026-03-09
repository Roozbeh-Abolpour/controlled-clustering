# Controlled Clustering

This repository implements a **controlled online clustering framework** where a Model Predictive Control (MPC) layer regulates the dynamical behavior of an incremental K-means algorithm.

The project explores the idea of **controlling machine learning algorithms** by shaping their input data streams rather than directly modifying their internal parameters.

## Main Idea

Incremental learning algorithms such as K-means can be viewed as nonlinear dynamical systems

x_{t+1} = F(x_t, z_t)

where  
- x_t is the internal model state (cluster centers, probabilities)  
- z_t is the incoming data stream.

In this project we introduce a control input

z̃_t = z_t + u_t

where u_t is computed by a **Model Predictive Controller (MPC)**.

This allows the controller to:

- stabilize cluster evolution
- reduce oscillations
- improve convergence behavior

without altering the internal structure of the learning algorithm.

## Project Structure
src/controlled_clustering/
│
├── clustering/
│ └── kmeans.py
│
├── controller/
│ └── mpc.py
│
├── data/
│ ├── datastream.py
│ └── data.csv
│
├── identification/
│ ├── arx.py
│ └── rls.py
│
└── main.py


### Modules

**clustering**

Implements the incremental K-means algorithm.

**controller**

Model Predictive Controller used to regulate the learning dynamics.

**identification**

Online system identification tools (ARX model + Recursive Least Squares).

**data**

Streaming data interface.

---

## Installation

Create a virtual environment (optional but recommended)

Activate it

Windows

Install dependencies

---

## Running the example

Run the main experiment:


This compares

- uncontrolled clustering
- MPC-controlled clustering

and plots the evolution of clustering performance.

It must b highlighted that it is important to tune the DDMPC via its ARX oreders (mu,my), prediction horizon (N), iterations in between each sample-time (T), and input bound (dev). 
---

## Research Context

This repository accompanies research on

**control-theoretic stabilization of online learning algorithms**.

The approach combines ideas from

- machine learning
- control theory
- online system identification
- predictive control

to regulate the evolution of learning systems.

---

## License

MIT License