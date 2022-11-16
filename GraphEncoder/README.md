### Research: "Applying Graph Theory to Define the Useful Information for Control System with Wireless Communication"

This work aims to reduce the information sent to the controller by explicitly identifying the dimensionality of a system and the hidden state variable from video streams. The starting point is to model the system dynamics from video representation via a neural network with bottleneck latent embeddings. Then, after training the dynamics neural network, we calculate the minimum number of independent variables needed to describe the dynamical systems. The second stage is to design a latent reconstruction neural network to identify the governing state variables with exact dimensions. Then, creating a graph that present the system dynamics and theis state variables.

Paper: https://www.overleaf.com/2724959377dnqfbbpctfcq
1. Producing data for training CartPole Example
Contents in the file [cart_pole_discreet.py](cart_pole_discreet)
2. Converting from Image to Graph
Contents in the file [IMGgraph.ipynb](IMGgraph)
3. Code for Graph Auto-Encoder and Variation Graph Auto-Encoder
Contents in the file [GAE_VGAE.ipynb](GAE_VGAE)

You also can simulate all the steps in one round using the file: [VGAE_all.ipynb](VGAE_all)

## 1. Cart-Pole
### _Some useful variants of openai-gym environment_

Cart-Pole [Custom] is a modified version of openai-gym cartpole v1 that is developed focusing on visual-based controlling.

### General modifications:
- Initial angle can be set via env.reset()
- Simulation duration can be modified via env.reset()
- Success range of the angle is extended to the range \[-45,45\] degrees
- env.render() is modified
  - Mode 'rgb_array' is extended to return grayscale and black-and-white pixel views
  - Allows to return pixel view without rendering the scene on a window
    - _Known issue: a window is rendered and set to hidden during viewer initialization to avoid generating empty pixel outputs_
- Pixel views can be compressed to low resolution views with env.down_scale()
- Reward is modified based on the deviation from the refernce state [_to be improved in future_]

### cart_pole_discreet.py:

- Added action for no force, i.e., force with zero magnitude

### Installation

Install the dependencies.

```python
pip install -U gym
pip install -U pyglet==1.5.27
```

For testing with exporting views.

```python
pip install opencv-python
```

## 3. Variation Graph Auto-Encoder
### Installation

Install the dependencies.
```python
pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
pip install -q torch-geometric
pip install -q munkres
```
