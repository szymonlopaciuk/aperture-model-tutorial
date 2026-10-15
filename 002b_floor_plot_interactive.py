from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

# Load the lattice + the model
env = xt.load("toy_ring.json")
env.aperture.from_json("aperture_model.json")

ring = env["ring"]
ring.aperture.plot_floor_projection(aspect='equal')
plt.show()
