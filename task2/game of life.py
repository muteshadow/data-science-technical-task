import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import tkinter as tk
from tkinter import simpledialog

# Generate initial grid (26x26) with random 0/1 values
grid = np.random.randint(0, 2, (26, 26))

# Count alive neighbors for a single cell
def count_neighbors(grid, row, col):
  neighbors = 0

  for i in range(row - 1, row + 2):
    for j in range(col - 1, col + 2):

      # Check grid boundaries
      if i < 0 or j < 0 or i >= grid.shape[0] or j >= grid.shape[1]:
        continue

      # Skip the cell itself
      if i == row and j == col:
        continue

      neighbors += grid[i][j]

  return neighbors

# Determine next state of a single cell
def next_cell_state(grid, row, col):
  alive = grid[row][col]
  neighbors = count_neighbors(grid, row, col)

  if alive == 1:
    if neighbors == 2 or neighbors == 3:
      return 1
    else:
      return 0
  else:
    if neighbors == 3:
      return 1
    else:
      return 0

# Perform one full evolution step
def next_step(grid):
  new_grid = np.zeros_like(grid)

  for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
      new_grid[i][j] = next_cell_state(grid, i, j)

  return new_grid

# Ask user for number of steps
root = tk.Tk()
root.withdraw() # Hide main Tk window

user_input = simpledialog.askstring("Input", "How many evolution steps to simulate? (max 50):")

try:
  user_steps = int(user_input) if user_input else 20
except ValueError:
  user_steps = 20

if user_steps > 50:
  print("Too many steps requested. Limiting to 20 steps")
  user_steps = 20

# Cell age grid (used for coloring)
age_grid = np.zeros_like(grid, dtype=float)

# Custom colormap (blue → pink)
colors = ["#dcf2ff","#32a1fe","#436b9b","#f4709c","#b83969"]
custom_cmap = LinearSegmentedColormap.from_list("blue_pink_life", colors)

# Visualization loop
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

for i in range(user_steps):
  # Next generation
  grid = next_step(grid) 

  # Update age grid
  age_grid[grid == 1] += 1
  age_grid[grid == 0] = 0

  ax.clear()

  img = ax.imshow(age_grid, cmap=custom_cmap, interpolation='nearest')

  ax.set_title(f"Generation #{i + 1}")
  ax.axis("off")

  # Footnote
  footnote = "LEGEND:\nLight blue — newborn cell\nPink — living more than 5 generations"
  ax.text(0, 28, footnote, fontsize=10, color="gray", style="italic")

  # Speed
  plt.pause(0.15)

# Finalize visualization
plt.ioff()
plt.show()