import random
import matplotlib.pyplot as plt

# set custom font for vizual
plt.rcParams["font.family"] = "Comic Sans MS"

# generate 26 random integer values
values = [random.randint(0, 26) for _ in range(26)]

# values = [
#     2, 7, 25, 14, 14, 21, 20, 6, 13, 3,
#     23, 14, 1, 8, 1, 8, 16, 21, 1, 17,
#     7, 5, 21, 17, 18, 10
# ]

# finds the deepest lake/lakes in the given height
def deepest_lake_depth(heights):
  n = len(heights)
  max_depth = 0
  lakes = []

  # iterate over all possible pairs of boundaries
  for i in range(n):
    for j in range(i + 2, n):
      water_level = min(heights[i], heights[j])
      inside = heights[i+1:j]

      # skip if there is no space between
      if not inside: continue

      bottom = min(inside)

      # check if a valid lake is formed
      if all(h < water_level for h in inside):
        depth = water_level - bottom

        # new max depth found
        if depth > max_depth:
          max_depth = depth 
          lakes = [(i, j, depth, water_level, bottom)]

        # another lake with the same max depth
        elif depth == max_depth:

          # avoid nested duplicate
          if not any(i >= l[0] and j <= l[1] for l in lakes):
            lakes.append((i, j, depth, water_level, bottom)) 
  
  return max_depth, lakes

# run lake algorithm
max_depth, found_lakes = deepest_lake_depth(values)

# plot setup
plt.figure(figsize=(12, 6))
plt.plot(values, color="#00ebbb", linewidth=1.5, label="Mountains")

# color and style for lakes
colors = ["#32a1fe", "#00d9e9", "#8577cb"]
styles = [":", "--", '-.']

# vizual each detected lake
for idx, (i, j, depth, water_level, bottom) in enumerate(found_lakes):
  color = colors[idx % len(colors)]
  style = styles[idx % len(styles)]

  # lake boundaries
  plt.plot(range(i, j+1), values[i:j+1], color=color, linewidth=3, label=f"Lake {idx+1} (Depth: {depth})")
  
  # water level
  plt.hlines(y=water_level, xmin=i, xmax=j, colors=color, linestyles="--", alpha=0.8)
  
  # vertical water filling
  for k in range(i+1, j):
    plt.vlines(x=k, ymin=values[k], ymax=water_level, color=color, linestyles=style, linewidth=0.8, alpha=0.5)

# final plot styling
plt.title(f"Deepest lake depth = {depth}")
plt.xlabel("Index")
plt.ylabel("Height")
plt.legend()
plt.grid(axis='y', alpha=0.2)
plt.show()