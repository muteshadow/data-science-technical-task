import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

def calculate_bayesian_probs():
  # Initial coin probabilities (probability of Head)
  p = np.array([0.12, 0.27, 0.21, 0.96])

  # Initial prior: all coins are equally likely
  posterior = np.ones(4) / 4

  # Read and preprocess user input
  sequence = entry.get().upper().replace(" ", "")
  if not sequence or not all(c in 'HT' for c in sequence):
    messagebox.showerror("Input Error", "Please enter a sequence containing only 'H' or 'T'")
    return
  
  # Clear previous results from the table
  for row in tree.get_children():
    tree.delete(row)

  # Bayesian updating for each observation
  for i, o in enumerate(sequence, 1):
    likelihood = p if o == 'H' else (1-p)
    posterior = posterior * likelihood
    posterior /= posterior.sum()

    # Probability of Head in the next flip
    next_p_h = round((posterior * p).sum(), 2)

    # Insert result into the table
    tree.insert("", "end", value=(i, o, f"{next_p_h:.2f}")) 

# GUI Setup
root = tk.Tk()
root.title("Bayesian Coin Flip Tracker")
root.geometry("400x450")

# Top input frame
frame_top = tk.Frame(root)
frame_top.pack(fill="x", padx=10, pady=10)

tk.Label(frame_top, text="Enter flip results (H - Head, T - Tail):").pack(pady=5)

entry = tk.Entry(frame_top)
entry.pack(pady=5, fill="x")
entry.insert(0, "HHHTHTHHH") # Default example

btn_calc = tk.Button(frame_top, text="Calculate", command=calculate_bayesian_probs, bg="#4CAF50", fg="white")
btn_calc.pack(pady=10, fill="x")

# Results table
columns = ("test_no", "observed", "next_p")
tree = ttk.Treeview(root, columns=columns, show="headings")

tree.heading("test_no", text="Test #")
tree.heading("observed", text="Observed result")
tree.heading("next_p", text="P(H) in next flip")

tree.column("test_no", width=80, anchor="center")
tree.column("observed", width=120, anchor="center")
tree.column("next_p", width=150, anchor="center")

tree.pack(pady=10, padx=10, fill="both", expand=True)

root.mainloop()