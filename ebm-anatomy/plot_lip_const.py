import sys
import os

import matplotlib.pyplot as plt
import pandas as pd


csv = sys.argv[1]
df = pd.read_csv(csv)
y = df["Lip_const"] if "Lip_const" in df.columns else df.iloc[:, -1]

plt.plot(df["iteration"] if "iteration" in df.columns else range(len(df)), y)
plt.xlabel("iteration")
plt.ylabel("Lip_const")
plt.tight_layout()

# Sauvegarde dans le même dossier que le CSV
csv_dir = os.path.dirname(os.path.abspath(csv))
csv_name = os.path.splitext(os.path.basename(csv))[0]
output_path = os.path.join(csv_dir, f"{csv_name}_plot.png")

plt.savefig(output_path, dpi=150)
print(f"Plot sauvegardé : {output_path}")
plt.show()