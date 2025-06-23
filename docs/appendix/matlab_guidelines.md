## 1. General Rules

* **Language:** All figure labels, annotations, legends, and code comments must be in English. No Chinese characters are permitted.
* **File Format:** Export figures in **PDF** format using `plt.savefig('filename.pdf', format='pdf')` to preserve vector quality and embed fonts.

---

## 2. Figure Size & Resolution

| Scenario                        | Size (width × height, inches) | DPI     |
| ------------------------------- | ----------------------------- | ------- |
| One‑column figure               | 3.3 × 2.5                     | 300–600 |
| Two‑column figure               | 6.6 × 5.0                     | 300–600 |
| 2×2 grid of subplots            | 7.0 × 7.0                     | 300–600 |
| 3 rows of subplots (single‑col) | 3.3 × 9.0                     | 300–600 |

> **Why these sizes?**
>
> * Typical journal column width ≈85 mm (one‑column) or ≈170 mm (two‑column). 1 inch ≈ 25.4 mm.
> * DPI ≥300 ensures print clarity; choose 600 DPI for high‑detail heatmaps or density plots.

**Example:**

```python
plt.figure(figsize=(3.3, 2.5), dpi=300)
```

---

## 3. Fonts, Sizes & Color Palette

* **Global font settings:**

  ```python
  plt.rcParams.update({
      'font.family': 'serif',
      'font.serif': ['Arial'],
      'font.size': 8,
      'text.color': 'k',
      'axes.labelcolor': 'k',
      'xtick.color': 'k',
      'ytick.color': 'k',
      'legend.fontsize': 8,
  })
  ```
* **Custom color palette:**

  ```python
  colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf']
  ```

| Element                   | Font Size (pt) | Color       | Notes                                         |
| ------------------------- | -------------- | ----------- | --------------------------------------------- |
| Axis labels               | 8              | `'#000000'` | e.g. `ax.set_xlabel('Time (s)', fontsize=8)` |
| Tick labels               | 8              | `'#000000'` | `ax.tick_params(labelsize=8)`                 |
| Legend text               | 8              | `'#000000'` | `ax.legend(frameon=False)`                    |
| Annotation text           | 8              | `'#000000'` | `ax.text(x, y, 'note', fontsize=7)`           |
| Subplot subtitles (A, B…) | 8              | `'#000000'` | Placed above each subplot, no bold            |

---

## 4. Lines, Markers & Legend

* **Line width:** `linewidth=1.0`
* **Marker size:** `markersize=6`
* **Legend frame:** prefer no frame or a thin edge:

  ```python
  ax.legend(frameon=False)
  # or
  ax.legend(edgecolor='k', linewidth=0.5)
  ```

**Example:**

```python
ax.plot(x, y, linewidth=1.2, marker='o', markersize=5,
        color=colors[0], label='Sample')
ax.legend(frameon=False)
```

---

## 5. Single‑Plot Complete Example

```python
import matplotlib.pyplot as plt
import numpy as np

# Global settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Arial'],
    'font.size': 8,
    'text.color': 'k',
    'axes.labelcolor': 'k',
    'xtick.color': 'k',
    'ytick.color': 'k',
    'legend.fontsize': 8,
})

primary_colors = ['#699ECA', '#FF8C00', '#F898CB', '#4DAF4A']
backup_colors  = ['#D65190', '#731A73', '#FFCB5B', '#E87B1E', '#0076B9', '#3D505A', '#0098B2']

# Create figure and axis
fig, ax = plt.subplots(figsize=(3.3, 2.5), dpi=300)

# Sample data
t = np.linspace(0, 10, 100)
S = np.sin(t)

# Plot
ax.plot(t, S, linewidth=1.2, color=colors[0], label='Sine wave')
ax.set_xlabel('Time (s)', fontsize=8)
ax.set_ylabel('Amplitude', fontsize=8)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.legend(frameon=False)

# Save as PDF
plt.tight_layout()
plt.savefig('figure1.pdf', format='pdf')
```

*Note:* No in‑figure title is set; all descriptive text belongs in the manuscript caption.

---

## 6. Multi‑Panel Subplots (2×2) Setup

```python
fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.0), dpi=300,
                        sharex=True, sharey=True)
letters = ['A', 'B', 'C', 'D']
for i, ax in enumerate(axs.flat):
    ax.plot(t, np.sin(t + i), linewidth=1.2,
            color=colors[i], label=f'Series {i+1}')
    ax.set_xlim(0, 10)
    if i in [2, 3]:
        ax.set_xlabel('Time (s)', fontsize=8)
    if i in [0, 2]:
        ax.set_ylabel('Amplitude', fontsize=8)
    # Panel subtitle
    ax.text(0.5, 1.02, letters[i],
            transform=ax.transAxes,
            fontsize=8, fontweight='normal', ha='center', va='top', color='k')
    ax.tick_params(labelsize=8)
# Adjust spacing
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.tight_layout()
# Save as PDF
plt.savefig('figure2_subplots.pdf', format='pdf')
```

---

## 7. Output & Submission Tips

* Always embed fonts in the PDF: `plt.savefig('...pdf', format='pdf')` handles this by default with matplotlib.
* The submitted figure files should contain only graphical elements (axis labels, panel letters, scale bars), **no embedded manuscript captions**.
* Provide full detailed captions in the manuscript text where figures are referenced.
