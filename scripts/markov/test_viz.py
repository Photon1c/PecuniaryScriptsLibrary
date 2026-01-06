#!/usr/bin/env python3
"""
Quick test of the visualization fix
"""

import matplotlib.pyplot as plt
import mplcyberpunk

# Test the visualization background fix
plt.style.use("cyberpunk")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Simple test plot
ax.plot([1, 2, 3], [1, 4, 2], 'o-')
ax.set_title("Test Visualization", fontsize=14, fontweight='bold')

plt.tight_layout()
mplcyberpunk.add_glow_effects()

# Save with light gray background
plt.savefig('test_viz_lightgray.png', dpi=150, bbox_inches='tight',
           facecolor='lightgray', edgecolor='none')

print("Test visualization saved with light gray background")
plt.close()