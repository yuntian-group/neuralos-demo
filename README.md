---
title: Neural Computer
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# Neural Computer Demo

This is a demonstration of a Neural Computer system that can generate computer screen interactions in real-time. The system uses a trained diffusion model to predict what the screen should look like based on mouse movements, clicks, and keyboard inputs.

## How to Use

1. **Wait for the model to load** - This may take a minute or two on first startup
2. **Click anywhere on the canvas** to begin interacting
3. **Move your mouse** around to see the model predict screen changes
4. **Click and drag** to simulate mouse interactions
5. **Use keyboard inputs** while focused on the canvas
6. **Use the controls** to:
   - Reset the simulation
   - Adjust sampling steps (lower = faster, higher = better quality)
   - Toggle RNN mode for even faster inference

## Settings

- **Sampling Steps**: Controls the quality vs speed tradeoff (1-50 steps)
- **Use RNN**: Enables faster inference mode using RNN output directly
- **Reset**: Clears the simulation and starts fresh

## Technical Details

This system uses a specialized diffusion model trained on computer interaction data. The model can predict realistic screen changes based on user inputs in real-time.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
