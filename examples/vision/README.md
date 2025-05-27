# Vision Hallucination Evaluation with CheckEmbed

This example demonstrates an end-to-end experiment for assessing hallucinations in image generation using CheckEmbed and Stable Diffusion 3.5.

## Structure

```
imgs/                    
    └── counting_items/    # Images generated for this experiment
main.py                  # Script to generate images, embeddings, and run CheckEmbed
plot.py                  # Script to create the plot
requirements.txt         
README.md                # This document
```

## Usage

### 1. Generate Images & Embeddings, Run CheckEmbed

```bash
python main.py --start_idx 0 --end_idx 8
```
Varying the `--start_idx` and `--end_idx` parameters allows you to process in parallel. However, run the CheckEmbed step sequentially.

The prompts are hardcoded in `main.py` and are designed to generate images with a specific number of items. 

* Outputs:
  * `imgs/counting_items/`: Generated PNG images.
  * `clip_embeddings/counting_items/`: JSON files of CLIP embeddings.
  * `checkembed_outputs/counting_items/`: CheckEmbed result JSONs.

### 2. Plot Results

```bash
python plot.py
```

* Reads `checkembed_outputs/counting_items/` and manual counts from `imgs/`.
* Produces a violin plot (CheckEmbed score vs. number of items) saved to `plots/`.

## Configuration

* Paths in `main.py` and `plot.py` (e.g., `path/to/...`) should be updated to your local directories before running.
* Modify `input_prompts` in `main.py` to extend or change prompt sets.

## Results

* Plots are saved as PDF in `plots/` directory.
* Compare CheckEmbed scores against manual correctness counts to evaluate precision.

## License & Author

* BSD-style license (see `LICENSE`).
* Main author: Eric Schreiber, ETH Zurich.
