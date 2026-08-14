# Particle Life

![Particle Life preview](preview.gif)

A GPU compute-shader implementation of [Particle Life](https://en.wikipedia.org/wiki/Particle_Life) — a simulation of particles sorted into types, where each type has an attraction or repulsion value toward every other type. Particles move based only on those values. No shape, cluster, or behavior is programmed directly; everything you see (clusters, orbiting pairs, chains, moving swarms) emerges from those simple per-type rules running at scale.

100,000+ particles, simulated and rendered entirely on the GPU.

## Running

```
cargo run --release
```

Requires the Rust toolchain pinned in `.tool-versions` (1.95+, since nannou 0.20 depends on Bevy 0.19).

## Controls

| Input | Effect |
| --- | --- |
| **Space** | Cycle through interaction-matrix presets |
| **R** | Randomize the interaction matrix |
| **P** | Pause / resume |
| **F** | Toggle fullscreen |
| **=** / **-** | Zoom in / out |
| **Arrow keys** | Pan the camera |
| Drag the slider | Adjust simulation speed |
| Drag a matrix cell | Hand-tune attraction/repulsion between two types |

The interaction-matrix grid: row and column headers are colored by particle type, with a chevron on each showing which axis to read (row → across, column → down). Each interior cell is that row-type's attraction to that column-type — drag up to attract, down to repel.

## How it works

- **Simulation**: a uniform spatial grid rebuilt every frame via a 5-pass GPU counting sort (clear → count → prefix-sum → scatter → force), so each particle only checks its 3×3 neighboring cells instead of every other particle. The world wraps toroidally. A per-cell scan cap bounds worst-case cost even when particles cluster into dense clumps.
- **World vs. screen**: the simulation runs in a fixed-size world (larger than the window), independent of viewport size — so zoom, pan, and fullscreen only change what's visible, never the particle count or simulation itself.
- **Rendering**: particles are drawn as instanced quads reading directly from the same GPU buffer the compute shader writes, colored by type via an HSV palette. The on-screen control panel is a second, minimal render pipeline (plain colored triangles, no external UI library) that stays screen-locked regardless of camera zoom/pan.

## Stack

Rust · [nannou](https://nannou.cc/) (Bevy-based) · WebGPU (wgpu) compute shaders · WGSL
