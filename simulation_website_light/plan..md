# Ski Jump Simulation Website Plan

## Goal

Build a separate, browser-based visualization tool for exploring how wind, speed, and jumper-angle parameters affect the state-space model's predicted ski-jump trajectory and distance.

The application should:

- run the existing state-space model faithfully;
- update quickly when sliders change;
- animate the jump smoothly;
- provide a freely rotatable 3D view of the landing hill and trajectory;
- provide multiple selectable scientific plots;
- use a dark, high-contrast visual design;
- clearly distinguish model predictions from physical or causal conclusions.

## Repository investigation

The repository contains research notebooks, exported state-space matrices, and an existing Python Shiny prototype.

### Important files

- `simulation/10current_model.ipynb`
  - Defines the current states, controls, preprocessing, model fitting, cross-validation, and simulation recurrence.
- `simulation/export_app_avg_model.py`
  - The clearest reproducible script for fitting and exporting the production matrices.
- `simulation/8sliders.ipynb`
  - Contains earlier slider and simulation experiments.
- `skijump_app/app.py`
  - Current interactive Shiny application and runtime SSM simulation.
- `skijump_app/script.js`
  - Current fixed-projection canvas animation.
- `skijump_app/hill_profile.csv`
  - Longitudinal X/Z landing-hill profile.
- `skijump_app/xImportant_AVG.csv`
  - The 139-point baseline state sequence used by the current application.
- `skijump_app/matrixA_AVG.csv`
- `skijump_app/matrixB_AVG.csv`
- `skijump_app/matrixC_AVG.csv`
- `skijump_app/matrixD_AVG.csv`
  - Production state-space matrices.

### Existing application

The existing application is a Python Shiny app. From `skijump_app/`, it can be run with:

```powershell
..\flight_env\Scripts\shiny.exe run --reload app.py
```

The upload tab uses a server-rendered Matplotlib 3D plot. The slider tab uses a JavaScript canvas with a fixed oblique projection. It visually suggests depth, but it is not a freely rotatable 3D scene.

## State-space model

The model uses the discrete recurrence:

```text
x[t + 1] = A * x[t] + B * u[t]
```

It also defines the observation equation:

```text
y[t] = C * x[t] + D * u[t]
```

The current application loads all four matrices but only uses A and B to generate the trajectory.

| Matrix | Shape | Purpose |
| --- | ---: | --- |
| A | 14 x 14 | State evolution |
| B | 14 x 20 | Effect of the controls on the next state |
| C | 4 x 14 | State-to-observation mapping |
| D | 4 x 20 | Direct control-to-observation mapping |

### State vector

The 14 states, in order, are:

1. X position [m]
2. Y position [m]
3. Z position [m]
4. X velocity [m/s]
5. Y velocity [m/s]
6. Z velocity [m/s]
7. Resulting speed [m/s]
8. Opening angle [degrees]
9. Left roll angle [degrees]
10. Right roll angle [degrees]
11. Left yaw angle [degrees]
12. Right yaw angle [degrees]
13. Left stalling angle [degrees]
14. Right stalling angle [degrees]

### Control vector

The 20 controls, in order, are:

1. Takeoff-zone wind speed
2. Middle-zone wind speed
3. Landing-zone wind speed
4. Takeoff-zone wind tangent
5. Middle-zone wind tangent
6. Landing-zone wind tangent
7. Takeoff-zone crosswind
8. Middle-zone crosswind
9. Landing-zone crosswind
10. Takeoff-zone turbulence
11. Middle-zone turbulence
12. Landing-zone turbulence
13. Resulting speed [m/s]
14. Opening angle [degrees]
15. Left roll angle [degrees]
16. Right roll angle [degrees]
17. Left yaw angle [degrees]
18. Right yaw angle [degrees]
19. Left stalling angle [degrees]
20. Right stalling angle [degrees]

### Observation vector

The four fitted observations are:

1. X position
2. Y position
3. Z position
4. Height above ground

### Production fitting setup

`simulation/export_app_avg_model.py` fits the deployed model using:

- 203 accepted flight sequences;
- one excluded problematic sequence, index 169;
- Ridge regression;
- `alpha = 10`;
- no fitted intercept;
- 14 states and 20 controls;
- all accepted transitions for the final production matrices.

Cross-validation is used for evaluation, while the production matrices are fitted on all accepted sequences.

### Control construction in the current app

The current application does not replace all controls with constant slider values.

- The first 12 sliders are offsets from the average wind conditions.
- Speed and body-angle controls preserve their 139-point time-varying baseline profiles.
- Each body slider shifts its entire profile so that its mean matches the selected slider value.
- The additional overall-angle offset is applied to the seven angle profiles, but not to speed.

This behavior should be reproduced exactly in the new simulation engine.

## Numerical findings and limitations

### Default trajectory

The current production model and default controls produce approximately:

```text
Endpoint X:       177.37 m
Endpoint Y:        -0.51 m
Endpoint Z:      -105.53 m
Displayed length: 206.39 m
```

These values should become parity fixtures for the browser implementation.

### Model evaluation

The current notebook reports approximately:

```text
Average cross-validation trajectory error: 1.63 m
Average-flight baseline error:              2.72 m
```

The error is an X-aligned mean 3D distance with a penalty for unmatched trajectory tails, rather than a simple same-timestep RMSE.

### Stability

The spectral radius of the production A matrix is approximately:

```text
1.0135
```

This is slightly above one. The website should simulate only the trained flight horizon by default. It should not imply that long extrapolations beyond the 139-point sequence are reliable.

### Interpretation

Speed and body angles appear both in the state vector and in the control vector. This makes coefficient-level causal interpretation difficult and can introduce correlated predictors.

The interface and documentation should use language such as:

> Model-predicted effect within the fitted SSM and observed parameter ranges.

It should not claim that a slider demonstrates a physically causal relationship.

Slider limits should remain tied to observed training ranges. Combining several individually valid extremes can still create an unusual or out-of-distribution scenario, so the interface should show a warning for extreme combinations.

### Local sensitivity

In a local one-slider-at-a-time test, speed had the largest immediate influence on predicted endpoint distance. Increasing mean speed by 0.1 m/s changed the displayed distance by approximately +0.67 m. Individual wind slider steps generally produced smaller changes.

This result is useful for prioritizing sensitivity plots, but it must be presented as a response of this fitted model rather than a general physics result.

### Time handling

Preprocessing uses a nominal 0.05-second step. The current Shiny application creates a synthetic time sequence beginning with `0, 4, 4.05, ...`, while its JavaScript animation ignores that sequence and advances every 45 milliseconds.

The new application should use one consistent simulation timeline:

```text
t[i] = i * 0.05 seconds
```

Playback speed can then be changed independently without changing the underlying physical timeline.

### Jump length

The current displayed length is:

```text
sqrt(X_endpoint^2 + Z_endpoint^2)
```

The first implementation should preserve this definition for numerical parity. The UI should name it clearly. A later metric can calculate distance along the hill profile or use the original `Position` definition if its exact semantics are confirmed.

### Hill geometry

`hill_profile.csv` contains 207 X/Z points, spanning approximately:

```text
X:   1.32 m to 177.03 m
Z:  -3.09 m to -105.76 m
```

It is a longitudinal profile, not a complete 3D terrain survey. The initial 3D scene can extrude this profile laterally to produce a useful landing surface with grid lines and distance markers.

That surface should be described as a model-derived 3D landing hill. A genuinely accurate Planica map would later require DEM, GIS, CAD, or surveyed terrain data.

## Recommended technology

Use a fully client-side web application:

- Vite
- React
- TypeScript
- Three.js through React Three Fiber
- Drei/OrbitControls for camera interaction
- Apache ECharts for scientific plots
- Vitest for numerical tests

The SSM is small enough to run immediately in the browser. Each step requires multiplication of a 14 x 14 state matrix and a 14 x 20 control matrix, over only 138 transitions. A server is unnecessary for ordinary interaction.

Benefits of a client-side implementation:

- immediate slider feedback;
- smooth rendering without server round trips;
- simple static deployment;
- deterministic simulations;
- easier synchronization between animation, plots, and controls;
- no Python runtime required for end users.

Python remains the authoritative environment for fitting and exporting matrices. The browser consumes versioned exported model assets.

## Proposed architecture

```text
Exported SSM assets
        |
        v
TypeScript simulation engine
        |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
3D hill and jumper    Scientific plots     Metrics/sensitivity
```

The simulation engine should be independent of React and rendering code. This makes it possible to test the numerical output without starting the UI.

## Proposed folder structure

```text
simulation_website/
|-- public/
|   |-- data/
|   |   |-- model.json
|   |   |-- baseline-flight.json
|   |   `-- hill-profile.json
|   `-- models/
|       `-- skier.glb
|-- src/
|   |-- simulation/
|   |   |-- ssm.ts
|   |   |-- controls.ts
|   |   |-- metrics.ts
|   |   `-- model-types.ts
|   |-- scene/
|   |   |-- JumpScene.tsx
|   |   |-- LandingHill.tsx
|   |   |-- Jumper.tsx
|   |   `-- WindIndicators.tsx
|   |-- plots/
|   |-- controls/
|   |-- state/
|   |-- tests/
|   |-- App.tsx
|   `-- main.tsx
|-- package.json
|-- vite.config.ts
`-- README.md
```

## Data assets

Create versioned JSON files inside `simulation_website/public/data/` rather than loading files directly from `skijump_app/` at runtime.

`model.json` should contain:

- schema version;
- model name;
- training description;
- sampling interval;
- trained horizon;
- A, B, C, and D matrices;
- ordered state names and units;
- ordered control names and units;
- slider limits and step sizes;
- baseline wind means;
- warnings and provenance.

`baseline-flight.json` should contain:

- the initial 14-state vector;
- the 139-point baseline state sequence;
- the eight time-varying speed/body profiles;
- default slider values.

`hill-profile.json` should contain:

- longitudinal X/Z points;
- surface width used for visualization;
- coordinate-system description;
- distance-marker metadata.

A later export script should generate these files from the authoritative Python data so that matrix updates are reproducible.

## Interface design

### Overall layout

Use a responsive dark dashboard:

```text
+----------------------+------------------------------------------+
| Controls             | 3D ski-jump scene                       |
|                      |                                          |
| Playback             | rotatable / zoomable / pannable         |
| Wind zones           | baseline and modified trajectories       |
| Speed and angles     | animated skier and landing point         |
| Presets/reset        |                                          |
+----------------------+------------------------------------------+
| Selectable plots and model diagnostics                          |
+-----------------------------------------------------------------+
```

On narrow screens, controls should become a drawer and plots should stack under the 3D scene.

### Left-side controls

Include:

- play/pause;
- restart;
- timeline scrubber;
- playback-speed selector;
- reset to average;
- baseline trajectory toggle;
- takeoff-zone wind controls;
- middle-zone wind controls;
- landing-zone wind controls;
- speed control;
- opening, roll, yaw, and stalling-angle controls;
- overall-angle offset;
- optional presets;
- in-range and unusual-combination warnings.

Slider labels must include units and indicate whether the value is an absolute mean or an offset from the baseline.

### 3D scene

The first 3D scene should contain:

- an extruded landing-hill surface generated from `hill_profile.csv`;
- takeoff and landing markers;
- distance lines and labels;
- X/Y/Z axes that can be toggled;
- baseline trajectory;
- modified trajectory;
- an animated jumper;
- wind-zone regions;
- optional wind arrows;
- a highlighted current point and landing point;
- perspective and orthographic camera presets.

Camera controls should support:

- left-drag/touch rotation;
- wheel/pinch zoom;
- right-drag/two-finger pan;
- damping for visually smooth movement;
- reset-camera button;
- side, top, landing, and perspective presets.

### Plot selection

Users should be able to choose which plots are visible. Initial plot groups:

1. Trajectory
   - X/Z side view
   - X/Y top view
   - Y/Z front view
   - baseline versus modified position
2. Motion
   - X/Y/Z position versus time
   - Vx/Vy/Vz versus time
   - total speed versus time
3. Jumper configuration
   - opening angle
   - roll angles
   - yaw angles
   - stalling angles
4. Environment
   - wind speed by zone
   - tangent wind by zone
   - crosswind by zone
   - turbulence by zone
5. Outcome
   - predicted distance
   - landing X/Y/Z
   - maximum Z
   - maximum absolute lateral deviation
   - clearance above the hill
6. Model diagnostics
   - A matrix heatmap
   - B matrix heatmap
   - C matrix heatmap
   - D matrix heatmap
   - one-at-a-time parameter sensitivity

Plots should share the playback cursor so that hovering or scrubbing highlights the same timestep in the 3D scene.

### Visual theme

Use a near-black/navy background with high-contrast but restrained colors.

Suggested roles:

- background: near-black navy;
- cards: slightly lighter blue-gray;
- primary text: near-white;
- secondary text: muted blue-gray;
- baseline trajectory: cyan;
- modified trajectory: orange;
- terrain and reference lines: cool green/blue;
- warning: amber;
- invalid/out-of-range state: red.

Plot grids should remain visible without dominating the data. Use a colorblind-safe series palette and never rely on color alone to distinguish baseline and modified trajectories.

## Smoothness and performance

Simulation and rendering should be separated.

1. Recalculate the full 139-point trajectory when controls change.
2. Store the result as immutable simulation output.
3. Update the Three.js line geometry only when the result changes.
4. Animate the skier independently using the render clock.
5. Interpolate between adjacent 0.05-second states.
6. Update plot cursors without rebuilding complete charts on every frame.

Performance measures:

- target 60 frames per second on typical desktop hardware;
- cap device pixel ratio on high-density displays;
- adapt rendering resolution on weaker devices;
- render on demand when playback and camera movement stop;
- reuse geometries and materials;
- avoid creating React state updates on every animation frame;
- debounce expensive plot updates if needed;
- keep shadows and post-processing modest initially.

## Implementation phases

### Phase 1: Project scaffold and scientific contract

1. Scaffold Vite, React, and TypeScript in `simulation_website/`.
2. Add linting, formatting, type checking, and Vitest.
3. Document the coordinate system, state order, control order, units, sampling interval, and distance definition.
4. Convert the current matrices and baseline inputs into versioned JSON assets.
5. Record hashes or source timestamps for data provenance.

### Phase 2: Browser SSM engine

1. Implement typed matrix-vector multiplication.
2. Implement `x[t + 1] = A*x[t] + B*u[t]`.
3. Implement `y[t] = C*x[t] + D*u[t]` for diagnostics.
4. Implement existing wind-offset and body-profile-shift behavior.
5. Implement the global angle offset.
6. Implement outcome metrics.
7. Add validation and finite-value checks.
8. Prevent simulation past the supported horizon by default.

### Phase 3: Numerical parity tests

Test the TypeScript engine against Python reference results for:

- default slider values;
- each individual slider plus one step;
- selected minimum and maximum values;
- combined but in-range scenarios;
- matrix and vector dimensions;
- exact control order;
- trajectory length and endpoint;
- absence of NaN and infinite values.

Initial default acceptance tolerance:

```text
Maximum per-state absolute difference: 1e-9 to 1e-7
```

The exact tolerance can be adjusted for JavaScript floating-point accumulation after comparing reference output.

### Phase 4: First 3D vertical slice

1. Generate a landing surface by extruding the X/Z hill profile across Y.
2. Add axes, grid lines, distance markers, and lighting.
3. Draw the baseline and modified trajectories.
4. Add a simple jumper object or marker.
5. Add OrbitControls and camera presets.
6. Add play, pause, restart, scrub, and playback speed.
7. Interpolate animation between simulation points.

### Phase 5: Controls and reactive updates

1. Recreate all current sliders with correct limits and units.
2. Group wind controls into takeoff, middle, and landing zones.
3. Group the body controls by angle type.
4. Add reset-to-average behavior.
5. Add baseline/modified comparison.
6. Recalculate and redraw immediately after slider changes.
7. Add scenario warnings and parameter presets.

### Phase 6: Plots and analysis

1. Add selectable ECharts panels.
2. Synchronize plot cursors with playback.
3. Add baseline-versus-modified plots.
4. Add trajectory projections.
5. Add motion and angle time series.
6. Add outcome metrics.
7. Add A/B/C/D heatmaps with state/control labels.
8. Add one-at-a-time sensitivity analysis.

### Phase 7: Visual refinement

1. Add an improved skier model and orientation along the flight path.
2. Add lane markings, landing zones, and clearer terrain materials.
3. Add wind arrows and zone overlays.
4. Tune lighting, fog, shadows, and depth cues.
5. Add adaptive rendering quality.
6. Complete responsive and mobile layouts.
7. Verify dark-theme contrast and keyboard accessibility.

### Phase 8: Documentation and deployment

1. Document development and production build commands.
2. Document how to regenerate browser data after retraining the SSM.
3. Document the model limitations and non-causal interpretation.
4. Add a static deployment workflow.
5. Verify the production build on desktop and mobile browsers.

## First milestone

The first usable milestone should contain:

- the exact production SSM in the browser;
- all existing sliders;
- a rotatable 3D landing hill;
- baseline and modified trajectories;
- a smoothly animated jumper marker;
- play, pause, restart, speed, and scrub controls;
- predicted jump distance and landing position;
- two selectable plots;
- numerical parity tests against Python.

This milestone establishes correctness and interaction before investing in more elaborate graphics.

## Acceptance criteria

### Numerical correctness

- Matrix dimensions and ordering match the Python model.
- Default browser trajectory matches the Python reference within tolerance.
- Every slider modifies the intended control.
- Baseline and modified trajectories are reproducible.
- Simulations remain finite across documented slider ranges.

### Visualization

- The hill and trajectory are visible and readable in the dark theme.
- The camera can rotate, pan, zoom, and reset freely.
- Playback is smooth and independent of display refresh rate.
- The skier, 3D view, timeline, metrics, and plots remain synchronized.
- Baseline and modified trajectories are visually distinguishable without relying only on color.

### Usability

- Controls are grouped and labelled with units.
- Users can reset the scenario in one action.
- Users can choose which plots are displayed.
- Unusual combinations and model limitations are visible.
- The layout remains usable on typical laptop and tablet widths.

### Performance

- Slider changes produce near-immediate trajectory updates.
- Animation targets 60 FPS on typical desktop hardware.
- Paused scenes do not continuously consume unnecessary rendering resources.
- A production build can be served as static files.

## Future extensions

- Import a real Planica DEM or surveyed terrain mesh.
- Load and compare measured jumps with simulated trajectories.
- Export scenarios and plots.
- Save/share parameter presets in the URL.
- Add uncertainty bands from cross-validation models.
- Add multiple jumpers or scenario comparison.
- Display eigenvalues and model stability diagnostics.
- Add alternative nonlinear or physics-informed models behind the same visualization interface.
- Add Web Workers if later models become computationally expensive.

## Known repository state

Before this website folder was created, Git already reported existing modified or untracked files in and around `skijump_app/`. Those files belong to the existing work and should be preserved. The new website should remain isolated inside `simulation_website/` unless an explicit data-export change is later approved.
