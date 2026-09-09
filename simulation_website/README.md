# Ski Jump SSM Lab

Interactive, browser-based visualization of the production ski-jump state-space model in this repository.

The application runs the full 14-state, 20-control model locally in the browser, renders the fitted landing-hill profile in WebGL, and synchronizes a rotatable 3D flight animation with plots, matrices, controls, and outcome metrics.

## Features

- Exact `x[t+1] = A·x[t] + B·u[t]` recurrence used by `skijump_app`.
- `y[t] = C·x[t] + D·u[t]` observation diagnostics.
- All 12 wind offsets, eight speed/body profile means, and the overall angle offset.
- Baseline-versus-modified trajectory comparison.
- Play, pause, restart, timeline scrubber, and playback speeds.
- Rotatable, zoomable, pannable Three.js hill with camera presets.
- Animated jumper, wind-zone indicators, distance markers, and landing point.
- Selectable trajectory, state, control, observation, sensitivity, and matrix plots.
- Responsive dark interface with an off-canvas control drawer on small screens.
- No simulation server: after the static assets load, calculations remain in the browser.

## Requirements

- Node.js 20.19 or newer
- npm
- Python environment at `../flight_env` only when regenerating model assets

## Development

From `simulation_website/`:

```powershell
npm install
npm run dev
```

The Vite development server prints the local URL, normally `http://localhost:5173`.

## Verification

```powershell
npm test
npm run lint
npm run build
```

The parity suite checks:

- A/B/C/D dimensions and control ordering;
- the default endpoint and displayed distance against Python-generated fixtures;
- the first control vector;
- finite simulation output at every individual slider minimum and maximum;
- 0.05-second playback interpolation.

The reference default is approximately:

```text
endpoint = [177.367318956, -0.508112297, -105.529263855]
displayed distance = 206.386994172 m
```

## Production build

```powershell
npm run build
npm run preview
```

The deployable static application is written to `dist/`. Vite uses a relative base path, so the build can be hosted at a site root or a nested static path.

## Regenerating browser model assets

The application never reads from `skijump_app` at runtime. Versioned JSON assets are generated from its audited CSV files:

```powershell
npm run export:model
```

This runs `scripts/export_model.py` and regenerates:

- `public/data/model.json`
- `public/data/baseline-flight.json`
- `public/data/hill-profile.json`

The model asset records SHA-256 hashes for every source CSV. Regenerate the files whenever the production matrices, baseline state sequence, or hill profile changes, then rerun the parity suite.

The exporter converts already-fitted matrices; it does not retrain them. Production fitting remains in `../simulation/export_app_avg_model.py`.

## Scientific contract

### Coordinate system

- X: forward along the jump, metres.
- Y: lateral movement across the hill, metres.
- Z: vertical elevation relative to takeoff, metres.
- Three.js maps model `(X, Y, Z)` to scene `(X, Z, Y)` because Three.js uses its second coordinate as vertical.

### State order

```text
X, Y, Z, Vx, Vy, Vz, speed, opening,
roll left, roll right, yaw left, yaw right,
stall left, stall right
```

### Control order

```text
wind speed:      takeoff, middle, landing
tangent wind:   takeoff, middle, landing
crosswind:      takeoff, middle, landing
turbulence:     takeoff, middle, landing
speed, opening, roll L/R, yaw L/R, stall L/R
```

Wind sliders are offsets from the fitted average wind values. Speed and angle sliders shift the means of their complete 139-point baseline profiles. The global angle offset affects the seven angle controls, not speed.

### Time and distance

- Sampling interval: 0.05 seconds.
- Supported horizon: 139 points / 138 transitions / 6.90 seconds.
- Displayed distance: `sqrt(X_endpoint² + Z_endpoint²)`, retained for parity with the existing app.

## Interpretation and limitations

- Results are predictions of a fitted linear SSM, not causal physical conclusions.
- Speed and body angles occur in both state and control vectors, which limits coefficient-level causal interpretation.
- The A matrix spectral radius is approximately 1.0135. The UI therefore limits simulation to the trained horizon.
- Slider limits come from observed training ranges, but combining multiple extremes can still produce an unusual scenario.
- The landing surface is a longitudinal X/Z profile extruded across Y. It is not a surveyed 3D Planica terrain mesh.
- The `C/D` height observation is available as a model diagnostic. Terrain clearance is separately calculated against the displayed hill profile.

See [plan..md](./plan..md) for the investigation, rationale, phased plan, and future extensions.

