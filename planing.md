Use this dataset to fit or simulate dynamics with:



1. State-space models
   
2. Physics-informed machine learning
   
3. Reinforcement learning, etc.







# PLAN





Thanks for sharing the full context — this is a well-scoped and interesting project. Since the goal is to build a \*\*dynamic model (simulator) of ski jumping flight\*\*, you're already making strong progress by preprocessing the data.



Let’s break it down into \*\*clear, actionable next steps\*\*, tailored to your task:



---



\### ✅ \*\*Immediate Next Step: Statistical Exploration\*\*



Before jumping into modeling, it's critical to understand the data deeply.



\#### 1. \*\*Explore All Variables Statistically\*\*



\* Loop through all CSVs and extract:



&nbsp; \* Position over time: `X \[m]`, `Y \[m]`, `Z \[m]`

&nbsp; \* Velocity components: `Vx \[m/s]`, `Vy \[m/s]`, `Vz \[m/s]`

&nbsp; \* Ski angles: `SkiAngle1`, `SkiAngle2`, etc.

&nbsp; \* Wind info (you've already flattened it)

\* For each variable:



&nbsp; \* Plot \*\*mean\*\*, \*\*std\*\*, \*\*min\*\*, \*\*max\*\*, \*\*histogram\*\*

&nbsp; \* Time-series plots for position and velocity

&nbsp; \* Correlations (which you already started doing — great!)



\#### 2. \*\*Normalize coordinate systems\*\*



Make sure you're working in a consistent frame:



\* Does `Y` always represent height?

\* Are the angles relative to horizontal or another axis?

\* Should time always start at `t = 0`?



---



\### 🚀 \*\*Phase 2: Trajectory Analysis\*\*



Once you understand the data...



\#### 3. \*\*Compute Derived Quantities\*\*



\* Total velocity: `v = sqrt(vx² + vy² + vz²)`

\* Angle of attack (between velocity vector and ski orientation)

\* Flight path angle (from position change)



\#### 4. \*\*Cluster or Group Flights\*\*



\* Group flights by jump length or other criteria

\* Try PCA to see dominant motion patterns

\* Align them in time to compare similar flight phases (e.g. takeoff, apex, landing)



---



\### 📈 \*\*Phase 3: Model Building\*\*



You mentioned \*\*State Space Models\*\* — these are great for continuous dynamic systems. In this context:



\#### 5. \*\*Define State and Input\*\*



A state might be:



```text

\[x, y, z, vx, vy, vz, theta1, theta2]  # Position, velocity, ski angles

```



Inputs could be:



```text

\[wind speed at 12 sensors]

```



Then, model how the state evolves:



```math

x\_{t+1} = A x\_t + B u\_t + noise

```



You can start with:



\* Linear approximation

\* Kalman filtering

\* Use tools like Python’s `statsmodels`, `filterpy`, or even `PyTorch` if you go nonlinear later



---



\### 📊 \*\*Phase 4: Visualization \& Validation\*\*



\* Simulate new trajectories and compare to real ones

\* Plot simulated vs real in 3D

\* Animate flight paths

\* Error metrics: RMSE between simulated and actual trajectories



---



\### 🧠 Summary: Suggested Roadmap



| Phase            | Goal                                         | Tools                       |

| ---------------- | -------------------------------------------- | --------------------------- |

| ✅ Data Analysis  | Understand all variables, trends, and issues | pandas, matplotlib, seaborn |

| ✈️ Trajectory    | Compare real flights, extract flight phases  | numpy, scipy, stats         |

| 🔁 Modeling      | Build dynamic model (e.g., state space)      | statsmodels, filterpy       |

| 📽 Visualization | Simulate, validate, and animate              | matplotlib (3D), plotly     |



---



Let me know if you'd like example code for \*\*time alignment\*\*, \*\*state space modeling\*\*, or \*\*flight animation\*\* — happy to help you get started!













103138...C2 je faulty file

103606

103914

102840











Phase 1: Clean and Explore

Finish your cleaned dataset export (you're nearly done)



Add basic visualizations: plot flight curves, angles, velocity vectors over time



Phase 2: Try lightweight SSM

Downsample to fixed-length vectors (e.g., 50 time slots per flight)



Build baseline SSM model (simple linear matrices A, B, C)



See what interpretability you can extract



Phase 3: Move to PIML

If SSM performs poorly due to heavy interpolation or you want richer modeling:



Use PIML (or Neural ODEs) on raw flight traces



Optionally incorporate known physics as loss terms (e.g., conserving energy or aerodynamic constraints)













PLAN:   

* understand the code   ish
* preveriš korelacijo pri vseh ločenih y
* simulated loop čez vse + a je smiselno xz plottat, xyz? 







You’re correct: classic SSM doesn’t update its parameters when new data comes in (unless you use adaptive or recursive estimation methods like Kalman filters or adaptive control).



Or how modern machine learning uses state-space models (like S4 or HiPPO) in sequence modeling, which actually combines both worlds?





under question:  (ok y)

* 103139	ok
* 103149	ok
* 103503	ok
* 103604	ok
* 103610	ok
* 103706	ok
* 103713	ok
* 103754	ok





under question: (ni ok y)

* 103117	ok
* 103150	ok
* 103407	ok
* 103418	ok
* 103420	ok
* 103917	ok



* ish: 102833	ok
* ish: 102852	ok
* ish: 103056	ok
* ish: 103921	ok







Napej plan:



* sporoči jakobu

* na neki točki daš nekam da bo skupno dostopno ig (maybe git)

* additionally lahko nrdiš ml model, k bo v večih epochih popravu svoje predictions







W4...+Y  (negativna)

W6...+Y  (negativna)

isto 8, 9, 10, 12

---bližnji file isto +7,11

---oddaljen file  vsi pozitivni





W11...-Y  (negativna)

---bližnji file isto

---oddaljen file  vsi pozitivni













* find the not simulated 3.5 or 2.3 graph... 102834



















