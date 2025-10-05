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



  \* Position over time: `X \\\[m]`, `Y \\\[m]`, `Z \\\[m]`

  \* Velocity components: `Vx \\\[m/s]`, `Vy \\\[m/s]`, `Vz \\\[m/s]`

  \* Ski angles: `SkiAngle1`, `SkiAngle2`, etc.

  \* Wind info (you've already flattened it)

\* For each variable:



  \* Plot \*\*mean\*\*, \*\*std\*\*, \*\*min\*\*, \*\*max\*\*, \*\*histogram\*\*

  \* Time-series plots for position and velocity

  \* Correlations (which you already started doing — great!)



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

\\\[x, y, z, vx, vy, vz, theta1, theta2]  # Position, velocity, ski angles

```



Inputs could be:



```text

\\\[wind speed at 12 sensors]

```



Then, model how the state evolves:



```math

x\\\_{t+1} = A x\\\_t + B u\\\_t + noise

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











W4...+Y  (negativna)

W6...+Y  (negativna)

isto 8, 9, 10, 12

---bližnji file isto +7,11

---oddaljen file  vsi pozitivni





W11...-Y  (negativna)

---bližnji file isto

---oddaljen file  vsi pozitivni







Napej plan:



* preglednejše/optimizacija
* additionally lahko nrdiš ml model, k bo v večih epochih popravu svoje predictions







💡 Bonus Suggestion: Combine Approaches

You could train a PINN model offline, and then use a Kalman Filter online to refine the states as you "simulate" or "render" the jump. This hybrid is powerful.







ssm je trained na whole data, čeprov je simulated trajectory quite ok, bi to maaybe lahko vplival na sam pinn





potencialni:



103121

103122



103411

103652





mid flights:



* 102843
* 102847







=================================

A NEW



FAULTY:

* 103114	ok
* 103121	ok
* 103149	ok
* 103139	ok
* 103138	ok
* 103119	ok
* 103117	ok
* 103151	ok





MID:

* 103140





==================================











Gre za physics-informed nevronsko mrežo, ki temelji na fiksnih state-space matrikah, ki jih med učenjem ne spreminjamo. Med forward pass-om model poleg prejšnjih napovedi upošteva tudi fizikalne lastnosti sistema, kot so masa letalca (65 kg), lastnosti vetra in gravitacijska sila. Ti fizikalni parametri so vključeni neposredno v enačbe gibanja, ki tvorijo dodaten člen v funkciji izgube (loss function).





1: 23.3

2: 16.45

















SIKDD

 	(https://aile3.ijs.si/dunja/SiKDD2024/

https://aile3.ijs.si/dunja/SiKDD2024/Papers/IS2024\_-\_SIKDD\_2024\_paper\_6.pdf)



* uvod: kaj delaš kako drugi ta problem rešujejo, zakaj je to sploh problem



* metodologija
* slike kaj se kej dogaja



kaj delam trenutno







test: 30 ish napaka

train: 22.5









2.08...test

4.05...train







cross validation

... razdeliš na 2 sample





average train error: 2.3272606977559476

average test\_error: 25.348198639757697





test: 3.47  (zone average)



















### dolžina zadnjega skoka:



* Actual: 192.5
* normal simulated: 190.8



###### vrednost **tangente vetra** v vseh conah: simulirana dolžina

* 5:  196.709
* 2:  191.445
* 0:  187.938
* -3: 182.681













### Plan za naprej:



* dodej kvadrate vetrov (znebiš linearnosti)
* še enkrat prever kodo če vse deluje,ker ne bi smel bit več, pol pa loh probaš še ostale funkcije.  







"Next each primary parameter was also used as an argument

for a number of mathematical functions in an effort to see if any

correlations aren’t linear but perhaps squared, cubed or another

elementary function. The functions used were: 𝑥^2, 𝑥^3, ln 𝑥, sin 𝑥, cos 𝑥, tan 𝑥, arcsin 𝑥, arccos 𝑥, arctan 𝑥 to try and capture any

elementary nonlinear dependence within the model." -urbanč





Logarithmic Wind Profile: In cases where wind speed needs to be estimated at different heights, the logarithmic wind profile is used: V = (Vt / k) \* ln(h / Z0). 

V is the wind speed at height h.

Vt is the friction velocity.

k is the Karman constant (approximately 0.4).

h is the target height.

Z0 is the roughness length of the surface. 







average train error: 0.10143426384736466

average test\_error: 3.328850756481124





the basics:
average train error: 1.8703356296090468
average test_error: 1.9304972738826247

more wind:
average train error: 1.7829841594248412
average test_error: 2.027487604678465


quad: 
the basics:
average train error: 1.801839759037089
average test_error: 3.268810691066911

more wind:
average train error: 1.6363794505908065
average test_error: 3.7439383977570326




LINKI:

- https://link.springer.com/article/10.1007/s10409-021-01148-1
- učbenik za NUM (Razširjen uvod v numerične metode
Bor Plestenjak)
- CERN ski jumping physics (https://cds.cern.ch/record/1009275/files/p269.pdf)
- https://www.google.si/books/edition/Ski_Jumping/G2pPEQAAQBAJ?hl=en&gbpv=0
- SSM yt video (https://youtu.be/g1AqUhP00Do?si=NlCog3HoVHhqQLEh)
- NN statquest video (https://youtu.be/CqOfi41LfDw?si=zhH_LABzN-28G1j0)





ergmann. 2025. What is a state space model? Accessed: 2025-09-24.
https://www.ibm.com/think/topics/state-space-model


STUFF ZA NAPREJ

-kote v U in povezat s sliders
-distribucije stvari as in analiza podatkov
-dodajanje več funkcij v model





















