## 



## Data processing (Slide: Data \& Features)



For this part, I’ll talk about how we prepared the data and how the model works.



We started with around 220 CSV files, but after removing incomplete or corrupted jumps, we ended up with about 210 clean jumps for modeling. Each file represents one ski jump from the main flying hill in Planica and contains measurements of position in XYZ coordinares, time, length (which is measured in a straight line from the take-off point to the ski-jumpers location), height above ground, 3 speed components, a set of ski angles and a compact “wind” column that actually crams multiple wind attributes together. **(change slide to slika)** A few things to note, on how the coordinate system was placed. The take-off point is at the centre at (0, 0, 0). X coordinate runs along side the hill, Y across (so it measures width)and Z measures height. **(change slide to slika)** The angles measured are opening, and then for each ski a yaw, stalling and roll angle. And the compact wind column stores data for 12 wind sensors placed along the Planica jumping hill each measuring 5 features like wind speed, the strength of cross wind, turbulence,... For analysis we grouped them into three zones.





## Slide: Different approches and error function (45–60s)



At the start we tried a few different approaches. We used a pure SSM model and also a SSM model we tried enhancing with physics informed neural networks, but it turned out the second and more complex option performed worse, since all the flight trajectories were sent out of bounds of the Planica ski jumping hill. But to be able to have a more accurate comparison we needed a way to calculate en error. 

To compare simulated vs. real jumps, we couldn’t rely on timestamps, because jumps were measured at different time stamps.

So we interpolated both trajectories and compared the distances in each point, and if one jump ended earlier, we penalized the length mismatch by adding the tail distance. That way, models are graded both on path shape and final distance.

For final estimation, we used leave-one-out cross-validation: train on all but one, predict the held-out jump, repeat for every jump.



We then compared feeding all 12 wind sensors directly vs. feeding zone-averaged wind. Surprisingly zone-averaged had the best result. We also tried to capture wind nonlinearities by adding squared wind values. Which also improved the results. Currently the best version uses winds averaged by zone and with added square wind values. The error comes up to

train error 1.76 m and test error 1.84 m.



## Application: Neca

















## 

## 

## 

## 

## 

## \\\\Slide: What to remember (20–30s)



The pure SSM gave the best accuracy and generalization for our data.



Wind zone averaging was the winning control design: small sacrifice in train error, better test error.



Naïve nonlinear expansions didn’t help; if we add complexity, it should be targeted and physically motivated.



##### (Hand off to the next presenter / section.)

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## Data processing (Slide: Data \& Features)



For this part, I’ll walk you through how we prepared the data and how the model works.



We started with 223 CSV files—each file is one ski jump from the Gorišek brothers flying hill in Planica. Every file has measurements sampled every meter of air distance from take-off, so the number of rows depends on how far the athlete flew.



Each raw file comes with 17 columns, including the 3D position X, Y, Z, time, height above ground, a set of ski angles—opening, stalling left and right, roll left and right, yaw left and right—and three speed components: horizontal, vertical, and resulting speed. There’s also a compact “wind” column that actually crams multiple attributes together using the “|” character.



We unpacked that wind column into 72 separate features: 12 sensors times 6 wind characteristics for each—overall wind speed, tangent component along the hill, crosswind across the hill, turbulence, the “clean” tangent without turbulence, and the timestamp of the measurement. Because wind is measured less frequently than the jump telemetry, a wind value persists until a new wind reading arrives. We ultimately dropped the wind timestamp itself; it’s irregular and doesn’t add predictive power for our setup.



A couple of coordinate notes: the take-off point is (0, 0, 0). X runs down the hill, Y across, Z up. “Position” is the along-track distance; it starts negative because the start gate can move depending on wind, so the distance to the take-off point varies across jumps.



There are 12 wind sensors spread along the hill. For analysis we grouped them into three zones: sensors 1–4, 5–8, and 9–12, from early to late parts of the flight path. After removing incomplete or corrupted jumps, we ended up with about 200 clean jumps for modeling.



## Modeling approaches (Slide: Modeling Options)



We tested a few ways to model the jump trajectory.



First, we explored traditional physics-based models. They’re appealing, but with the data we had it was hard to faithfully capture all forces acting on the jumper in real time.



Second, we tried a hybrid: a state-space model—SSM—augmented by a physics-informed neural network (a PINN). The idea was that the SSM gives a solid baseline, and the PINN learns residual corrections while being nudged by physical laws—mass, gravity, wind effects—through the loss function.



In practice, that hybrid underperformed the pure SSM. The added complexity didn’t translate to better accuracy for our dataset. So we focused on the SSM family.



## Our SSM setup (Slide: State–Space Model)



Here’s how we framed the SSM.



We organized each jump’s data into three vectors:



State vector: what describes the system internally—X, Y, Z, the jumper’s velocities, and all the ski angles: opening, stalling, roll, and yaw.



Observation vector: what is directly measured—X, Y, Z, and height above ground.



Control vector: the external inputs—wind. Specifically, for each zone we aggregate the 12 sensors into zone-level averages for speed, tangent, cross, and turbulence. So instead of feeding all sensors individually, we feed zone averages as controls.



We then estimate the SSM matrices A, B, C, D using ridge regression:



A maps current state to next state,



B maps current controls to next state,



C maps current state to the current observation,



D maps current controls to the current observation.



Once A, B, C, D are learned, we roll the model forward: starting from the initial state, we iteratively predict the next state and observation using the current control inputs. That produces a full simulated trajectory for the jump under given wind conditions and initial state.



## Evaluation setup (Slide: Error Metric \& CV)



To compare simulated jumps to real ones, we need them aligned. Jumps don’t share identical timestamps or lengths, so we interpolate both trajectories onto a common grid—think of matching them at natural X positions from start to finish. Our error is the norm of the point-wise differences along the overlapping parts. If one jump ends earlier, we add the remaining distance between the tail of the shorter and longer trajectories—so models are penalized if they predict the wrong flight distance.



Because we had about 200 jumps, we used leave-one-out cross-validation: for each jump, we train on all the others and test on the held-out one. We track average error for train and test.



## Key results (Slide: What Worked Best)



Two practical findings:



First, pure SSM beat the SSM+PINN hybrid in our setting. The SSM captures the temporal dynamics cleanly without the extra learning burden of the PINN.



Second, how we feed wind matters. We compared using all 12 sensors individually versus the zone averages:



With all sensors, average point error was 1.67 m on train and 1.89 m on test.



With zone-averaged wind, it was 1.76 m on train and 1.82 m on test.



So averaging the wind by zone slightly regularizes the inputs: we give the model a cleaner, less noisy control signal that generalizes a bit better—slightly higher train error but lower test error.



We also tried to capture nonlinear wind effects by adding squared wind terms. Those didn’t materially improve accuracy, suggesting that, given our features and sampling, the zone-averaged linear control captured the bulk of the useful signal.



## Interactive app (Slide: Shiny for Python Demo)



Neca



## Takeaways for practice (Slide: Practical Notes)



A few pragmatic notes you can use:



Feature engineering for wind: zone-level aggregation was a sweet spot between detail and noise; throwing all sensors at the model did not help test performance.



Model choice: in data regimes like this, a well-regularized linear SSM can outperform more complex hybrids.



Evaluation: aligning trajectories on a common along-track grid and penalizing length mismatch gives a fairer, more informative error than simple time-aligned comparisons.



That’s the core of the data processing, modeling, app, and results. In the next segment, you’ll see… \[handoff to the next presenter’s topic].





1.78

2.01    (veter vsi)



1.76

1.84    (veter quad zone)



1.64

2.05    (veter quad vsi)



All sensors: train error 1.78 m, test error 2.01 m.

Zone averages: train error 1.76 m, test error 1.84 m.



