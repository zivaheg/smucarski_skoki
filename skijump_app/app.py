from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as pytime
from scipy.interpolate import interp1d
import io
import os
import random
import math

#nalozim datoteko s podatki 
#datoteka = pd.read_csv("cleaned/999_999_Jumper_Anon_ANO_1_20240409-102839_C_OfficialResults_cleaned.csv")
#102839
#priprava podatkov za trajektorijo
#x = datoteka["X [m]"].to_numpy()
#y = datoteka["Y [m]"].to_numpy()
#z = datoteka["Z [m]"].to_numpy()
#t = datoteka["Time"].to_numpy()
#last_position = datoteka["Position"].iloc[-1] #končna pozicija, ki je ubistvu dolzina skoka

# create interpolation functions, da bo zadeva tekla bolj smoothly
#x_interp = interp1d(t, x, fill_value="extrapolate")
#y_interp = interp1d(t, y, fill_value="extrapolate")
#z_interp = interp1d(t, z, fill_value="extrapolate")

#razlike = np.diff(t) * 1000 #tabela casovnih intervalov med dvema zaporednima tockama
                            #sekunde -> milisekunde
#razlike = np.append(razlike, 0) #da na koncu ni zamika

#default_df = pd.read_csv("default_jump.csv")

# folder with jumps
JUMP_FOLDER = "cleaned"

def random_jump_file():
    files = [f for f in os.listdir(JUMP_FOLDER) if f.endswith(".csv")]
    return os.path.join(JUMP_FOLDER, random.choice(files)) if files else None

def make_interpolators(df):
    t = df["Time"].to_numpy()
    return {
        'x': interp1d(t, df["X [m]"].to_numpy(), bounds_error=False, fill_value="extrapolate"),
        'y': interp1d(t, df["Y [m]"].to_numpy(), bounds_error=False, fill_value="extrapolate"),
        'z': interp1d(t, df["Z [m]"].to_numpy(), bounds_error=False, fill_value="extrapolate"),
    }, t

initial_file = random_jump_file()
if initial_file:
    default_df = pd.read_csv(initial_file)
else:
    default_df = pd.DataFrame(columns=["X [m]", "Y [m]", "Z [m]", "Time", "Position"])


# Example: assuming CSVs have no headers, one matrix per CSV
A = pd.read_csv("matrixA_AVG.csv", header=None).to_numpy()
B = pd.read_csv("matrixB_AVG.csv", header=None).to_numpy()
C = pd.read_csv("matrixC_AVG.csv", header=None).to_numpy()
D = pd.read_csv("matrixD_AVG.csv", header=None).to_numpy()


models = [A, B, C, D]

X_important = pd.read_csv("xImportant_AVG.csv", header=None).to_numpy()

# The model was trained with controls in this exact order.  The average wind
# values come from average_flight_normalized.csv, while the time-varying speed
# and angle profiles are already present in columns 6:14 of X_important.
BASELINE_WIND_MEANS = np.array([
    1.0747, 1.3615, 1.0120,
    0.5363, 0.5560, 0.1634,
    0.5514, 0.7760, 0.4583,
    0.1245, 0.1821, 0.1360,
])
BASELINE_BODY_PROFILES = X_important[:, 6:14].copy()
BASELINE_BODY_MEANS = BASELINE_BODY_PROFILES.mean(axis=0)
DEFAULT_BODY_SLIDER_VALUES = np.round(BASELINE_BODY_MEANS, 1)
DEFAULT_SLIDER_VALUES = np.concatenate([
    np.zeros(12), DEFAULT_BODY_SLIDER_VALUES, np.zeros(1)
])

# Slider limits use the 1st and 99th percentiles of the per-jump control
# means in the 203-sequence training set. Wind limits are expressed as offsets
# from BASELINE_WIND_MEANS; body limits are the selected profile means.
WIND_OFFSET_LIMITS = [
    (-0.6, 0.9), (-0.8, 1.0), (-0.6, 1.2),
    (-1.0, 1.4), (-1.6, 1.5), (-1.5, 1.7),
    (-0.4, 0.5), (-0.6, 1.1), (-0.3, 0.5),
    (-0.07, 0.13), (-0.10, 0.19), (-0.08, 0.12),
]
BODY_MEAN_LIMITS = [
    (28.8, 33.4),    # speed
    (14.9, 39.1),    # opening
    (13.7, 34.9),    # roll left
    (-36.0, -15.8),  # roll right
    (3.7, 21.1),     # yaw left
    (-21.6, -5.3),   # yaw right
    (-7.7, 5.3),     # stalling left
    (-6.9, 7.1),     # stalling right
]

#definiram UI
app_ui = ui.page_fluid(
    ui.tags.style("""
        .slider-simulation-layout {
            display: grid;
            grid-template-columns: minmax(280px, 0.9fr) minmax(500px, 2.2fr) minmax(290px, 1fr);
            gap: 12px;
            align-items: start;
            width: 100%;
        }

        .zone-stack {
            display: grid;
            gap: 8px;
        }

        .zone-stack .card,
        .angle-column .card {
            margin-bottom: 0;
        }

        .zone-stack .card-header,
        .angle-column .card-header {
            padding: 0.45rem 0.7rem;
        }

        .zone-controls {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            column-gap: 10px;
            padding: 0.45rem 0.65rem 0.2rem;
        }

        .zone-controls .shiny-input-container,
        .angle-column .shiny-input-container {
            width: 100%;
            margin-bottom: 0.35rem;
        }

        .zone-controls label,
        .angle-column label {
            font-size: 0.82rem;
            line-height: 1.1;
        }

        .simulation-column {
            min-width: 0;
            text-align: center;
        }

        .simulation-actions {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 18px;
            margin-bottom: 4px;
        }

        .simulation-plot {
            width: 100%;
            min-width: 0;
        }

        .angle-column .card-body {
            padding: 0.6rem 0.75rem 0.35rem;
        }

        @media (max-width: 1200px) {
            .slider-simulation-layout {
                grid-template-columns: minmax(260px, 0.9fr) minmax(460px, 2fr);
            }

            .angle-column {
                grid-column: 1 / -1;
            }

            .angle-column .card-body {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                column-gap: 14px;
            }
        }

        @media (max-width: 780px) {
            .slider-simulation-layout {
                grid-template-columns: 1fr;
            }

            .simulation-column,
            .angle-column {
                grid-column: 1;
            }

            .angle-column .card-body {
                grid-template-columns: 1fr;
            }
        }
    """),
    ui.navset_pill_list(
        ui.nav_panel(
            "upload CSV",
            ui.h2("Ski jump animation", style="text-align: center;"), #naslov
            ui.input_file("file1", "choose CSV file", accept=[".csv"], multiple=False),
            ui.help_text("if no file is uploaded, the default jump is shown"),
            ui.output_text_verbatim("file_info"),
            ui.div(
                ui.input_action_button("start_button", "start jump"), #gumb, ki zacne animacijo
                ui.input_action_button("random_button", "new random jump"), #gumb, ki zacne animacijo
                style="text-align: center; margin-top: 30px;" #30 pixlov prostora
            ),
            ui.div(
                ui.output_text("jump_length_upload"),
                style="text-align: center; margin-top: 20px;"
            ),
            ui.div(
                ui.output_plot("traj_plot_upload", height="640px", width="960px"),
                #najprej je bilo 800 in 960, zdej je 80% tega
                style="display: flex; justify-content: center;"
            )
        ),
        ui.nav_panel(
            "use sliders",
            ui.h2("Ski jump animation", style="text-align: center;"), #naslov
            ui.help_text(
                "Slider limits cover the central 98% of control means observed "
                "in the training jumps. Combining several limits may still "
                "represent an unusual jump."
            ),
            ui.div(
                ui.div(
                    ui.card(
                        ui.card_header("first zone"),
                        ui.div(
                            ui.input_slider("average_wind_speed1", "wind speed offset", min=WIND_OFFSET_LIMITS[0][0], max=WIND_OFFSET_LIMITS[0][1], value=0, step=0.1),
                            ui.input_slider("average_wind_tangent1", "wind tangent offset", min=WIND_OFFSET_LIMITS[3][0], max=WIND_OFFSET_LIMITS[3][1], value=0, step=0.1),
                            ui.input_slider("average_wind_cross1", "cross wind offset", min=WIND_OFFSET_LIMITS[6][0], max=WIND_OFFSET_LIMITS[6][1], value=0, step=0.1),
                            ui.input_slider("average_wind_turbulence1", "turbulence offset", min=WIND_OFFSET_LIMITS[9][0], max=WIND_OFFSET_LIMITS[9][1], value=0, step=0.01),
                            class_="zone-controls"
                        )
                    ),
                    ui.card(
                        ui.card_header("second zone"),
                        ui.div(
                            ui.input_slider("average_wind_speed2", "wind speed offset", min=WIND_OFFSET_LIMITS[1][0], max=WIND_OFFSET_LIMITS[1][1], value=0, step=0.1),
                            ui.input_slider("average_wind_tangent2", "wind tangent offset", min=WIND_OFFSET_LIMITS[4][0], max=WIND_OFFSET_LIMITS[4][1], value=0, step=0.1),
                            ui.input_slider("average_wind_cross2", "cross wind offset", min=WIND_OFFSET_LIMITS[7][0], max=WIND_OFFSET_LIMITS[7][1], value=0, step=0.1),
                            ui.input_slider("average_wind_turbulence2", "turbulence offset", min=WIND_OFFSET_LIMITS[10][0], max=WIND_OFFSET_LIMITS[10][1], value=0, step=0.01),
                            class_="zone-controls"
                        )
                    ),
                    ui.card(
                        ui.card_header("third zone"),
                        ui.div(
                            ui.input_slider("average_wind_speed3", "wind speed offset", min=WIND_OFFSET_LIMITS[2][0], max=WIND_OFFSET_LIMITS[2][1], value=0, step=0.1),
                            ui.input_slider("average_wind_tangent3", "wind tangent offset", min=WIND_OFFSET_LIMITS[5][0], max=WIND_OFFSET_LIMITS[5][1], value=0, step=0.1),
                            ui.input_slider("average_wind_cross3", "cross wind offset", min=WIND_OFFSET_LIMITS[8][0], max=WIND_OFFSET_LIMITS[8][1], value=0, step=0.1),
                            ui.input_slider("average_wind_turbulence3", "turbulence offset", min=WIND_OFFSET_LIMITS[11][0], max=WIND_OFFSET_LIMITS[11][1], value=0, step=0.01),
                            class_="zone-controls"
                        )
                    ),
                    class_="zone-stack"
                ),
                ui.div(
                    ui.div(
                        ui.input_action_button("start_button_sliders", "start jump"),
                        ui.output_text("jump_length_sliders"),
                        class_="simulation-actions"
                    ),
                    ui.div(
                        ui.output_plot("traj_plot_sliders", height="600px", width="100%"),
                        class_="simulation-plot"
                    ),
                    class_="simulation-column"
                ),
                ui.div(
                    ui.card(
                        ui.card_header("angles"),
                        ui.help_text("These values shift the full baseline flight profiles to the selected mean."),
                        ui.input_slider("speed", "mean speed [m/s]", min=BODY_MEAN_LIMITS[0][0], max=BODY_MEAN_LIMITS[0][1], value=DEFAULT_BODY_SLIDER_VALUES[0], step=0.1),
                        ui.input_slider("opening_angle", "mean opening angle [deg]", min=BODY_MEAN_LIMITS[1][0], max=BODY_MEAN_LIMITS[1][1], value=DEFAULT_BODY_SLIDER_VALUES[1], step=0.1),
                        ui.input_slider("roll_angle_left", "mean roll angle left [deg]", min=BODY_MEAN_LIMITS[2][0], max=BODY_MEAN_LIMITS[2][1], value=DEFAULT_BODY_SLIDER_VALUES[2], step=0.1),
                        ui.input_slider("roll_angle_right", "mean roll angle right [deg]", min=BODY_MEAN_LIMITS[3][0], max=BODY_MEAN_LIMITS[3][1], value=DEFAULT_BODY_SLIDER_VALUES[3], step=0.1),
                        ui.input_slider("yaw_angle_left", "mean yaw angle left [deg]", min=BODY_MEAN_LIMITS[4][0], max=BODY_MEAN_LIMITS[4][1], value=DEFAULT_BODY_SLIDER_VALUES[4], step=0.1),
                        ui.input_slider("yaw_angle_right", "mean yaw angle right [deg]", min=BODY_MEAN_LIMITS[5][0], max=BODY_MEAN_LIMITS[5][1], value=DEFAULT_BODY_SLIDER_VALUES[5], step=0.1),
                        ui.input_slider("stalling_angle_left", "mean stalling angle left [deg]", min=BODY_MEAN_LIMITS[6][0], max=BODY_MEAN_LIMITS[6][1], value=DEFAULT_BODY_SLIDER_VALUES[6], step=0.1),
                        ui.input_slider("stalling_angle_right", "mean stalling angle right [deg]", min=BODY_MEAN_LIMITS[7][0], max=BODY_MEAN_LIMITS[7][1], value=DEFAULT_BODY_SLIDER_VALUES[7], step=0.1),
                        ui.input_slider("overall_angle_offset", "additional offset for all angles [deg]", min=-1.5, max=1.5, value=0, step=0.1)
                    ),
                    class_="angle-column"
                ),
                class_="slider-simulation-layout"
            )
        ),
        id="tab",
        widths = [2, 10] 
    )
    #style="width: 250px;"
)

#server logic

def server(input, output, session):
    #data = reactive.Value(None)
    #interpolators = reactive.Value(None)
    #t_values = reactive.Value(None)
    # reactive storage
    data_upload = reactive.Value(default_df.copy())  # For CSV upload tab
    data_sliders = reactive.Value(default_df.copy()) # For sliders synthetic jump tab
    #interpolators, t_init = make_interpolators(default_df)
    #interpolators = reactive.Value(interpolators)
    #t_values = reactive.Value(t_init)

    # Reactive vector of all slider values
    #slider_vector = reactive.Value([])
    interpolators_upload, t_values_upload = make_interpolators(default_df)
    interpolators_upload = reactive.Value(interpolators_upload)
    t_values_upload = reactive.Value(t_values_upload)

    interpolators_sliders = reactive.Value(None)
    t_values_sliders = reactive.Value(None)

    frame_index_upload = reactive.Value(0.0)
    animation_start_upload = reactive.Value(None)
    frame_index_sliders = reactive.Value(0.0)
    animation_start_sliders = reactive.Value(None)

    # 20 model controls followed by one global angle-offset setting.
    slider_vector = reactive.Value(DEFAULT_SLIDER_VALUES.copy())

    @reactive.Effect
    @reactive.event(input.file1)
    def load_data():
        file_info = input.file1()
        if not file_info:
            return
        
        try:
            #df = pd.read_csv(file_info[0]["datapath"])
            if "datapath" in file_info[0]:
                df = pd.read_csv(file_info[0]["datapath"])
            else:
                # use the in-memory bytes
                df = pd.read_csv(io.BytesIO(file_info[0]["data"]))

            required_cols = ["X [m]", "Y [m]", "Z [m]", "Time", "Position"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("csv file ne vsebuje vseh zahtevanih stolpcev")
            
            data_upload.set(df)
            #t = df["Time"].to_numpy()
            interp, t = make_interpolators(df)
            interpolators_upload.set(interp)
            t_values_upload.set(t)

            frame_index_upload.set(t[0])
            animation_start_upload.set(None) #ustavi animacijo, ce ta poteka
            
            # Create interpolation functions with bounds_error=False
            #interpolators.set({
            #    'x': interp1d(t, df["X [m]"].to_numpy(), bounds_error=False, fill_value="extrapolate"),
            #    'y': interp1d(t, df["Y [m]"].to_numpy(), bounds_error=False, fill_value="extrapolate"),
            #    'z': interp1d(t, df["Z [m]"].to_numpy(), bounds_error=False, fill_value="extrapolate")
            #})

        except Exception as e:
            print(f"error loading file: {str(e)}")
            #data.set(None)
            #interpolators.set(None)
            #t_values.set(None)

    #new random jump button
    @reactive.Effect
    @reactive.event(input.random_button)
    def random_jump():
        f = random_jump_file()
        if f:
            df = pd.read_csv(f)
            data_upload.set(df)
            interp, t = make_interpolators(df)
            interpolators_upload.set(interp)
            t_values_upload.set(t)

            frame_index_upload.set(t[0])
            animation_start_upload.set(None) #ustavi animacijo, ce ta poteka

    #file info text
    @output
    @render.text
    def file_info():
        file_info = input.file1()
        if not file_info:
            return "using default jump (upload a CSV to replace it)."
        df = data_upload.get()
        if df is None or df.empty:
            return "invalid file: required columns are [X [m], Y [m], Z [m], Time, Position]"
        return f"file loaded: {file_info[0]['name']}"
    
    @output
    @render.text
    def jump_length_upload():
        if not data_upload.get().empty:
            return f"jump length: {data_upload.get()['Position'].iloc[-1]} m"
        return ""

    @output
    @render.text
    def jump_length_sliders():
        if not data_sliders.get().empty:
            return f"jump length: {data_sliders.get()['Position'].iloc[-1]} m"
        return ""

    @reactive.Effect
    def update_slider_vector():
        # read all sliders
        values = [
            input.average_wind_speed1(), input.average_wind_speed2(), input.average_wind_speed3(), 
            input.average_wind_tangent1(), input.average_wind_tangent2(), input.average_wind_tangent3(), 
            input.average_wind_cross1(), input.average_wind_cross2(), input.average_wind_cross3(), 
            input.average_wind_turbulence1(), input.average_wind_turbulence2(), input.average_wind_turbulence3(),
            input.speed(), input.opening_angle(), input.roll_angle_left(), input.roll_angle_right(),
            input.yaw_angle_left(), input.yaw_angle_right(),
            input.stalling_angle_left(), input.stalling_angle_right(),
            input.overall_angle_offset()
        ]
        slider_vector.set(values)

    def make_control_sequence(X_important, sliders):
        """Build in-distribution controls by shifting the baseline profiles."""
        file_length = len(X_important)
        controls = np.empty((file_length, 20), dtype=float)

        # Wind sliders are offsets from the average wind conditions.
        controls[:, :12] = BASELINE_WIND_MEANS + np.asarray(sliders[:12])

        # Preserve the time-varying body profiles. The sliders select their
        # desired means instead of replacing every timestep with a constant.
        selected_body_means = np.asarray(sliders[12:20], dtype=float)
        body_offsets = selected_body_means - BASELINE_BODY_MEANS
        controls[:, 12:20] = BASELINE_BODY_PROFILES + body_offsets

        # Apply the global angle offset to angles only, not to speed.
        controls[:, 13:20] += sliders[20]
        return controls

    def app_sim_AVGbase(X_important, models, sliders=np.zeros(21)):

        A, B, C, D = models[0], models[1], models[2], models[3]
        x0 = X_important[0]       #initial state...maybe don't adjust
        X_sim = [x0]
        X_coord = [x0[:3]]
        file_length = np.shape(X_important)[0]
        U_seq = make_control_sequence(X_important, sliders)

        for t in range(file_length-1):
            x_next = A @ X_sim[-1] + B @ U_seq[t]
            X_sim.append(x_next)
            X_coord.append(x_next[:3])
        X_sim = np.array(X_sim)


        return X_coord

    def make_synthetic_jump(models, X_important, sliders):

        coordinates = app_sim_AVGbase(X_important, models, sliders)  # returns list of [X,Y,Z]
        dff = pd.DataFrame(coordinates, columns=["X [m]", "Y [m]", "Z [m]"])

        num_rows = len(coordinates)
    
        # Time column
        time_column = [0]
        if num_rows > 1:
            time_column.append(4)
        for i in range(2, num_rows):
            time_column.append(4 + (i-1)*0.05)
        dff["Time"] = time_column
        
        # Position column
        position = [0] * num_rows
        x_end, z_end = dff["X [m]"].iloc[-1], dff["Z [m]"].iloc[-1]
        distance = round(math.sqrt(x_end**2 + z_end**2))
        position[-1] = distance
        dff["Position"] = position
        
        return dff

    @reactive.Effect
    def update_synthetic_jump():
        sliders = slider_vector.get()
        if sliders is None:
            return

        # Generate synthetic jump DataFrame
        synthetic_df = make_synthetic_jump(models, X_important, sliders)

        # Store in reactive data for plotting
        data_sliders.set(synthetic_df)

        # Update interpolators and time values for animation
        interp, t = make_interpolators(synthetic_df)
        interpolators_sliders.set(interp)
        t_values_sliders.set(t)
        frame_index_sliders.set(t[0])
        animation_start_sliders.set(None)  # stop previous animation if running


    #to je samo da mi na app izpiše celo tabelo vrednosti na slajderjih, tega se ne rabi
    @output
    @render.text
    def slider_values():
        # get the current vector
        values = slider_vector.get()
        return str(values)

    frame_index = reactive.Value(0.0)
    animation_start = reactive.Value(None)

    @output
    @render.plot
    def traj_plot_upload():
        df = data_upload.get()
        if df is None or df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Please upload a CSV file with ski jump data", 
                   ha='center', va='center')
            ax.axis('off')
            return fig
        
        #df = data.get()
        x = df["X [m]"].to_numpy()
        y = df["Y [m]"].to_numpy()
        z = df["Z [m]"].to_numpy()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, color="blue", linewidth=2)

        current_time = frame_index_upload.get()
        t_max = t_values_upload.get()[-1]
        
        if current_time > t_max:
            current_time = t_max
            
        if interpolators_upload.get():
            interp = interpolators_upload.get()
            ax.plot([interp['x'](current_time)], 
                    [interp['y'](current_time)], 
                    [interp['z'](current_time)], 
                    marker='o', color='red', markersize=8)
        
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_ylim(-15, 15)
        ax.grid(True)
        return fig
    
    @output
    @render.plot
    def traj_plot_sliders():
        df = data_sliders.get()
        if df is None or df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Please upload a CSV file with ski jump data", 
                   ha='center', va='center')
            ax.axis('off')
            return fig
        
        #df = data.get()
        x = df["X [m]"].to_numpy()
        y = df["Y [m]"].to_numpy()
        z = df["Z [m]"].to_numpy()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, color="blue", linewidth=2)

        current_time = frame_index_sliders.get()
        t_max = t_values_sliders.get()[-1]
        
        if current_time > t_max:
            current_time = t_max
            
        if interpolators_sliders.get():
            interp = interpolators_sliders.get()
            ax.plot([interp['x'](current_time)], 
                    [interp['y'](current_time)], 
                    [interp['z'](current_time)], 
                    marker='o', color='red', markersize=8)
        
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_ylim(-15, 15)
        ax.grid(True)
        return fig
    
    #ko se pritisne "start jump" gumb
    @reactive.Effect
    @reactive.event(input.start_button)
    def start_animation_upload():
        if data_upload.get() is None:
            return
        animation_start_upload.set(pytime.time())
        frame_index_upload.set(0)
        reactive.invalidate_later(0.01)  # Changed from 0 to 0.01
    
    @reactive.Effect
    @reactive.event(input.start_button_sliders)
    def start_animation_sliders():
        if data_sliders.get() is None:
            return
        animation_start_sliders.set(pytime.time())
        frame_index_sliders.set(0)
        reactive.invalidate_later(0.01)

    @reactive.Effect
    def update_animation_upload():
        if animation_start_upload.get() is None or data_upload.get() is None:
            return
        
        elapsed_time = pytime.time() - animation_start_upload.get()
        t_max = t_values_upload.get()[-1]
        
        if elapsed_time > t_max:
            elapsed_time = t_max
            animation_start_upload.set(None)
        
        frame_index_upload.set(elapsed_time)
        reactive.invalidate_later(0.01)
        #current_frame = 0 
        # Find the first frame where time > elapsed
        #for i in range(len(t)):
        #    if t[i] > trenutni_cas:
        #        break
        #    current_frame = i
        
        #frame_index.set(current_frame)

        #nadaljuj animacijo, ce se se ni zakljucila
        #if current_frame < len(t) - 1: #nacrtuje naslednji update
        #    next_update = t[current_frame + 1] - trenutni_cas
        #    reactive.invalidate_later(max(0.01, next_update)) #minimum 10 ms delay
        #else:
        #    animation_start.set(None) #ustavi animacijo, ce pride do konca

    @reactive.Effect
    def update_animation_sliders():
        if animation_start_sliders.get() is None or data_sliders.get() is None:
            return
        
        elapsed = pytime.time() - animation_start_sliders.get()
        t_max = t_values_sliders.get()[-1]

        if elapsed > t_max:
            elapsed = t_max
            animation_start_sliders.set(None)

        frame_index_sliders.set(elapsed)
        reactive.invalidate_later(0.01)

#skombinira UI in server v shiny app
app = App(app_ui, server)

