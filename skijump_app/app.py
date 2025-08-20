from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as pytime
from scipy.interpolate import interp1d
import io
import os
import random

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

#definiram UI
app_ui = ui.page_fluid(
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
                ui.output_text("jump_length"),
                style="text-align: center; margin-top: 20px;"
            ),
            ui.div(
                ui.output_plot("traj_plot", height="640px", width="960px"),
                #najprej je bilo 800 in 960, zdej je 80% tega
                style="display: flex; justify-content: center;"
            )
        ),
        ui.nav_panel(
            "use sliders",
            ui.h2("Ski jump animation", style="text-align: center;"), #naslov
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("1. cona"),
                    ui.input_slider("average_wind_speed1", "povprečna hitrost vetra v coni 1", min=-5, max=5, value=0, step=0.1),
                    ui.input_slider("average_wind_tangent1", "povprečna moč vetra v coni 1", min=-5, max=5, value=0, step=0.1),
                    ui.input_slider("average_wind_cross1", "povprečna pravokotna moč vetra v coni 1", min=-5, max=5, value=0, step=0.1),
                    ui.input_slider("average_wind_turbulence1", "povprečna turbulenca vetra v coni 1", min=-5, max=5, value=0, step=0.1)
                ),
                ui.card(
                    ui.card_header("2. cona"),
                    ui.input_slider("average_wind_speed2", "povprečna hitrost vetra v coni 2", min=-5, max=5, value=0, step=0.1),
                    ui.input_slider("average_wind_tangent2", "povprečna moč vetra v coni 2", min=-5, max=5, value=0, step=0.1),
                    ui.input_slider("average_wind_cross2", "povprečna pravokotna moč vetra v coni 2", min=-5, max=5, value=0, step=0.1),
                    ui.input_slider("average_wind_turbulence2", "povprečna turbulenca vetra v coni 2", min=-5, max=5, value=0, step=0.1)
                ),
                ui.card(
                    ui.card_header("3. cona"),
                    ui.input_slider("average_wind_speed3", "povprečna hitrost vetra v coni 3", min=-5, max=5, value=0, step=0.1),
                    ui.input_slider("average_wind_tangent3", "povprečna moč vetra v coni 3", min=-5, max=5, value=0, step=0.1),
                    ui.input_slider("average_wind_cross3", "povprečna pravokotna moč vetra v coni 3", min=-5, max=5, value=0, step=0.1),
                    ui.input_slider("average_wind_turbulence3", "povprečna turbulenca vetra v coni 3", min=-5, max=5, value=0, step=0.1)
                ),
                ui.card(
                    ui.card_header("koti"),
                    ui.input_slider("opening_angle", "opening angle", min=0, max=90, value=0, step=0.1),
                    ui.input_slider("stalling_angle_left", "stalling angle left", min=-20, max=20, value=0, step=0.1),
                    ui.input_slider("stalling_angle_right", "stalling angle right", min=-20, max=20, value=0, step=0.1),
                    ui.input_slider("roll_angle_left", "roll angle left", min=-60, max=60, value=0, step=0.1),
                    ui.input_slider("roll_angle_right", "roll angle rigth", min=-60, max=60, value=0, step=0.1),
                    ui.input_slider("yaw_angle_left", "yaw angle left", min=0, max=20, value=0, step=0.1),
                    ui.input_slider("yaw_angle_right", "yaw angle right", min=-20, max=0, value=-20, step=0.1)
                )
            ),
            ui.div(
                ui.input_action_button("start_button", "start jump"), #gumb, ki zacne animacijo
                style="text-align: center; margin-top: 30px;" #30 pixlov prostora
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
    data = reactive.Value(default_df)
    interpolators, t_init = make_interpolators(default_df)
    interpolators = reactive.Value(interpolators)
    t_values = reactive.Value(t_init)

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
            
            data.set(df)
            #t = df["Time"].to_numpy()
            interp, t = make_interpolators(df)
            interpolators.set(interp)
            t_values.set(t)

            frame_index.set(t[0])
            animation_start.set(None) #ustavi animacijo, ce ta poteka
            
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
            data.set(df)
            interp, t = make_interpolators(df)
            interpolators.set(interp)
            t_values.set(t)

            frame_index.set(t[0])
            animation_start.set(None) #ustavi animacijo, ce ta poteka

    #file info text
    @output
    @render.text
    def file_info():
        file_info = input.file1()
        if not file_info:
            return "using default jump (upload a CSV to replace it)."
        df = data.get()
        if df is None or df.empty:
            return "invalid file: required columns are [X [m], Y [m], Z [m], Time, Position]"
        return f"file loaded: {file_info[0]['name']}"
    
    @output
    @render.text
    def jump_length():
        if not data.get().empty:
            return f"dolžina skoka: {data.get()['Position'].iloc[-1]} m"
        return ""

    frame_index = reactive.Value(0.0)
    animation_start = reactive.Value(None)

    @output
    @render.plot
    def traj_plot():
        df = data.get()
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

        current_time = frame_index.get()
        t_max = t_values.get()[-1]
        
        if current_time > t_max:
            current_time = t_max
            
        if interpolators.get():
            interp = interpolators.get()
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
    def start_animation():
        if data.get() is None:
            return
        animation_start.set(pytime.time())
        frame_index.set(0)
        reactive.invalidate_later(0.01)  # Changed from 0 to 0.01
    
    @reactive.Effect
    def update_animation():
        if animation_start.get() is None or data.get() is None:
            return
        
        elapsed_time = pytime.time() - animation_start.get()
        t_max = t_values.get()[-1]
        
        if elapsed_time > t_max:
            elapsed_time = t_max
            animation_start.set(None)
        
        frame_index.set(elapsed_time)
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


#skombinira UI in server v shiny app
app = App(app_ui, server)

