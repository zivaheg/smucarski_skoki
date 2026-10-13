import pandas as pd

def flatten_wind_blob(blob):
    rows = [r.split('|') for r in blob.split(';') if r.strip()]     #razdelimo na vrstice, razdelimo na vrednosti
    out = {}
    for row in rows:
        wind_name = row[1]  #Wi
        out[f"{wind_name}_Speed"] = float(row[2])
        out[f"{wind_name}_Tangent"] = float(row[3])
        out[f"{wind_name}_Turbulence"] = float(row[4])
        out[f"{wind_name}_CleanTan"] = float(row[5])
        out[f"{wind_name}_Cross"] = float(row[6])
        
    return pd.Series(out)   #series da lahko dodamo v tabelo

def timestamp_to_seconds(ts):
    """Return seconds within the local day, including hours and minutes."""
    time_str = ts.split('T', 1)[1]
    time_str = time_str.removesuffix('Z').split('+', 1)[0]
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)
