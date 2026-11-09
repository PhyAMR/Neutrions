from src.data_loader import load_events, load_total, load_false_events
from src.plots import plot_all_combos, plot_distributions 


_RAW_TO_PHYS: dict[str, str] = {
    "time":     "t",
    "energy":   "E",
    "x":        "x",
    "y":        "y",
    "z":        "z",
    "r":        "r",
    "r2":       "r2",
    "phi":      "phi",
    "distance": "D",
}

# Individual per-event columns kept in the output (both _1 and _2 variants).
# Named t_1, E_1, x_1 … and t_2, E_2, x_2 …
_INDIV_BASE = list(_RAW_TO_PHYS.values())   # ['t','E','x','y','z','r','r2','phi','D']

EVENTS_INDIV_COLS = (
    [f"{p}_1" for p in _INDIV_BASE] +
    [f"{p}_2" for p in _INDIV_BASE] + 
    ["dwall_1", "dwall_2"]
)

# Delta columns: Δt, ΔE, Δx, Δy, Δz, Δr, Δr2, Δphi, ΔD + derived ΔR
EVENTS_DELTA_COLS = [f"d{p}" for p in _INDIV_BASE] + ["dR"]
# dR = sqrt(dx²+dy²+dz²) — 3D spatial separation, key IBD discriminant

# Label columns kept in the output: We store all the labels that indicate the nature of
# the event
EVENTS_LABEL_COLS = ["pair_id_1", "pair_id_2", "label_1", "label_2", "label_name_1", "label_name_2", "is_ibd", "false_ibd", "pair_type"]

EVENTS_COUNTS_COLS = ["k_window", "dt_shifted", "n_before", "n_on", "n_gap",
                     "n_in_window", "is_dc"]

EVENTS_FEATURE_COLS = EVENTS_INDIV_COLS + EVENTS_DELTA_COLS + EVENTS_COUNTS_COLS



if __name__ == "__main__":
    events = load_events()
    #total  = load_total()
    

    print("events columns:", events.columns.tolist())
    #print("total  columns:", total.columns.tolist())
    
    #Pairwise plots for the full per-event dataset
    #plot_all_combos(
    #    total,
    #    combos=TOTAL_FEATURE_COLS,
    #    label_col="label_name",
    #    row_col="pair_id",
    #    col_col="label",
    #    threshold_row=None,
    #    threshold_col=None,
    #    save=True,
    #)
#
    # Pairwise plots for the paired-event delta dataset
    plot_all_combos(
        events,
        combos=EVENTS_FEATURE_COLS,
        label_col="label_name_2",
        row_col="pair_id_2",
        col_col="label_2",
        threshold_row=None,
        threshold_col=None,
        save=True,
    )

    

    plot_distributions(
        events,
        tag="energy_events",
        cat = "label_name_2",
        cols = ["E_1"],
    )

    #false_events = load_false_events(columns = ["E_1", "label_name_2"])
    #print("false_events columns:", false_events.columns.tolist())
    #plot_distributions(
    #    false_events,
    #    tag="energy_false_events",
    #    cat = "label_name_2",
    #    cols = ["E_1"],
    #)

    #plot_distributions(
    #    events,
    #    tag="full_events",
    #    cat = "label_name_2",
    #)
    #plot_distributions(
    #    false_events,
    #    tag="full_false_events",
    #    cat = "label_name_2",
    #)
