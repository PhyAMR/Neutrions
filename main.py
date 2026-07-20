from src.data_loader import load_events, load_total, load_false_events
from src.plots import plot_distributions 


if __name__ == "__main__":
    #events = load_events()
    #total  = load_total()
    

    #print("events columns:", events.columns.tolist())
    #print("total  columns:", total.columns.tolist())
    
    # Pairwise plots for the full per-event dataset
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
    ## Pairwise plots for the paired-event delta dataset
    #plot_all_combos(
    #    events,
    #    combos=EVENTS_FEATURE_COLS,
    #    label_col="label_name_2",
    #    row_col="pair_id_2",
    #    col_col="label_2",
    #    threshold_row=None,
    #    threshold_col=None,
    #    save=True,
    #)

    

    #plot_distributions(
    #    events,
    #    tag="energy_events",
    #    cat = "label_name_2",
    #    cols = ["E_1"],
    #)

    false_events = load_false_events(columns = ["E_1", "label_name_2"])
    print("false_events columns:", false_events.columns.tolist())
    plot_distributions(
        false_events,
        tag="energy_false_events",
        cat = "label_name_2",
        cols = ["E_1"],
    )

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