from src.data_loader import load_events, load_total
from src.plots import plot_all_combos, TOTAL_FEATURE_COLS, EVENTS_FEATURE_COLS

if __name__ == "__main__":
    events = load_events()
    total  = load_total()

    print("events columns:", events.columns.tolist())
    print("total  columns:", total.columns.tolist())

    # Pairwise plots for the full per-event dataset
    plot_all_combos(
        total,
        combos=TOTAL_FEATURE_COLS,
        label_col="label_name",
        row_col="pair_id",
        col_col="label",
        threshold_row=None,
        threshold_col=None,
        save=True,
    )

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