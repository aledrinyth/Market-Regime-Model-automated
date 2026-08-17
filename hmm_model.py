import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def evaluate_model(model, X_train, X_test, df_train):
        """
        Evaluate a fitted Gaussian HMM.

        Parameters
        ----------
        model : GaussianHMM
            Trained HMM.
        X_train : ndarray
            Training feature matrix.
        X_test : ndarray
            Test feature matrix.
        df_train : DataFrame
            Training dataframe containing the original (unscaled) features.

        Returns
        -------
        results : dict
            Dictionary containing model metrics and statistics.
        """

        # -------------------------------
        # Log-likelihood
        # -------------------------------

        train_ll = model.score(X_train)
        test_ll = model.score(X_test)

        # -------------------------------
        # Number of parameters
        # (Approximation for Gaussian HMM)
        # -------------------------------

        n_states = model.n_components
        n_features = X_train.shape[1]

        n_transition = n_states * (n_states - 1)

        n_startprob = n_states - 1

        n_means = n_states * n_features

        if model.covariance_type == "full":
            n_covars = n_states * (n_features * (n_features + 1) / 2)
        elif model.covariance_type == "diag":
            n_covars = n_states * n_features
        else:
            raise NotImplementedError("Only full and diag covariance supported.")

        k = n_transition + n_startprob + n_means + n_covars

        n = len(X_train)

        aic = -2 * train_ll + 2 * k
        bic = -2 * train_ll + np.log(n) * k

        # -------------------------------
        # Hidden states
        # -------------------------------

        states = model.predict(X_train)

        df = df_train.copy()
        df["State"] = states

        # -------------------------------
        # State statistics
        # -------------------------------

        state_summary = (
            df.groupby("State")
            .agg(
                Count=("State", "count"),
                Mean_Return=("Daily Log Return", "mean"),
                Return_STD=("Daily Log Return", "std"),
                Volatility=("20-Day Rolling Volatility", "mean"),
                Drawdown=("Max Drawdown", "mean"),
                RSI=("RSI", "mean"),
                ATR=("ATR", "mean"),
                VIX=("Close_VIX", "mean"),
            )
            .round(4)
        )

        # -------------------------------
        # Average regime duration
        # -------------------------------

        durations = []

        current_state = states[0]
        duration = 1

        for s in states[1:]:

            if s == current_state:
                duration += 1

            else:
                durations.append(duration)
                current_state = s
                duration = 1

        durations.append(duration)

        avg_duration = np.mean(durations)

        # -------------------------------
        # Number of switches
        # -------------------------------

        switches = np.sum(states[:-1] != states[1:])

        # -------------------------------
        # Regime probabilities
        # -------------------------------

        regime_probs = model.predict_proba(X_train)

        # -------------------------------
        # Results dictionary
        # -------------------------------

        results = {
            "Number of States": n_states,
            "Train Log Likelihood": train_ll,
            "Test Log Likelihood": test_ll,
            "AIC": aic,
            "BIC": bic,
            "Average Regime Duration": avg_duration,
            "Number of Switches": switches,
            "Transition Matrix": model.transmat_,
            "Start Probabilities": model.startprob_,
            "State Summary": state_summary,
            "Regime Probabilities": regime_probs,
        }

        return results

def find_optimal_hmm_states(X_train, X_test, df_train, max_states=10):


    comparison = []

    for n_states in range(2, max_states):

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=500,
            random_state=42
        )

        model.fit(X_train)

        results = evaluate_model(model, X_train, X_test, df_train)

        comparison.append({
            "States": n_states,
            "Train LL": results["Train Log Likelihood"],
            "Test LL": results["Test Log Likelihood"],
            "AIC": results["AIC"],
            "BIC": results["BIC"],
            "Avg Duration": results["Average Regime Duration"],
            "Switches": results["Number of Switches"]
        })

    comparison_df = pd.DataFrame(comparison)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.expand_frame_repr", False)
    print(comparison_df)


def main():

    feature_cols = [
        "Daily Log Return_scaled",
        "20-Day Rolling Volatility_scaled",
        "Distance from 200-Day MA_scaled",
        "Max Drawdown_scaled",
        "Volume Change_scaled",
        "ATR_scaled",
        "RSI_scaled",
        "MACD_scaled",
        "Close_VIX",
    ]

    # Load train/test files
    train_path = "data/VAS_historical_train_data_engineered.csv"
    test_path = "data/VAS_historical_test_data_engineered.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Prepare the feature matrix
    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values

    # Fit the HMM
    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=500,
        random_state=42,
    )

    model.fit(X_train)
    train_states = model.predict(X_train)
    test_states = model.predict(X_test)

    df_train["State"] = train_states
    df_test["State"] = test_states

    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    df_combined["Date"] = pd.to_datetime(df_combined["Date"])
    df_combined = df_combined.sort_values("Date").reset_index(drop=True)



    results = evaluate_model(model, X_train, X_test, df_train)

    print(f"Train Log Likelihood : {results['Train Log Likelihood']:.2f}")
    print(f"Test Log Likelihood  : {results['Test Log Likelihood']:.2f}")
    print(f"AIC                  : {results['AIC']:.2f}")
    print(f"BIC                  : {results['BIC']:.2f}")
    print(f"Average Duration     : {results['Average Regime Duration']:.2f}")
    print(f"Number of Switches   : {results['Number of Switches']}")

    print("\nTransition Matrix")
    print(results["Transition Matrix"])

    print("\nState Summary")
    print(results["State Summary"])



    # Manually Evaluating and Looking at Model
    print(model.transmat_)

    summary = (
        df_train
        .groupby("State")
        .agg({
            "Daily Log Return": ["mean", "std"],
            "20-Day Rolling Volatility": "mean",
            "Max Drawdown": "mean",
            "RSI": "mean",
            "Close": "count"
        })
    )
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.expand_frame_repr", False)

    print(summary.to_string())

    df_combined.to_csv("data/VAS_historical_data_with_three_states.csv", index=False)

    return df_combined, model



# if 4 states:
# Regime 0 = Low Volatility & Bullish
# Regime 1 = High Volatility & Bearish/ Recovery
# Regime 2 = Highish Volatility & Bullish/ Recovery
# Regime 3 = High Volatility & Bearish

# if 3 states:
# Regime 0 = Low Volatility & Recovery
# Regime 1 = High Volatility & Bullish/ Rally
# Regime 2 = High Volatility & Bearish

def plot_hmm_plotly():

    # get the pre requisite data
    df_combined, model  = main()

    df = df_combined.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    no_of_colours = model.n_components

    # regime config for coluors

    if no_of_colours == 3:
        regime_config = {
            0: {
                "label": "Low Volatility & Recovery",
                "bg_color": "rgba(46, 204, 113, 0.22)",
                "solid_color": "#2ecc71",
            },
            1: {
                "label": "High Volatility & Bullish / Rally",
                "bg_color": "rgba(241, 196, 15, 0.22)",
                "solid_color": "#f1c40f",
            },
            2: {
                "label": "High Volatility & Bearish",
                "bg_color": "rgba(231, 76, 60, 0.22)",
                "solid_color": "#e74c3c",
            },

        }
    elif no_of_colours == 4: 
        regime_config = {
            0: {
                "label": "Low Volatility & Bullish",
                "bg_color": "rgba(46, 204, 113, 0.22)",  # Emerald Green
                "solid_color": "#2ecc71",
            },
            1: {
                "label": "High Volatility & Bearish / Recovery",
                "bg_color": "rgba(241, 196, 15, 0.22)",  # Amber Yellow
                "solid_color": "#f1c40f",
            },
            2: {
                "label": "Highish Volatility & Bullish / Recovery",
                "bg_color": "rgba(155, 89, 182, 0.22)",  # Amethyst Purple - actually looks nice
                "solid_color": "#9b59b6",
            },
            3: {
                "label": "High Volatility & Bearish",
                "bg_color": "rgba(231, 76, 60, 0.22)",  # Alizarin Red
                "solid_color": "#e74c3c",
            },
        }
    else:
        print("Wrong number of colours meaning wrong number of regimes")

    fig = go.Figure()

    # Add background shaded bands for consecutive states
    start_idx = 0
    current_state = int(df["State"].iloc[0])

    for i in range(1, len(df)):
        next_state = int(df["State"].iloc[i])
        # Checks whether the next state is the same state so it can chain states for the highlightinf
        if next_state != current_state:
            config = regime_config.get(
                # Set the fallback colour as a light gray with 20% opacity - should be pretty obvious
                current_state, {"bg_color": "rgba(200, 200, 200, 0.2)"}
            )
            fig.add_vrect(
                x0=df["Date"].iloc[start_idx],
                x1=df["Date"].iloc[i - 1],
                fillcolor=config["bg_color"],
                layer="below",
                line_width=0,
            )
            start_idx = i
            current_state = next_state

    # Add final regime segment
    config = regime_config.get(
        # fallback colour is still gray
        current_state, {"bg_color": "rgba(200, 200, 200, 0.2)"}
    )
    fig.add_vrect(
        x0=df["Date"].iloc[start_idx],
        x1=df["Date"].iloc[-1],
        fillcolor=config["bg_color"],
        layer="below",
        line_width=0,
    )

    # Provides the price line animation and hover animation
    df["Regime_Label"] = df["State"].map(
        lambda s: regime_config.get(s, {}).get("label", f"State {s}")
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="VAS Close Price",
            line=dict(color="#1f77b4", width=2),
            customdata=df["Regime_Label"],
            hovertemplate=(
                "<b>Date:</b> %{x|%b %d, %Y}<br>"
                "<b>Close Price:</b> $%{y:.2f}<br>"
                "<b>Regime:</b> %{customdata}"
                "<extra></extra>"
            ),
        )
    )

    # Add the Legend Items for Regime Colors
    for state, config in regime_config.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=config["label"],
                marker=dict(size=12, color=config["solid_color"], symbol="square"),
                showlegend=True,
            )
        )

    # Styling and Layout for Streamlit
    fig.update_layout(
        title={
            "text": "<b>VAS Close Price with Hidden Markov Model Regimes</b>",
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
        },
        xaxis_title="Date",
        yaxis_title="Close Price ($AUD)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        height=600,
    )

    return fig




def plot_hmm_states(df_combined, model):



    # Plot the price and highlight state regions in the background
    fig, ax = plt.subplots(figsize=(15, 7))

    state_colors = {state: plt.cm.tab10(state % 10) for state in range(model.n_components)}

    # Shade each consecutive state segment
    start_idx = 0
    current_state = int(df_combined["State"].iloc[0])
    for i in range(1, len(df_combined)):
        next_state = int(df_combined["State"].iloc[i])
        if next_state != current_state:
            ax.axvspan(
                df_combined["Date"].iloc[start_idx],
                df_combined["Date"].iloc[i - 1],
                color=state_colors[current_state],
                alpha=0.12,
                zorder=0,
            )
            start_idx = i
            current_state = next_state

    ax.axvspan(
        df_combined["Date"].iloc[start_idx],
        df_combined["Date"].iloc[-1],
        color=state_colors[current_state],
        alpha=0.12,
        zorder=0,
    )

    # Price line
    ax.plot(df_combined["Date"], df_combined["Close"], label="VAS Close Price", color="blue", linewidth=1.5, zorder=2)

    # Create a legend for the regimes in the same place as ax.legend()
    regime_handles = []
    regime_labels = ['Low Volatility & Recovery', 'High Volatility & Bullish/ Rally', 'High Volatility & Bearish']
    for state in range(model.n_components):
        handle = plt.Line2D(
            [0], [0],
            color=state_colors[state],
            lw=8,
            alpha=0.2,
            label=f"Regime {state}",
        )
        regime_handles.append(handle)

    ax.set_title("VAS Close Price with Hidden Markov Model States")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend(
        handles=regime_handles,
        labels=regime_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    fig.tight_layout()
    plt.savefig("data/VAS_historical_data_with_three_states.png", dpi=300, bbox_inches="tight")
    plt.show()



# plot_hmm_plotly(df_combined, model)

# plot_hmm_states(df_combined, model)

if __name__ == "__main__":
    main()