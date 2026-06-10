import base64
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_datasets, build_lookups
from src.prediction import predict_match, get_h2h_score
from src.tournament import simulate_tournament
from src.styling import (
    apply_global_styles,
    team_label,
    format_probability,
)

# 1. Page Configuration (Must be first Streamlit call)
st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

# Inject CSS to load a flag-compatible emoji font globally
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap');
    html, body, [data-testid="stText"], kbd, span {
        font-family: 'Noto Color Emoji', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

apply_global_styles()

STATIC_DIR = Path(__file__).parent / "data" / "static"

PAGE_BACKGROUNDS = {
    "probabilities": "fifa_wc2026_tournamnet_america.webp",
    "live_simulation": "fifa_wc2026_tournamnet_america.webp",
    "match_explorer": "fifa_wc2026_tournamnet_america.webp",
}

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _encode_image(filename):
    """Return (mime_type, base64_string) for an image in the static dir."""
    path = STATIC_DIR / filename
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return mime, encoded


def set_page_background(filename, tile_px=180):
    """Tile a small, 80% transparent background image across the current page."""
    mime, encoded = _encode_image(filename)
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)),
                url("data:{mime};base64,{encoded}");
            background-size: auto, {tile_px}px auto;
            background-position: center;
            background-attachment: fixed;
            background-repeat: repeat;
        }}
        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_trophies():
    """Animate trophies raining down the page (a balloons-style celebration)."""
    trophies = "".join(
        f'<div class="trophy" style="left:{left}%; '
        f'animation-delay:{delay}s; animation-duration:{duration}s; '
        f'font-size:{size}px;">🏆</div>'
        for left, delay, duration, size in [
            (5, 0.0, 3.2, 36),
            (15, 0.4, 3.8, 28),
            (27, 0.9, 3.0, 44),
            (38, 0.2, 4.2, 30),
            (50, 0.6, 3.5, 38),
            (62, 1.1, 3.1, 26),
            (73, 0.3, 4.0, 42),
            (84, 0.8, 3.6, 32),
            (93, 0.5, 3.3, 40),
        ]
    )

    st.markdown(
        f"""
        <style>
        @keyframes trophy-fall {{
            0% {{ transform: translateY(-10vh) rotate(0deg); opacity: 0; }}
            10% {{ opacity: 1; }}
            100% {{ transform: translateY(110vh) rotate(360deg); opacity: 0; }}
        }}
        .trophy-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
        }}
        .trophy {{
            position: absolute;
            top: 0;
            animation-name: trophy-fall;
            animation-timing-function: ease-in;
            animation-iteration-count: 1;
        }}
        </style>
        <div class="trophy-container">{trophies}</div>
        """,
        unsafe_allow_html=True,
    )


data = load_datasets()
lookups = build_lookups()

df_probs = data["probabilities"]
df_groups = data["groups"]

team_to_elo = lookups["team_to_elo"]
team_to_form = lookups["team_to_form"]
team_to_fifa_rank = lookups["team_to_fifa_rank"]
team_to_fifa_rank_change = lookups["team_to_fifa_rank_change"]


def show_header():
    st.markdown(
        """
        <div class="main-title">⚽ World Cup 2026 Predictor</div>
        <div class="sub-title">
        Poisson goal model + Monte Carlo simulation + interactive tournament demo.
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_probability_table(df):
    """Format probability columns for display."""
    display_df = df.copy()
    probability_cols = [
        "r32_prob",
        "r16_prob",
        "qf_prob",
        "sf_prob",
        "final_prob",
        "winner_prob",
    ]

    for col in probability_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x * 100:.1f}%")

    if "team" in display_df.columns:
        display_df["team"] = display_df["team"].apply(team_label)

    return display_df


def page_probabilities():
    set_page_background(PAGE_BACKGROUNDS["probabilities"])
    show_header()

    st.subheader("🏆 Precomputed tournament probabilities")
    st.write("This page loads the saved Monte Carlo results from CSV, so it is fast.")

    col1, col2, col3 = st.columns(3)
    top_team = df_probs.sort_values("winner_prob", ascending=False).iloc[0]

    # Re-use our country code mapping for the top card flag
    country_iso_map = {
        "Argentina": "ar", "Australia": "au", "Austria": "at", "Belgium": "be", 
        "Brazil": "br", "Canada": "ca", "Colombia": "co", "Croatia": "hr", 
        "Curaçao": "cw", "Czech Republic": "cz", "DR Congo": "cd", "Ecuador": "ec", 
        "Egypt": "eg", "England": "gb-eng", "France": "fr", "Germany": "de", 
        "Haiti": "ht", "Iran": "ir", "Iraq": "iq", "Italy": "it", "Japan": "jp", 
        "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl", "Norway": "no", 
        "Panama": "pa", "Paraguay": "py", "Portugal": "pt", "Qatar": "qa", 
        "Saudi Arabia": "sa", "Scotland": "gb-sct", "Senegal": "sn", "South Africa": "za", 
        "South Korea": "kr", "Spain": "es", "Switzerland": "ch", "Tunisia": "tn", 
        "Turkey": "tr", "Ukraine": "ua", "United States": "us", "Uruguay": "uy", 
        "Uzbekistan": "uz"
    }

    top_team_name = str(top_team["team"])
    top_team_iso = country_iso_map.get(top_team_name)

    with col1:
        # 1. Render the clean flag image directly above the metric title
        if top_team_iso:
            st.image(f"https://flagcdn.com/w80/{top_team_iso}.png", width=60)
        else:
            st.write("🏳️")
            
        # 2. Display the pure text metric without Windows fallback text issues
        st.metric("Most likely winner", top_team_name)

    with col2:
        st.metric("Win probability", format_probability(top_team["winner_prob"]))

    with col3:
        st.metric("FIFA rank", int(top_team["fifa_rank"]))

    st.markdown("### Top winner probabilities")
    top_n = st.slider("How many teams to show?", min_value=5, max_value=48, value=15)

    chart_df = (
        df_probs
        .sort_values("winner_prob", ascending=False)
        .head(top_n)
        .copy()
    )

    chart_df["team_label"] = chart_df["team"].apply(team_label)

    def rank_color(rank):
        if rank < 3:
            return "#D62828"   # red
        if rank < 10:
            return "#2A9D3F"   # green
        return "#1A3A8F"       # dark blue

    bar_colors = [rank_color(i) for i in range(len(chart_df))]

    fig = go.Figure(
        go.Bar(
            x=chart_df["team_label"],
            y=chart_df["winner_prob"],
            marker_color=bar_colors,
            hovertemplate="%{x}<br>Win probability: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Win probability",
        yaxis_tickformat=".0%",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    
    axis_style = dict(
        showline=True,
        linecolor="white",
        linewidth=2,
        ticks="outside",
        tickcolor="white",
        zeroline=False,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(
        **axis_style,
        gridcolor="rgba(255,255,255,0.15)",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Full probability table")

    table_cols = [
        "team",
        "confederation",
        "fifa_rank",
        "rank_change",
        "elo",
        "form_score",
        "r32_prob",
        "r16_prob",
        "qf_prob",
        "sf_prob",
        "final_prob",
        "winner_prob",
    ]

    # 1. Copy the clean raw data
    display_df = df_probs[table_cols].copy()
    
    # 2. Map percentage columns manually
    prob_cols = ["r32_prob", "r16_prob", "qf_prob", "sf_prob", "final_prob", "winner_prob"]
    for col in prob_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x * 100:.1f}%")

    # 3. Dynamic Flag Mapping: Convert names to real image flags instead of text codes
    # This dictionary maps your exact team names to their lowercase ISO country codes
    country_iso_map = {
        "Argentina": "ar", "Australia": "au", "Austria": "at", "Belgium": "be", 
        "Brazil": "br", "Canada": "ca", "Colombia": "co", "Croatia": "hr", 
        "Curaçao": "cw", "Czech Republic": "cz", "DR Congo": "cd", "Ecuador": "ec", 
        "Egypt": "eg", "England": "gb-eng", "France": "fr", "Germany": "de", 
        "Haiti": "ht", "Iran": "ir", "Iraq": "iq", "Italy": "it", "Japan": "jp", 
        "Mexico": "mx", "Morocco": "ma", "Netherlands": "nl", "Norway": "no", 
        "Panama": "pa", "Paraguay": "py", "Portugal": "pt", "Qatar": "qa", 
        "Saudi Arabia": "sa", "Scotland": "gb-sct", "Senegal": "sn", "South Africa": "za", 
        "South Korea": "kr", "Spain": "es", "Switzerland": "ch", "Tunisia": "tn", 
        "Turkey": "tr", "Ukraine": "ua", "United States": "us", "Uruguay": "uy", 
        "Uzbekistan": "uz"
    }

    def render_html_flag_label(team_name):
        iso = country_iso_map.get(team_name)
        if iso:
            # Injects an actual image element that works beautifully on any operating system
            return f'<img src="https://flagcdn.com/w20/{iso}.png" width="20" style="vertical-align: middle; margin-right: 8px;"> {team_name}'
        return f'🏳️ {team_name}'

    if "team" in display_df.columns:
        display_df["team"] = display_df["team"].apply(render_html_flag_label)

    # 4. Convert directly to a raw text-only HTML layout
    raw_html = display_df.to_html(index=False, escape=False)

    # 5. Inject premium dark grid container CSS
    st.markdown(
        """
        <style>
        .wc-table-container {
            width: 100%;
            overflow-x: auto;
            background-color: #111827 !important;
            border-radius: 12px;
            padding: 8px;
        }
        .wc-table-container table {
            width: 100% !important;
            border-collapse: collapse !important;
            background-color: #111827 !important;
        }
        .wc-table-container th {
            background-color: #1F2937 !important;
            color: #FFFFFF !important;
            font-weight: bold !important;
            padding: 12px 10px !important;
            text-align: left !important;
            border-bottom: 3px solid #374151 !important;
        }
        .wc-table-container td {
            background-color: #111827 !important;
            color: #FFFFFF !important;
            padding: 12px 10px !important;
            border-bottom: 1px solid #1F2937 !important;
            text-align: left !important;
            vertical-align: middle !important;
        }
        .wc-table-container tr:nth-child(even) td {
            background-color: #1F2937 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    wrapped_html = f'<div class="wc-table-container">{raw_html}</div>'
    st.write(wrapped_html, unsafe_allow_html=True)


def render_group_table(group_table):
    """Render a group table with top 2 highlighted."""
    table = group_table.copy()
    table["team"] = table["team"].apply(team_label)

    display_cols = [
        "group_rank",
        "team",
        "played",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "goal_difference",
        "points",
    ]

    def highlight_qualified(row):
        if row["group_rank"] <= 2:
            return ["background-color: rgba(48, 209, 88, 0.25); color: white;"] * len(row)
        if row["group_rank"] == 3:
            return ["background-color: rgba(255, 214, 10, 0.18); color: white;"] * len(row)
        return [""] * len(row)

    styled = table[display_cols].style.apply(highlight_qualified, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)


def page_live_simulation():
    set_page_background(PAGE_BACKGROUNDS["live_simulation"])
    show_header()

    st.subheader("🎮 Live tournament simulation")
    st.write("Click the button below to simulate one full World Cup path from groups to final.")

    run_button = st.button("▶️ Run one full tournament simulation", type="primary")

    if run_button:
        with st.spinner("Simulating tournament..."):
            tournament = simulate_tournament()

        summary = tournament["summary"]
        group_tables = tournament["group_tables"]
        group_matches = tournament["group_matches"]
        knockout_results = tournament["knockout_results"]

        st.markdown("## Group stage")

        for group_name in sorted(group_tables["group"].unique()):
            st.markdown(f"### Group {group_name}")

            group_matches_view = group_matches[
                group_matches["home_team"].isin(
                    group_tables[group_tables["group"] == group_name]["team"]
                )
            ].copy()

            for _, match in group_matches_view.iterrows():
                st.write(
                    f"{team_label(match['home_team'])} "
                    f"**{match['home_goals']} - {match['away_goals']}** "
                    f"{team_label(match['away_team'])}"
                )
                time.sleep(0.08)

            group_table = group_tables[group_tables["group"] == group_name]
            render_group_table(group_table)

        st.markdown("## Knockout stage")

        for round_name in ["R32", "R16", "QF", "SF", "Final"]:
            round_results = knockout_results[knockout_results["round"] == round_name]
            st.markdown(f'<div class="round-title">{round_name}</div>', unsafe_allow_html=True)

            cols = st.columns(2)
            for idx, (_, match) in enumerate(round_results.iterrows()):
                with cols[idx % 2]:
                    st.markdown(
                        f"""
                        <div class="team-card">
                            {team_label(match['home_team'])} 
                            <strong>{match['home_goals']} - {match['away_goals']}</strong> 
                            {team_label(match['away_team'])}
                            <br>
                            <span class="small-muted">
                            Winner: {team_label(match['winner'])}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.08)

        st.markdown(
            f"""
            <div class="winner-card">
                🏆 Winner: {team_label(summary["winner"])} 🏆
                <br>
                <span style="font-size:18px;">Runner-up: {team_label(summary["runner_up"])}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        show_trophies()


def page_match_explorer():
    set_page_background(PAGE_BACKGROUNDS["match_explorer"])
    show_header()

    st.subheader("🔎 Match explorer")
    teams = sorted(df_groups["nation"].unique())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Team A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)

    with col2:
        away_team = st.selectbox("Team B", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)

    if home_team == away_team:
        st.warning("Please select two different teams.")
        return

    pred = predict_match(home_team, away_team)

    st.markdown("### Model probabilities")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(f"{team_label(home_team)} xG", f"{pred['home_xg']:.2f}")
    with col2:
        st.metric(f"{team_label(away_team)} xG", f"{pred['away_xg']:.2f}")
    with col3:
        st.metric("Team A win", format_probability(pred["home_win_prob"]))
    with col4:
        st.metric("Draw", format_probability(pred["draw_prob"]))
    with col5:
        st.metric("Team B win", format_probability(pred["away_win_prob"]))

    st.markdown("### Most likely scorelines")
    scorelines = (
        pred["score_probs"]
        .sort_values("probability", ascending=False)
        .head(10)
        .copy()
    )

    scorelines["scoreline"] = (
        scorelines["home_goals"].astype(str)
        + " - "
        + scorelines["away_goals"].astype(str)
    )
    scorelines["probability"] = scorelines["probability"].map(lambda x: f"{x * 100:.1f}%")

    st.dataframe(
        scorelines[["scoreline", "probability"]],
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Team context")
    context_data = pd.DataFrame([
        {
            "team": team_label(home_team),
            "elo": team_to_elo.get(home_team),
            "fifa_rank": team_to_fifa_rank.get(home_team),
            "rank_change": team_to_fifa_rank_change.get(home_team),
            "form_score": team_to_form.get(home_team),
            "h2h_score_vs_opponent": get_h2h_score(home_team, away_team),
        },
        {
            "team": team_label(away_team),
            "elo": team_to_elo.get(away_team),
            "fifa_rank": team_to_fifa_rank.get(away_team),
            "rank_change": team_to_fifa_rank_change.get(away_team),
            "form_score": team_to_form.get(away_team),
            "h2h_score_vs_opponent": get_h2h_score(away_team, home_team),
        },
    ])

    st.dataframe(context_data, width="stretch", hide_index=True)


def main():
    page = st.sidebar.radio(
        "Choose page",
        [
            "🏆 Precomputed probabilities",
            "🎮 Live simulation",
            "🔎 Match explorer",
        ],
    )

    if page == "🏆 Precomputed probabilities":
        page_probabilities()
    elif page == "🎮 Live simulation":
        page_live_simulation()
    elif page == "🔎 Match explorer":
        page_match_explorer()


if __name__ == "__main__":
    main()