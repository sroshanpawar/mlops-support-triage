"""
Streamlit CRM Dashboard for the Smart Post-Purchase Support Triage System.
Professional, dark-themed interface with real-time analytics.
"""

import json
import time
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000"

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nexus Triage — CRM Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global Theme ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Reduce Top Padding ────────────────────────────────────────────── */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    /* ── Header ────────────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* ── Metric Cards ──────────────────────────────────────────────────── */
    .metric-card {
        background: rgba(30, 32, 48, 0.85); /* Slightly lighter dark background */
        padding: 1.5rem;
        border-radius: 12px; /* Rounded corners */
        border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle border */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.35);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0 0.2rem 0;
        letter-spacing: -1px;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: rgba(255, 255, 255, 0.5);
        font-weight: 500;
    }
    .color-blue { color: #4facfe; }
    .color-green { color: #43e97b; }
    .color-orange { color: #fa709a; }
    .color-purple { color: #a18cd1; }
    .color-red { color: #ff6b6b; }
    .color-yellow { color: #f9d423; }

    /* ── Status Badges ─────────────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-auto { background: rgba(67, 233, 123, 0.15); color: #43e97b; }
    .badge-escalated { background: rgba(250, 112, 154, 0.15); color: #fa709a; }
    .badge-discarded { background: rgba(161, 140, 209, 0.15); color: #a18cd1; }

    /* ── Section Headers ───────────────────────────────────────────────── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e0e0e0;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(79, 172, 254, 0.3);
    }

    /* ── Sidebar ───────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
    }
    
    /* ── Sidebar Navigation Tabs (st.radio styling) ────────────────────── */
    .stRadio > div[role="radiogroup"] {
        gap: 0.4rem;
        background: transparent;
    }
    .stRadio label {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 6px;
        padding: 0.6rem 1rem;
        transition: background 0.2s, color 0.2s;
        margin-bottom: 2px;
    }
    .stRadio label:hover {
        background: rgba(255, 255, 255, 0.1);
    }

    /* ── Hide default streamlit elements ───────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Tabs ───────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }

    /* ── Upload Area ───────────────────────────────────────────────────── */
    .upload-zone {
        background: rgba(30, 32, 48, 0.85); /* Match cards background */
        border: 2px dashed rgba(79, 172, 254, 0.3);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────

def api_call(method: str, endpoint: str, **kwargs):
    """Make an API call and handle errors."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = getattr(requests, method)(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API. Make sure FastAPI is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ API Error: {e.response.text}")
        return None

def render_action_badge(action: str) -> str:
    """Render a styled badge for the action type."""
    badge_class = {
        "Auto-Reply": "badge-auto",
        "Escalated": "badge-escalated",
        "Discarded": "badge-discarded",
    }.get(action, "badge-auto")
    return f'<span class="badge {badge_class}">{action}</span>'

def format_confidence(score: float) -> str:
    """Format confidence score with color coding."""
    pct = score * 100
    if pct >= 85:
        color = "#43e97b"
    elif pct >= 60:
        color = "#f9d423"
    else:
        color = "#ff6b6b"
    return f'<span style="color: {color}; font-weight: 600;">{pct:.1f}%</span>'


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Nexus Triage")
    st.markdown("---")

    # System status
    health = api_call("get", "/health")
    if health:
        st.markdown("### System Status")
        st.markdown(
            f"**API:** {'🟢 Online' if health['status'] == 'healthy' else '🔴 Offline'}"
        )
        st.markdown(
            f"**Model:** {'🟢 Loaded' if health['model_loaded'] else '🔴 Not Loaded'}"
        )
        st.markdown(
            f"**Database:** {'🟢 Connected' if health['database_connected'] else '🔴 Error'}"
        )
        st.markdown(f"**Version:** `{health['version']}`")
    else:
        st.markdown("### System Status")
        st.markdown("**API:** 🔴 Offline")
        st.markdown("_Start the backend:_")
        st.code("uvicorn backend.main:app --reload", language="bash")

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["📊 Dashboard", "📤 Upload & Process", "💬 Message Explorer", "🔍 Live Classifier"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<p style="color: rgba(255,255,255,0.3); font-size: 0.75rem; text-align: center;">'
        'Nexus Triage CRM v2.0<br>MLOps-Driven Architecture</p>',
        unsafe_allow_html=True,
    )


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    "<h1>⚡ Nexus Post-Purchase Support Triage</h1>"
    "<p>AI-powered customer message classification, confidence-based escalation & automated response system</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":

    stats = api_call("get", "/stats")

    if stats:
        # ── KPI Metric Cards ─────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Total Messages</div>'
                f'<div class="metric-value color-blue">{stats["total_messages"]:,}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Auto Replied</div>'
                f'<div class="metric-value color-green">{stats["auto_replied"]:,}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Escalated</div>'
                f'<div class="metric-value color-orange">{stats["escalated"]:,}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Discarded</div>'
                f'<div class="metric-value color-purple">{stats["discarded"]:,}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Avg Confidence</div>'
                f'<div class="metric-value color-yellow">{stats["avg_confidence"]*100:.1f}%</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with c6:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Escalation Rate</div>'
                f'<div class="metric-value color-red">{stats["escalation_rate"]:.1f}%</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ───────────────────────────────────────────────────────
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown('<div class="section-header">📊 Intent Distribution</div>', unsafe_allow_html=True)
            if stats["intent_distribution"]:
                intent_df = pd.DataFrame(
                    list(stats["intent_distribution"].items()),
                    columns=["Intent", "Count"],
                )
                fig = px.pie(
                    intent_df,
                    values="Count",
                    names="Intent",
                    color_discrete_sequence=["#4facfe", "#43e97b", "#fa709a", "#a18cd1", "#f9d423"],
                    hole=0.45,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0e0", family="Inter"),
                    legend=dict(font=dict(size=11)),
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=350,
                )
                fig.update_traces(textinfo="percent+label", textfont_size=11)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data yet. Upload and process messages to see distribution.")

        with chart_col2:
            st.markdown('<div class="section-header">⚡ Action Breakdown</div>', unsafe_allow_html=True)
            action_data = {
                "Action": ["Auto-Reply", "Escalated", "Discarded"],
                "Count": [stats["auto_replied"], stats["escalated"], stats["discarded"]],
            }
            action_df = pd.DataFrame(action_data)
            fig = px.bar(
                action_df,
                x="Action",
                y="Count",
                color="Action",
                color_discrete_map={
                    "Auto-Reply": "#43e97b",
                    "Escalated": "#fa709a",
                    "Discarded": "#a18cd1",
                },
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0", family="Inter"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                height=350,
            )
            fig.update_traces(marker_line_width=0, opacity=0.9)
            st.plotly_chart(fig, use_container_width=True)

        # ── Confidence distribution from recent messages ─────────────────
        st.markdown('<div class="section-header">📈 Confidence Score Distribution (Recent Messages)</div>', unsafe_allow_html=True)
        messages_data = api_call("get", "/messages", params={"limit": 200})
        if messages_data and messages_data["messages"]:
            conf_df = pd.DataFrame(messages_data["messages"])
            fig = px.histogram(
                conf_df,
                x="confidence_score",
                nbins=20,
                color="predicted_intent",
                color_discrete_sequence=["#4facfe", "#43e97b", "#fa709a", "#a18cd1", "#f9d423"],
                labels={"confidence_score": "Confidence Score", "predicted_intent": "Intent"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0", family="Inter"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Count"),
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                bargap=0.05,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No messages processed yet.")

    else:
        st.warning("Unable to load dashboard statistics. Check API connection.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Upload & Process
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📤 Upload & Process":

    st.markdown('<div class="section-header">📤 Upload Simulated Traffic File</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="upload-zone">'
        '<p style="color: #4facfe; font-size: 1.1rem; margin: 0 0 0.3rem 0;">📁 Drop your simulated_traffic.json file</p>'
        '<p style="color: rgba(255,255,255,0.4); font-size: 0.85rem; margin: 0;">JSON array of message objects with "text" field required</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type=["json"],
        help="Upload a simulated_traffic.json file containing customer messages.",
    )

    if uploaded_file is not None:
        try:
            content = json.loads(uploaded_file.read())
            st.success(f"✅ File loaded: **{uploaded_file.name}** — {len(content)} messages found")

            # Preview
            with st.expander("👁️ Preview uploaded messages", expanded=False):
                preview_df = pd.DataFrame(content)
                st.dataframe(preview_df, use_container_width=True, height=300)

            # Process button
            col_btn, col_spacer = st.columns([1, 3])
            with col_btn:
                process_btn = st.button("🚀 Process All Messages", type="primary", use_container_width=True)

            if process_btn:
                with st.spinner("Processing messages through the AI classifier..."):
                    progress = st.progress(0)
                    status_text = st.empty()

                    # Send batch to API
                    messages = []
                    for m in content:
                        messages.append({
                            "id": m.get("id"),
                            "customer_name": m.get("customer_name"),
                            "email": m.get("email"),
                            "text": m["text"],
                            "timestamp": m.get("timestamp"),
                            "channel": m.get("channel"),
                        })

                    # Simulate batch progress
                    batch_size = max(1, len(messages) // 10)
                    all_results = []
                    total_auto = 0
                    total_esc = 0
                    total_disc = 0

                    for i in range(0, len(messages), batch_size):
                        batch = messages[i:i + batch_size]
                        result = api_call("post", "/batch-predict", json={"messages": batch})
                        if result:
                            all_results.extend(result["results"])
                            total_auto += result["auto_replied"]
                            total_esc += result["escalated"]
                            total_disc += result["discarded"]

                        pct = min(100, int((i + batch_size) / len(messages) * 100))
                        progress.progress(pct)
                        status_text.text(f"Processing batch {i // batch_size + 1}... ({pct}%)")
                        time.sleep(0.1)

                    progress.progress(100)
                    status_text.text("✅ All messages processed!")

                    # ── Results Summary ──────────────────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">📊 Processing Results</div>', unsafe_allow_html=True)

                    rc1, rc2, rc3, rc4 = st.columns(4)
                    with rc1:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-label">Total Processed</div>'
                            f'<div class="metric-value color-blue">{len(all_results)}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with rc2:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-label">Auto Replied</div>'
                            f'<div class="metric-value color-green">{total_auto}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with rc3:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-label">Escalated</div>'
                            f'<div class="metric-value color-orange">{total_esc}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with rc4:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-label">Discarded</div>'
                            f'<div class="metric-value color-purple">{total_disc}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Results table
                    if all_results:
                        st.markdown("<br>", unsafe_allow_html=True)
                        results_df = pd.DataFrame(all_results)
                        results_df["confidence_pct"] = (results_df["confidence_score"] * 100).round(1).astype(str) + "%"

                        df_display = results_df[["customer_name", "original_text", "predicted_intent",
                                       "confidence_pct", "action_taken"]].rename(columns={
                                "customer_name": "Customer",
                                "original_text": "Message",
                                "predicted_intent": "Intent",
                                "confidence_pct": "Confidence %",
                                "action_taken": "Action",
                            })
                        
                        def highlight_rows(row):
                            if row['Action'] == 'Escalated':
                                return ['background-color: rgba(250, 112, 154, 0.2)'] * len(row)
                            elif row['Action'] == 'Auto-Reply':
                                return ['background-color: rgba(67, 233, 123, 0.2)'] * len(row)
                            return [''] * len(row)
                            
                        styled_df = df_display.style.apply(highlight_rows, axis=1)

                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            height=400,
                        )

        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file. Please check the file format.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Message Explorer
# ═════════════════════════════════════════════════════════════════════════════

elif page == "💬 Message Explorer":

    st.markdown('<div class="section-header">💬 Processed Messages</div>', unsafe_allow_html=True)

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_intent = st.selectbox(
            "Filter by Intent",
            ["All", "Shipping_Inquiry", "Refund_Request", "Product_Dispute", "Price_Inquiry", "Spam"],
        )
    with fc2:
        filter_action = st.selectbox("Filter by Action", ["All", "Auto-Reply", "Escalated", "Discarded"])
    with fc3:
        limit = st.slider("Messages to show", 10, 500, 100)

    # Build query params
    params = {"limit": limit}
    if filter_intent != "All":
        params["intent"] = filter_intent
    if filter_action != "All":
        params["action"] = filter_action

    data = api_call("get", "/messages", params=params)

    if data and data["messages"]:
        st.caption(f"Showing {len(data['messages'])} of {data['total']} messages")

        df_msgs = pd.DataFrame(data["messages"])
        df_msgs["confidence_pct"] = (df_msgs["confidence_score"] * 100).round(1).astype(str) + "%"
        
        df_display = df_msgs[["customer_name", "original_text", "predicted_intent", 
                              "confidence_pct", "action_taken", "channel"]].rename(columns={
            "customer_name": "Customer",
            "original_text": "Message",
            "predicted_intent": "Intent",
            "confidence_pct": "Confidence %",
            "action_taken": "Action",
            "channel": "Channel"
        })
        
        def highlight_message_rows(row):
            if row['Action'] == 'Escalated':
                return ['background-color: rgba(250, 112, 154, 0.2)'] * len(row)
            elif row['Action'] == 'Auto-Reply':
                return ['background-color: rgba(67, 233, 123, 0.2)'] * len(row)
            return [''] * len(row)
            
        styled_msgs = df_display.style.apply(highlight_message_rows, axis=1)
        st.dataframe(styled_msgs, use_container_width=True, height=600)

    elif data:
        st.info("No messages found matching the selected filters.")
    else:
        st.warning("Unable to load messages. Check API connection.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Live Classifier
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Live Classifier":

    st.markdown('<div class="section-header">🔍 Live Message Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: rgba(255,255,255,0.5); font-size: 0.9rem;">'
        "Type or paste a customer message below to see real-time classification results.</p>",
        unsafe_allow_html=True,
    )

    test_message = st.text_area(
        "Customer Message",
        placeholder="e.g., Where is my order? It's been 2 weeks and I still haven't received it.",
        height=120,
    )

    col_classify, col_clear = st.columns([1, 3])
    with col_classify:
        classify_btn = st.button("🧠 Classify", type="primary", use_container_width=True)

    if classify_btn and test_message:
        with st.spinner("Classifying..."):
            result = api_call(
                "post",
                "/predict",
                json={
                    "text": test_message,
                    "customer_name": "Live Test",
                    "channel": "dashboard",
                },
            )

        if result:
            st.markdown("<br>", unsafe_allow_html=True)

            # Result cards
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Predicted Intent</div>'
                    f'<div class="metric-value color-blue" style="font-size: 1.4rem;">'
                    f'{result["predicted_intent"].replace("_", " ")}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with r2:
                conf_pct = result["confidence_score"] * 100
                conf_color = "color-green" if conf_pct >= 85 else ("color-yellow" if conf_pct >= 60 else "color-red")
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Confidence</div>'
                    f'<div class="metric-value {conf_color}">{conf_pct:.1f}%</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with r3:
                action_emoji = {"Auto-Reply": "✅", "Escalated": "🚨", "Discarded": "🗑️"}.get(
                    result["action_taken"], "❓"
                )
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">Action Taken</div>'
                    f'<div class="metric-value color-purple" style="font-size: 1.4rem;">'
                    f'{action_emoji} {result["action_taken"]}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Auto-reply preview
            if result.get("auto_reply_text"):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">📝 Generated Response</div>', unsafe_allow_html=True)
                st.info(result["auto_reply_text"])

    elif classify_btn:
        st.warning("Please enter a message to classify.")
