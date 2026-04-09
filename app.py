import re
import sqlite3
from datetime import datetime

import dateparser
from dateparser.search import search_dates
import streamlit as st
from sklearn.tree import DecisionTreeClassifier


# -------------------- ML priority model --------------------
CATEGORIES = {"assignment": 0, "work": 1, "event": 2, "other": 3}
X = [[2, 0], [6, 0], [12, 0], [48, 1], [72, 1], [2, 2], [80, 2], [3, 3], [100, 3]]
y = [1, 1, 1, 0, 0, 1, 0, 1, 0]

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


def predict_priority(due_date, category):
    if not due_date:
        return "Optional"
    hours = (due_date - datetime.now()).total_seconds() / 3600
    cat_id = CATEGORIES.get(category, 3)
    return "Critical" if clf.predict([[hours, cat_id]])[0] == 1 else "Optional"


# -------------------- Date extraction --------------------
def extract_reminder(text: str):
    text = (text or "").strip()
    if not text:
        return "", None

    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": datetime.now(),
        "RETURN_AS_TIMEZONE_AWARE": False,
    }

    found = search_dates(text, settings=settings)

    if not found:
        due = dateparser.parse(text, settings=settings)
        return text, due

    # Use the longest matched date phrase
    date_text, due = max(found, key=lambda x: len(x[0]))
    task = re.sub(re.escape(date_text), "", text, flags=re.IGNORECASE).strip(" ,.-")
    if not task:
        task = text

    return task, due


# -------------------- Database --------------------
conn = sqlite3.connect("reminders.db", check_same_thread=False)
cur = conn.cursor()

cur.execute(
    """
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task TEXT,
        due_date TEXT,
        category TEXT,
        priority TEXT,
        done INTEGER DEFAULT 0
    )
"""
)
conn.commit()

# Migration safety for old DB schema
cur.execute("PRAGMA table_info(reminders)")
cols = {row[1] for row in cur.fetchall()}
if "done" not in cols:
    cur.execute("ALTER TABLE reminders ADD COLUMN done INTEGER DEFAULT 0")
    conn.commit()


def add_reminder(task, due_date, category, priority):
    due_str = due_date.strftime("%Y-%m-%d %H:%M") if due_date else None
    cur.execute(
        "INSERT INTO reminders (task,due_date,category,priority) VALUES (?,?,?,?)",
        (task, due_str, category, priority),
    )
    conn.commit()


def get_reminders():
    cur.execute(
        """
        SELECT * FROM reminders
        ORDER BY
            CASE WHEN due_date IS NULL THEN 1 ELSE 0 END,
            due_date
    """
    )
    return cur.fetchall()


# -------------------- UI --------------------
st.set_page_config(page_title="Smart Reminders", page_icon="🗓️", layout="centered")

st.markdown(
    """
<style>
    .block-container { max-width: 720px; padding-top: 2rem; }
    h1 { font-weight: 500; letter-spacing: -0.5px; }
    .reminder-card {
        background: #f9f9f7;
        border: 1px solid #e8e8e4;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .tag {
        display: inline-block;
        font-size: 11px;
        padding: 2px 9px;
        border-radius: 99px;
        font-weight: 500;
        margin-right: 6px;
    }
    .critical { background: #fde8e8; color: #991b1b; }
    .optional { background: #e8f4e8; color: #166534; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🗓️ Smart Reminder System")
st.caption("AI-powered · NLP input · Predictive priority")

tab1, tab2, tab3 = st.tabs(["Add", "View", "Manage"])


# -------------------- Tab 1: Add --------------------
with tab1:
    st.markdown("#### New reminder")
    text = st.text_input(
        "Describe your task",
        placeholder="e.g. submit lab report by friday 5pm",
    )

    task, auto_date = "", None
    if text:
        task, auto_date = extract_reminder(text)
        if auto_date:
            st.success(f"Detected date: *{auto_date.strftime('%b %d, %Y - %H:%M')}*")
        else:
            st.info("No date detected in text - pick one below.")

    col1, col2 = st.columns(2)
    with col1:
        manual_date = st.date_input("Date", value=datetime.today())
        manual_time = st.time_input(
            "Time",
            value=datetime.now().replace(second=0, microsecond=0).time(),
        )
    with col2:
        category = st.selectbox("Category", ["assignment", "work", "event", "other"])
        priority_mode = st.radio("Priority", ["AI decides", "I'll choose"])

    if priority_mode == "I'll choose":
        priority = st.selectbox("Set priority", ["Critical", "Optional"])
    else:
        final_dt = auto_date if auto_date else datetime.combine(manual_date, manual_time)
        priority = predict_priority(final_dt, category)
        badge = "🔴 Critical" if priority == "Critical" else "🟢 Optional"
        st.markdown(f"AI priority: *{badge}*")

    if st.button("Add reminder", use_container_width=True):
        if not text.strip():
            st.warning("Please enter a task.")
        else:
            final_dt = auto_date if auto_date else datetime.combine(manual_date, manual_time)
            final_task = task if task else text
            add_reminder(final_task, final_dt, category, priority)
            st.success("Reminder added!")
            st.rerun()


# -------------------- Tab 2: View --------------------
with tab2:
    data = get_reminders()
    if not data:
        st.info("No reminders yet. Add one in the Add tab.")
    else:
        for r in data:
            icon = "🔴" if r[4] == "Critical" else "🟢"
            done = "✅ " if int(r[5] or 0) else ""
            tag_class = "critical" if r[4] == "Critical" else "optional"
            due_text = r[2] if r[2] else "No due date"

            st.markdown(
                f"""
                <div class="reminder-card">
                    <strong>{done}{r[1]}</strong><br>
                    <span style="color:#888;font-size:13px">📅 {due_text} &nbsp;·&nbsp; {r[3]}</span>
                    &nbsp;<span class="tag {tag_class}">{icon} {r[4]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------- Tab 3: Manage --------------------
with tab3:
    data = get_reminders()
    if not data:
        st.info("Nothing to manage yet.")
    else:
        for r in data:
            rid, task, due_date, category, priority, done = r
            due_text = due_date if due_date else "No due date"

            st.markdown(f"*{task}*  \n`{due_text}` · {category} · {priority}")

            col1, col2 = st.columns(2)
            with col1:
                new_done = st.checkbox("Done", value=bool(done), key=f"done_{rid}")
                if new_done != bool(done):
                    cur.execute(
                        "UPDATE reminders SET done=? WHERE id=?",
                        (1 if new_done else 0, rid),
                    )
                    conn.commit()
                    st.rerun()

            with col2:
                if st.button("🗑️ Delete", key=f"del_{rid}", use_container_width=True):
                    cur.execute("DELETE FROM reminders WHERE id=?", (rid,))
                    conn.commit()
                    st.success("Deleted.")
                    st.rerun()

            st.divider()
