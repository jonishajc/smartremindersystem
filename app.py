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


# -------------------- Date extraction (FIXED) --------------------
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

    # ✅ Pick FIRST detected date (more accurate than longest match)
    date_text, due = found[0]

    # ❗ Ensure time isn't default midnight if user mentioned time
    if due and due.hour == 0 and due.minute == 0:
        time_match = re.search(r'(\d{1,2}(:\d{2})?\s?(am|pm))', text, re.IGNORECASE)
        if time_match:
            corrected = dateparser.parse(time_match.group(), settings=settings)
            if corrected:
                due = due.replace(hour=corrected.hour, minute=corrected.minute)

    # Clean task text
    task = re.sub(re.escape(date_text), "", text, flags=re.IGNORECASE)
    task = re.sub(r'\b(remind me to|remind me|to)\b', '', task, flags=re.IGNORECASE)
    task = task.strip(" ,.-")

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

st.title("🗓️ Smart Reminder System")
st.caption("AI-powered · NLP input · Predictive priority")

tab1, tab2, tab3 = st.tabs(["Add", "View", "Manage"])


# -------------------- Add --------------------
with tab1:
    text = st.text_input(
        "Describe your task",
        placeholder="e.g. submit lab report by friday 5pm",
    )

    task, auto_date = "", None
    if text:
        task, auto_date = extract_reminder(text)
        if auto_date:
            st.success(f"Detected: {auto_date.strftime('%b %d, %Y - %H:%M')}")
        else:
            st.info("No date detected. Choose manually.")

    col1, col2 = st.columns(2)
    with col1:
        manual_date = st.date_input("Date", value=datetime.today())
        manual_time = st.time_input("Time", value=datetime.now().time())
    with col2:
        category = st.selectbox("Category", ["assignment", "work", "event", "other"])
        priority_mode = st.radio("Priority", ["AI decides", "I'll choose"])

    if priority_mode == "I'll choose":
        priority = st.selectbox("Set priority", ["Critical", "Optional"])
    else:
        final_dt = auto_date if auto_date else datetime.combine(manual_date, manual_time)
        priority = predict_priority(final_dt, category)
        st.write(f"AI Priority: {priority}")

    if st.button("Add reminder"):
        final_dt = auto_date if auto_date else datetime.combine(manual_date, manual_time)
        final_task = task if task else text
        add_reminder(final_task, final_dt, category, priority)
        st.success("Reminder added!")
        st.rerun()


# -------------------- View --------------------
with tab2:
    data = get_reminders()
    for r in data:
        st.write(f"{r[1]} | {r[2]} | {r[3]} | {r[4]}")


# -------------------- Manage --------------------
with tab3:
    data = get_reminders()
    for r in data:
        rid, task, due_date, category, priority, done = r
        st.write(f"{task} | {due_date}")

        if st.checkbox("Done", value=bool(done), key=f"d{rid}"):
            cur.execute("UPDATE reminders SET done=1 WHERE id=?", (rid,))
            conn.commit()

        if st.button("Delete", key=f"x{rid}"):
            cur.execute("DELETE FROM reminders WHERE id=?", (rid,))
            conn.commit()
            st.rerun()

