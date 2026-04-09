import io
import re
import sqlite3
from datetime import datetime, timedelta

import dateparser
from dateparser.search import search_dates
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


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

    date_text, due = max(found, key=lambda x: len(x[0]))
    task = re.sub(re.escape(date_text), "", text, flags=re.IGNORECASE).strip(" ,.-")
    if not task:
        task = text

    return task, due


# -------------------- Timetable parsing helpers --------------------
DAY_ALIASES = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}


def normalize_day(day_text: str):
    if not day_text:
        return None
    key = day_text.strip().lower()
    return DAY_ALIASES.get(key)


def parse_time_text(time_text: str):
    if not time_text:
        return None
    cleaned = time_text.strip().lower().replace(".", ":")
    if re.fullmatch(r"\d{1,2}", cleaned):
        cleaned += ":00"
    parsed = dateparser.parse(cleaned)
    if not parsed:
        return None
    return parsed.strftime("%H:%M")


def extract_text_from_upload(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        if pdfplumber is None:
            return "", "Install pdfplumber to parse PDF timetables."
        text_parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip(), None

    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        if Image is None or pytesseract is None:
            return "", "Install pillow and pytesseract for image OCR, or upload PDF/text."
        try:
            image = Image.open(io.BytesIO(data))
            text = pytesseract.image_to_string(image)
            return text.strip(), None
        except Exception:
            return "", "Image OCR failed. Upload a clearer file or edit schedule manually."

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore"), None

    return "", "Unsupported file type. Use PDF, image, or TXT."


def parse_timetable_text(raw_text: str):
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    entries = []
    if not lines:
        return entries

    pattern = re.compile(
        r"(?P<day>mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:rs|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)"
        r"[\s,:-]+(?P<subject>[A-Za-z][A-Za-z0-9 &/\-]{1,40})"
        r"[\s,:-]+(?P<start>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)"
        r"\s*(?:-|to|–|—)\s*"
        r"(?P<end>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
        flags=re.IGNORECASE,
    )

    for line in lines:
        match = pattern.search(line)
        if not match:
            continue
        day = normalize_day(match.group("day"))
        subject = match.group("subject").strip()
        start_time = parse_time_text(match.group("start"))
        end_time = parse_time_text(match.group("end"))
        if day and subject and start_time:
            entries.append(
                {
                    "day": day,
                    "subject": subject,
                    "start_time": start_time,
                    "end_time": end_time or "",
                }
            )

    return entries


def next_weekday_datetime(day_name: str, time_str: str):
    weekday_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    if day_name not in weekday_map or not time_str:
        return None

    now = datetime.now()
    target_weekday = weekday_map[day_name]
    hh, mm = [int(x) for x in time_str.split(":")]
    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    delta_days = (target_weekday - now.weekday()) % 7
    candidate = candidate + timedelta(days=delta_days)
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate


def find_matching_slots(task_text: str, timetable_rows):
    words = re.findall(r"[a-zA-Z]{3,}", (task_text or "").lower())
    if not words:
        return []
    matches = []
    for row in timetable_rows:
        subject = str(row.get("subject", "")).lower()
        if any(w in subject for w in words):
            matches.append(row)
    return matches


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
        done INTEGER DEFAULT 0,
        subject TEXT,
        remind_at TEXT
    )
"""
)
conn.commit()

cur.execute("PRAGMA table_info(reminders)")
cols = {row[1] for row in cur.fetchall()}
if "done" not in cols:
    cur.execute("ALTER TABLE reminders ADD COLUMN done INTEGER DEFAULT 0")
if "subject" not in cols:
    cur.execute("ALTER TABLE reminders ADD COLUMN subject TEXT")
if "remind_at" not in cols:
    cur.execute("ALTER TABLE reminders ADD COLUMN remind_at TEXT")
conn.commit()


def add_reminder(task, due_date, category, priority, subject=None, remind_at=None):
    due_str = due_date.strftime("%Y-%m-%d %H:%M") if due_date else None
    remind_str = remind_at.strftime("%Y-%m-%d %H:%M") if remind_at else None
    cur.execute(
        "INSERT INTO reminders (task,due_date,category,priority,subject,remind_at) VALUES (?,?,?,?,?,?)",
        (task, due_str, category, priority, subject, remind_str),
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
    .block-container { max-width: 900px; padding-top: 2rem; }
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
st.caption("Timetable-aware reminders · NLP input · Predictive priority")

if "timetable_rows" not in st.session_state:
    st.session_state.timetable_rows = []

tab0, tab1, tab2, tab3 = st.tabs(["Timetable", "Add", "View", "Manage"])


# -------------------- Tab 0: Timetable --------------------
with tab0:
    st.markdown("#### Upload timetable")
    up = st.file_uploader(
        "Upload PDF/Image/TXT timetable",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
    )

    if up:
        raw_text, error = extract_text_from_upload(up)
        if error:
            st.warning(error)
        if raw_text:
            rows = parse_timetable_text(raw_text)
            if rows:
                st.success(f"Detected {len(rows)} timetable slot(s). Review and edit below.")
                df = pd.DataFrame(rows)
                edited = st.data_editor(
                    df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="tt_editor",
                )
                st.session_state.timetable_rows = edited.fillna("").to_dict("records")
            else:
                st.info("Could not parse structured slots from text. Add rows manually below.")
                st.text_area("Extracted text preview", value=raw_text[:3000], height=180)

    if not st.session_state.timetable_rows:
        st.markdown("##### Add timetable rows manually")
        empty_df = pd.DataFrame(
            [
                {"day": "Monday", "subject": "Python", "start_time": "15:00", "end_time": "16:00"},
            ]
        )
        manual_df = st.data_editor(
            empty_df,
            use_container_width=True,
            num_rows="dynamic",
            key="tt_manual_editor",
        )
        if st.button("Save timetable rows", use_container_width=True):
            st.session_state.timetable_rows = manual_df.fillna("").to_dict("records")
            st.success("Timetable saved.")
            st.rerun()
    else:
        st.caption(f"Active timetable rows: {len(st.session_state.timetable_rows)}")


# -------------------- Tab 1: Add --------------------
with tab1:
    st.markdown("#### New reminder")
    text = st.text_input(
        "Describe your task",
        placeholder="e.g. python homework, submit by friday 5pm",
    )

    task, auto_date = "", None
    if text:
        task, auto_date = extract_reminder(text)
        if auto_date:
            st.success(f"Detected date: *{auto_date.strftime('%b %d, %Y - %H:%M')}*")
        else:
            st.info("No clear date in text. I'll try matching subject from your timetable.")

    matched_slot_dt = None
    matched_subject = None
    matched_label = None

    possible_slots = find_matching_slots(text, st.session_state.timetable_rows)
    if text and not auto_date and possible_slots:
        labels = [
            f"{r.get('subject', 'Unknown')} | {r.get('day', '')} {r.get('start_time', '')}"
            for r in possible_slots
        ]
        selected_label = st.selectbox(
            "I found matching class timings. Pick one:",
            labels,
        )
        chosen = possible_slots[labels.index(selected_label)]
        matched_subject = chosen.get("subject")
        matched_slot_dt = next_weekday_datetime(chosen.get("day", ""), chosen.get("start_time", ""))
        matched_label = selected_label
        if matched_slot_dt:
            st.success(f"Class slot selected: *{matched_slot_dt.strftime('%a %b %d, %Y - %H:%M')}*")
    elif text and not auto_date and st.session_state.timetable_rows:
        st.info("No subject match found in timetable. Pick manual date/time below.")

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

    remind_offset = st.selectbox(
        "Remind me before class/task by",
        [0, 5, 10, 15, 20, 30, 45, 60],
        index=4,
        format_func=lambda x: "At exact time" if x == 0 else f"{x} minutes",
    )

    if priority_mode == "I'll choose":
        priority = st.selectbox("Set priority", ["Critical", "Optional"])
    else:
        chosen_dt = auto_date or matched_slot_dt or datetime.combine(manual_date, manual_time)
        priority = predict_priority(chosen_dt, category)
        badge = "🔴 Critical" if priority == "Critical" else "🟢 Optional"
        st.markdown(f"AI priority: *{badge}*")

    if st.button("Add reminder", use_container_width=True):
        if not text.strip():
            st.warning("Please enter a task.")
        else:
            final_dt = auto_date or matched_slot_dt or datetime.combine(manual_date, manual_time)
            final_task = task if task else text
            final_subject = matched_subject
            remind_at = final_dt - timedelta(minutes=int(remind_offset)) if final_dt else None
            add_reminder(final_task, final_dt, category, priority, final_subject, remind_at)
            if matched_label:
                st.success(f"Reminder added for {matched_label}.")
            else:
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
            subj = r[6] if len(r) > 6 and r[6] else "General"
            remind_text = r[7] if len(r) > 7 and r[7] else "-"

            st.markdown(
                f"""
                <div class="reminder-card">
                    <strong>{done}{r[1]}</strong><br>
                    <span style="color:#888;font-size:13px">📅 {due_text} &nbsp;·&nbsp; {r[3]} &nbsp;·&nbsp; 📘 {subj}</span><br>
                    <span style="color:#666;font-size:12px">⏰ Remind at: {remind_text}</span>
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
            rid, task_name, due_date, category, priority, done, subject, remind_at = r
            due_text = due_date if due_date else "No due date"
            subject_text = subject if subject else "General"
            remind_text = remind_at if remind_at else "-"

            st.markdown(f"*{task_name}*  \n`{due_text}` · {category} · {priority} · {subject_text} · remind {remind_text}")

            col1, col2 = st.columns(2)
            with col1:
                new_done = st.checkbox("Done", value=bool(done), key=f"done_{rid}")
                if new_done != bool(done):
                    cur.execute("UPDATE reminders SET done=? WHERE id=?", (1 if new_done else 0, rid))
                    conn.commit()
                    st.rerun()

            with col2:
                if st.button("🗑️ Delete", key=f"del_{rid}", use_container_width=True):
                    cur.execute("DELETE FROM reminders WHERE id=?", (rid,))
                    conn.commit()
                    st.success("Deleted.")
                    st.rerun()

            st.divider()

