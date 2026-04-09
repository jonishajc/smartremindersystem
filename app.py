import re
import sqlite3
import base64
import json
from datetime import datetime, timedelta

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
    """
    Returns (task_text, due_datetime_or_None).
    FIX: dateparser is called with PREFER_DAY_OF_MONTH and explicit time
    handling so '6 pm' is never dropped.
    """
    text = (text or "").strip()
    if not text:
        return "", None

    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": datetime.now(),
        "RETURN_AS_TIMEZONE_AWARE": False,
        "PREFER_DAY_OF_MONTH": "first",
    }

    found = search_dates(text, settings=settings)

    if not found:
        due = dateparser.parse(text, settings=settings)
        return text, due

    # Use the longest matched date phrase so "tomorrow 6 pm" beats "tomorrow"
    date_text, due = max(found, key=lambda x: len(x[0]))

    # --- TIME FIX ---
    # If the matched phrase didn't include an explicit time but a time token
    # exists elsewhere in the original text, re-parse the full text.
    time_pattern = re.compile(
        r'\b(\d{1,2}(:\d{2})?\s*(am|pm)|'
        r'\d{1,2}(:\d{2})?\s*(AM|PM)|'
        r'midnight|noon|evening|morning|afternoon)\b',
        re.IGNORECASE,
    )
    if time_pattern.search(text) and not time_pattern.search(date_text):
        full_parse = dateparser.parse(text, settings=settings)
        if full_parse:
            due = full_parse

    task = re.sub(re.escape(date_text), "", text, flags=re.IGNORECASE).strip(" ,.-")
    if not task:
        task = text

    return task, due


# -------------------- Timetable helpers --------------------
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def parse_timetable_text(raw: str) -> list[dict]:
    """
    Parse a free-text timetable into structured slots.
    Each slot: {day, subject, start_time, end_time}
    Supports formats like:
      Monday 9:00-10:00 Python
      Tue 14:00 Math (1hr)
      Wednesday: Physics 11am-12pm
    """
    slots = []
    lines = raw.strip().splitlines()

    time_range_re = re.compile(
        r'(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*[-–to]+\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)',
        re.IGNORECASE,
    )
    single_time_re = re.compile(
        r'(\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))',
        re.IGNORECASE,
    )
    duration_re = re.compile(r'\((\d+(?:\.\d+)?)\s*hr?s?\)', re.IGNORECASE)

    day_aliases = {
        "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
        "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        day = None
        for d in DAYS:
            if d.lower() in line.lower():
                day = d
                break
        if not day:
            for alias, full in day_aliases.items():
                if re.search(r'\b' + alias + r'\b', line, re.IGNORECASE):
                    day = full
                    break
        if not day:
            continue

        start_dt = end_dt = None
        range_match = time_range_re.search(line)
        if range_match:
            start_dt = dateparser.parse(range_match.group(1))
            end_dt = dateparser.parse(range_match.group(2))
        else:
            single = single_time_re.search(line)
            if single:
                start_dt = dateparser.parse(single.group(1))
                dur = duration_re.search(line)
                if dur and start_dt:
                    end_dt = start_dt + timedelta(hours=float(dur.group(1)))
                elif start_dt:
                    end_dt = start_dt + timedelta(hours=1)

        if not start_dt:
            continue

        # Strip day + time tokens to get subject name
        subject = line
        subject = re.sub(r'\b' + day[:3] + r'\w*\b', '', subject, flags=re.IGNORECASE)
        for d in DAYS:
            subject = re.sub(r'\b' + d + r'\b', '', subject, flags=re.IGNORECASE)
        if range_match:
            subject = subject.replace(range_match.group(0), '')
        elif single_time_re.search(subject):
            subject = single_time_re.sub('', subject)
        subject = duration_re.sub('', subject)
        subject = re.sub(r'[:\-–|]', ' ', subject).strip()
        subject = re.sub(r'\s+', ' ', subject).strip()
        if not subject:
            subject = "Class"

        slots.append({
            "day": day,
            "subject": subject,
            "start_time": start_dt.strftime("%H:%M"),
            "end_time": end_dt.strftime("%H:%M") if end_dt else None,
        })

    return slots


def find_matching_class(task_text: str, timetable: list[dict], target_date: datetime | None):
    """
    Given a task description and an optional target date, find the best
    matching timetable slot.
    Returns a slot dict or None.
    """
    if not timetable or not task_text:
        return None

    task_lower = task_text.lower()
    candidates = []

    target_day = None
    if target_date:
        target_day = DAYS[target_date.weekday()]

    for slot in timetable:
        score = 0
        subj_lower = slot["subject"].lower()

        # Keyword overlap
        for word in re.findall(r'\w+', subj_lower):
            if len(word) > 3 and word in task_lower:
                score += 2

        # Day match
        if target_day and slot["day"].lower() == target_day.lower():
            score += 3

        if score > 0:
            candidates.append((score, slot))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def get_next_occurrence(slot: dict) -> datetime:
    """Return the next datetime this slot occurs."""
    today = datetime.now()
    day_idx = DAYS.index(slot["day"])
    current_day_idx = today.weekday()
    days_ahead = (day_idx - current_day_idx) % 7
    if days_ahead == 0:
        # check if time already passed today
        h, m = map(int, slot["start_time"].split(":"))
        if today.hour > h or (today.hour == h and today.minute >= m):
            days_ahead = 7
    target_date = today + timedelta(days=days_ahead)
    h, m = map(int, slot["start_time"].split(":"))
    return target_date.replace(hour=h, minute=m, second=0, microsecond=0)


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


# -------------------- Anthropic API helper --------------------
def call_claude(prompt: str, system: str = "", image_b64: str = None, image_mime: str = "image/jpeg") -> str:
    """Call Claude API. Optionally pass an image as base64."""
    import urllib.request
    import json as _json

    messages_content = []
    if image_b64:
        messages_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": image_mime, "data": image_b64},
        })
    messages_content.append({"type": "text", "text": prompt})

    payload = _json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system": system if system else "You are a helpful assistant.",
        "messages": [{"role": "user", "content": messages_content}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = _json.loads(resp.read())
    return data["content"][0]["text"]


def parse_timetable_with_ai(raw_text: str = None, image_b64: str = None, image_mime: str = "image/jpeg") -> list[dict]:
    """Use Claude to extract structured timetable from text or image."""
    system = (
        "You are a timetable parser. Extract class schedule from the input and return ONLY valid JSON. "
        "Output a JSON array of objects, each with keys: day (full weekday name), subject (string), "
        "start_time (HH:MM 24hr), end_time (HH:MM 24hr or null). "
        "Example: [{\"day\": \"Monday\", \"subject\": \"Python Programming\", \"start_time\": \"09:00\", \"end_time\": \"10:00\"}]"
        "Return ONLY the JSON array, no markdown, no explanation."
    )
    prompt = raw_text or "Extract the timetable from the image above."
    try:
        response = call_claude(prompt, system=system, image_b64=image_b64, image_mime=image_mime)
        response = response.strip()
        # Strip markdown fences if present
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'^```\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        return json.loads(response)
    except Exception as e:
        st.warning(f"AI timetable parsing failed: {e}. Trying rule-based parser...")
        if raw_text:
            return parse_timetable_text(raw_text)
        return []


# -------------------- Session state --------------------
if "timetable" not in st.session_state:
    st.session_state.timetable = []
if "pending_reminder" not in st.session_state:
    st.session_state.pending_reminder = None  # holds (task, due_dt, category, matched_slot)


# -------------------- UI --------------------
st.set_page_config(page_title="Smart Reminders", page_icon="🗓️", layout="centered")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 740px;
        padding-top: 1.5rem;
    }

    h1 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        font-size: 2rem;
    }

    .reminder-card {
        background: #fafaf8;
        border: 1.5px solid #ebebе8;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: box-shadow 0.2s;
    }

    .reminder-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.06); }

    .tag {
        display: inline-block;
        font-size: 11px;
        padding: 2px 10px;
        border-radius: 99px;
        font-weight: 600;
        margin-right: 6px;
        letter-spacing: 0.3px;
    }

    .critical { background: #fde8e8; color: #991b1b; }
    .optional { background: #e8f4e8; color: #166534; }

    .class-match-box {
        background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
        border: 1.5px solid #bfdbfe;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 10px 0;
    }

    .tt-slot {
        background: #f5f3ff;
        border: 1px solid #ddd6fe;
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 6px;
        font-size: 13.5px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 13px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🗓️ Smart Reminder System")
st.caption("NLP input · AI timetable parsing · Predictive priority")

tab1, tab2, tab3, tab4 = st.tabs(["Add", "View", "Manage", "Timetable"])


# -------------------- Tab 4: Timetable --------------------
with tab4:
    st.markdown("#### Your Timetable")
    st.markdown(
        "Paste your class schedule as text **or** upload a photo. "
        "Once saved, the reminder system will detect matching classes automatically."
    )

    input_mode = st.radio("Input method", ["Text", "Image upload"], horizontal=True)

    if input_mode == "Text":
        sample = (
            "Monday 9:00-10:00 Python Programming\n"
            "Tuesday 11:00-12:30 Data Structures\n"
            "Wednesday 14:00-15:00 Machine Learning\n"
            "Friday 10:00-11:00 Math"
        )
        tt_text = st.text_area(
            "Paste schedule here (one class per line)",
            placeholder=sample,
            height=180,
        )
        use_ai = st.checkbox("Use AI parser (more accurate for complex formats)", value=True)

        if st.button("Parse & Save Timetable", use_container_width=True):
            if tt_text.strip():
                with st.spinner("Parsing timetable..."):
                    if use_ai:
                        slots = parse_timetable_with_ai(raw_text=tt_text)
                    else:
                        slots = parse_timetable_text(tt_text)

                if slots:
                    st.session_state.timetable = slots
                    st.success(f"Saved {len(slots)} class slots!")
                else:
                    st.error("Couldn't parse any slots. Try a clearer format.")
            else:
                st.warning("Please enter your schedule.")

    else:
        uploaded = st.file_uploader(
            "Upload timetable image (photo, screenshot, PDF page)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded:
            st.image(uploaded, caption="Uploaded timetable", use_container_width=True)
            if st.button("Parse Timetable from Image", use_container_width=True):
                with st.spinner("Sending to AI for analysis..."):
                    raw_bytes = uploaded.read()
                    b64 = base64.b64encode(raw_bytes).decode()
                    mime = uploaded.type or "image/jpeg"
                    slots = parse_timetable_with_ai(image_b64=b64, image_mime=mime)

                if slots:
                    st.session_state.timetable = slots
                    st.success(f"Extracted {len(slots)} class slots from image!")
                else:
                    st.error("Couldn't extract schedule. Try a clearer image or use text input.")

    if st.session_state.timetable:
        st.markdown("---")
        st.markdown(f"**{len(st.session_state.timetable)} slots loaded:**")
        for slot in st.session_state.timetable:
            end_str = f" – {slot['end_time']}" if slot.get("end_time") else ""
            st.markdown(
                f'<div class="tt-slot">📚 <strong>{slot["subject"]}</strong> · '
                f'{slot["day"]} {slot["start_time"]}{end_str}</div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear timetable", type="secondary"):
            st.session_state.timetable = []
            st.rerun()


# -------------------- Tab 1: Add --------------------
with tab1:
    st.markdown("#### New reminder")

    # --- Handle pending timetable-based reminder ---
    if st.session_state.pending_reminder:
        p = st.session_state.pending_reminder
        task, base_dt, category, matched_slot = p

        st.markdown(
            f'<div class="class-match-box">'
            f'📚 <strong>Class found:</strong> {matched_slot["subject"]} on '
            f'{matched_slot["day"]} at {matched_slot["start_time"]}'
            f'{"–" + matched_slot["end_time"] if matched_slot.get("end_time") else ""}'
            f'<br><span style="color:#555;font-size:13px">When should I remind you?</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        remind_offset = st.selectbox(
            "Remind me…",
            ["10 mins before", "30 mins before", "1 hour before", "2 hours before",
             "At class time", "Custom time"],
        )

        class_dt = get_next_occurrence(matched_slot)

        if remind_offset == "10 mins before":
            final_reminder_dt = class_dt - timedelta(minutes=10)
        elif remind_offset == "30 mins before":
            final_reminder_dt = class_dt - timedelta(minutes=30)
        elif remind_offset == "1 hour before":
            final_reminder_dt = class_dt - timedelta(hours=1)
        elif remind_offset == "2 hours before":
            final_reminder_dt = class_dt - timedelta(hours=2)
        elif remind_offset == "At class time":
            final_reminder_dt = class_dt
        else:
            c1, c2 = st.columns(2)
            with c1:
                custom_d = st.date_input("Date", value=class_dt.date(), key="custom_d")
            with c2:
                custom_t = st.time_input("Time", value=class_dt.time(), key="custom_t")
            final_reminder_dt = datetime.combine(custom_d, custom_t)

        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✅ Confirm Reminder", use_container_width=True):
                priority = predict_priority(final_reminder_dt, category)
                add_reminder(task, final_reminder_dt, category, priority)
                st.success(
                    f"Reminder set for {final_reminder_dt.strftime('%b %d, %Y – %H:%M')}!"
                )
                st.session_state.pending_reminder = None
                st.rerun()
        with col_cancel:
            if st.button("✖ Cancel", use_container_width=True):
                st.session_state.pending_reminder = None
                st.rerun()

        st.stop()

    # --- Normal add flow ---
    text = st.text_input(
        "Describe your task",
        placeholder="e.g. submit lab report by friday 5pm  ·  python assignment tomorrow",
    )

    task, auto_date = "", None
    if text:
        task, auto_date = extract_reminder(text)
        if auto_date:
            st.success(f"Detected: **{task}** · 📅 {auto_date.strftime('%b %d, %Y – %H:%M')}")
        else:
            st.info("No date detected – pick one below or use your timetable.")

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

            # --- Timetable matching ---
            if st.session_state.timetable:
                matched = find_matching_class(final_task, st.session_state.timetable, final_dt)
                if matched:
                    st.session_state.pending_reminder = (final_task, final_dt, category, matched)
                    st.rerun()

            # No timetable match → add directly
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

            st.markdown(f"**{task}**  \n`{due_text}` · {category} · {priority}")

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
