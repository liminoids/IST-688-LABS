import json
import streamlit as st
import requests
from openai import OpenAI

# ---------- Styling ----------
# Dark base theme lives in .streamlit/config.toml (so every widget and nav
# link is dark). This CSS layers Times New Roman and the mint/blue palette
# on top of it.
STYLE = """
<style>
/* Times New Roman everywhere except Streamlit's icon font and code */
.stApp :not(span[data-testid="stIconMaterial"]):not(.material-symbols-rounded):not(code):not(pre):not(kbd) {
    font-family: "Times New Roman", Times, serif !important;
}
/* page + chrome */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #1b2a2e 0%, #1e3038 50%, #1c2a3a 100%);
}
[data-testid="stHeader"] { background: rgba(27, 42, 46, 0.8); }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent; }
[data-testid="stSidebar"] { background: #22343a; border-right: 1px solid #3d5a5e; }
/* title + captions */
h1 { color: #a8e6cf; font-weight: 600; letter-spacing: 0.01em; }
h2, h3 { color: #9ecfe8; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #8fc1b5 !important; }
/* text input */
[data-testid="stTextInput"] input {
    background: #26404a; color: #e6f3ef;
    border: 1px solid #5f9c94; border-radius: 12px;
}
[data-testid="stTextInput"] input::placeholder { color: #7fb3a8; }
[data-testid="stTextInput"] label { color: #8fc1b5; }
/* buttons */
.stButton > button {
    background: #26404a; color: #e6f3ef;
    border: 1px solid #5f9c94; border-radius: 12px;
}
.stButton > button:hover { border-color: #a8e6cf; color: #a8e6cf; }
/* answer cards, chat bubbles, json */
[data-testid="stChatMessage"], [data-testid="stExpander"], [data-testid="stJson"] {
    background: rgba(44, 72, 78, 0.85);
    border: 1px solid #4f7d7a;
    border-radius: 18px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
}
[data-testid="stChatMessage"] { padding: 0.9rem 1.1rem; }
[data-testid="stChatMessage"] p { color: #e6f3ef; line-height: 1.55; }
a { color: #9ecfe8; }
/* spinner / status text */
[data-testid="stSpinner"] p { color: #a8e6cf; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

st.title('Lab 5 - The “What to Wear” Bot')


# ---------- Part A: the weather function ----------
# This is the plain Python half of the "tool": the function our program
# actually runs. The model never touches it; it only ever sees the JSON
# description we write in Part B, and asks for this by name.
#
# location can be a city, a zip code, an airport code ('SYR'), or a
# landmark ('Eiffel+Tower'). wttr.in needs no API key.
# Units are hard-coded to Fahrenheit / mph.
def get_current_weather(location):
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        # unknown locations come back as plain text, not JSON
        raise Exception(f"Could not find a location named {location}")

    # j1 has three top-level sections:
    #   current_condition -- one entry, conditions right now
    #   weather           -- three days, each with min/max, astronomy, hourly
    #   nearest_area      -- the place wttr.in actually matched
    current = data["current_condition"][0]
    today = data["weather"][0]
    astro = today["astronomy"][0]
    area = data["nearest_area"][0]

    # Every 3 hours through today. "What to wear" depends on the day's
    # arc, not just the reading right now: a 48F morning that hits 72F
    # with a 60% chance of rain at 3pm is a layers-and-umbrella day.
    hourly = [
        {
            "time": f"{int(h['time']) // 100:02d}:00",
            "temp_f": float(h["tempF"]),
            "feels_like_f": float(h["FeelsLikeF"]),
            "chance_of_rain_pct": int(h["chanceofrain"]),
            "chance_of_snow_pct": int(h["chanceofsnow"]),
            "wind_mph": float(h["windspeedMiles"]),
            "description": h["weatherDesc"][0]["value"].strip(),
        }
        for h in today["hourly"]
    ]

    return {
        "location": f"{area['areaName'][0]['value']}, "
                    f"{area['region'][0]['value']}, "
                    f"{area['country'][0]['value']}",
        "observed_at": current.get("localObsDateTime") or current.get("observation_time", ""),
        "temperature_f": float(current["temp_F"]),
        "feels_like_f": float(current["FeelsLikeF"]),
        "description": current["weatherDesc"][0]["value"].strip(),
        "humidity_pct": int(current["humidity"]),
        "wind_mph": float(current["windspeedMiles"]),
        "uv_index": int(current["uvIndex"]),
        "precip_inches": float(current["precipInches"]),
        "today_high_f": float(today["maxtempF"]),
        "today_low_f": float(today["mintempF"]),
        "sunrise": astro["sunrise"],
        "sunset": astro["sunset"],
        "hourly": hourly,
    }


# ---------- Part B: the “What to Wear” bot ----------
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

MODEL = "gpt-5-mini"
DEFAULT_LOCATION = "Syracuse, NY"

# The JSON half of the tool: the description the model reads. This is all
# the model ever sees of get_current_weather. It decides from the
# description whether to ask for it, and fills in "location" from what
# the user typed. The name is the string our code matches on below.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get today's weather for a location: current conditions, "
                "feels-like temperature, humidity, wind, UV index, today's high "
                "and low, sunrise and sunset, and an every-three-hours forecast "
                "with temperature and chance of rain or snow. Call this before "
                "giving clothing or outdoor-activity advice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state or country, e.g. Syracuse, NY "
                                       "or Lima, Peru. Also accepts a zip code or "
                                       "airport code.",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

SYSTEM_PROMPT = (
    "You are the What to Wear bot. When someone asks what to wear or do "
    "outside, use the get_current_weather tool to fetch today's weather for "
    "their location, then advise them. Structure the answer as: "
    "(1) What to wear today - layers, footwear, and rain or sun gear, keyed to "
    "the day's range and feels-like temperatures, not just the current reading; "
    "(2) Outdoor activities - two or three that suit the conditions, with the "
    "best hours for them; (3) One heads-up - rain chance, wind, UV, or sunset "
    "time, whichever matters most today. Be concrete and brief. All "
    "temperatures are Fahrenheit."
)

st.caption("Type a city and the bot fetches today\u2019s weather, then tells you what to wear and what to do outside.")
location = st.text_input("Location", placeholder=DEFAULT_LOCATION)

if st.button("What should I wear?"):
    asked_for = location.strip() or DEFAULT_LOCATION
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"What should I wear today in {asked_for}, and what could I do outside?"},
    ]

    # Call 1: send the question plus the tool description. The model
    # decides whether it needs the weather (tool_choice="auto") and, if so,
    # replies with a request instead of an answer.
    first = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    reply = first.choices[0].message

    if reply.tool_calls:
        # Keep the model's request in the transcript so the second call
        # can see what it asked for.
        messages.append({
            "role": "assistant",
            "content": reply.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in reply.tool_calls
            ],
        })

        # OUR program runs the function for each request the model made.
        # Every tool_call id must get a matching tool message back.
        for call in reply.tool_calls:
            if call.function.name != "get_current_weather":
                result = {"error": f"unknown tool {call.function.name}"}
            else:
                args = json.loads(call.function.arguments or "{}")
                where = (args.get("location") or "").strip() or DEFAULT_LOCATION
                try:
                    with st.spinner(f"Checking the weather in {where}..."):
                        result = get_current_weather(where)
                    st.caption(f"Tool called: get_current_weather(\u201c{where}\u201d)")
                    with st.expander("Weather data the bot used"):
                        st.json(result)
                except Exception as e:
                    result = {"error": str(e)}
                    st.error(str(e))
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": json.dumps(result),
            })

        # Call 2: the weather is now in the context as a tool result.
        # Ask for the advice and stream it.
        messages.append({
            "role": "user",
            "content": "Using that weather, tell me what to wear today and suggest "
                       "outdoor activities that suit it.",
        })
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)
    else:
        # The model answered without asking for the weather.
        st.markdown(reply.content or "")
