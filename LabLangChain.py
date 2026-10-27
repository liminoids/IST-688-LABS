import os
import streamlit as st
import requests
from langchain.agents import create_agent

st.title('Lab – LangChain: The “What to Wear” Bot, Rebuilt')
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def get_current_weather(location: str = "Syracuse, NY") -> dict:
    """Get today's weather for a location: current conditions, feels-like
    temperature, humidity, wind, UV index, today's high and low, sunrise and
    sunset, and an every-three-hours forecast with temperature and chance of
    rain or snow. Call this before giving clothing or outdoor-activity advice.

    location: city and state or country, e.g. Syracuse, NY or Lima, Peru.
    Also accepts a zip code or airport code. Defaults to Syracuse, NY.
    """
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        raise Exception(f"Could not find a location named {location}")

    current = data["current_condition"][0]
    today = data["weather"][0]
    astro = today["astronomy"][0]
    area = data["nearest_area"][0]

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

agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_current_weather],
    system_prompt=SYSTEM_PROMPT,
)

location = st.text_input("Location", placeholder="Syracuse, NY")

if st.button("What should I wear?"):
    with st.spinner("Checking the weather…"):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"What should I wear in {location}?"}]}
        )
    with st.chat_message("assistant"):
        st.write(result["messages"][-1].content)

    with st.expander("Behind the scenes"):
        for message in result["messages"]:
            st.write(f"**{message.type}**")
            st.write(message.content)
            if message.type == "ai" and message.tool_calls:
                st.write(message.tool_calls)
# ---------- Part D, step 12 ----------
# 12a. Four messages:
#   1. human — my question ("What should I wear in Syracuse, NY?")
#   2. ai    — content empty; tool_calls asks for get_current_weather(location="Syracuse, NY")
#   3. tool  — the weather dict my function returned, as text
#   4. ai    — the advice
#   No system message in the list: the system prompt is sent with each model call, not stored.
# 12b. Message 2. Its content is empty; the request is in message.tool_calls —
#      a list holding the tool name, the arguments, and an id the tool message answers to.
# 12c. In Lab5.py:
#   human          → the {"role": "user", ...} dict in the messages = [...] list
#   ai (tool call) → first = client.chat.completions.create(...) and the
#                    messages.append({"role": "assistant", ..., "tool_calls": [...]}) after it
#   tool           → the for call in reply.tool_calls: loop, ending in
#                    messages.append({"role": "tool", "tool_call_id": call.id, ...})
#   ai (answer)    → stream = client.chat.completions.create(model=MODEL, messages=messages, stream=True)
#   Lab 5 also appended an extra user nudge ("Using that weather, tell me...") with no counterpart here.

