import streamlit as st
import openai
import json
import os
import textwrap
from typing import List, Dict

st.set_page_config(page_title="ACOTAR-style Interactive Novel", layout="wide")

####################
# Helper utilities #
####################
def init_session_state():
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "pc" not in st.session_state:
        st.session_state.pc = None  # player character dict
    if "story_history" not in st.session_state:
        st.session_state.story_history = []  # list of dicts {narrative, choice_text, raw_llm}
    if "current_scene" not in st.session_state:
        st.session_state.current_scene = None
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4o-mini"  # change if needed
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.8
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 500

def pretty_json(d):
    return json.dumps(d, indent=2, ensure_ascii=False)

###############
# Prompts     #
###############
DM_SYSTEM_PROMPT = textwrap.dedent("""
You are the Dungeon Master / Narrative Engine for an original, fan-inspired interactive novel set in a romantic high-fantasy fae world. 
DO NOT reference your system role or the model. The PLAYER should never see technical details or API text.

Rules:
1. Write evocative, novel-style narrative in the voice of lush romantic fantasy (no direct quotes from existing copyrighted novels).
2. Each response must be valid JSON (no extra text). The JSON object must contain exactly these keys:
   - "narrative": a string paragraph (2-6 sentences) continuing the story.
   - "choices": an array of 3 to 5 strings; short, distinct options the player can click next.
   - "state_updates": an object with any small changes to the player state (e.g., {"relationship_with_Rhys": 1, "hp": -1}) — can be empty.
   - "scene_id": a short identifier for the scene.
3. After the JSON object, nothing else should be output.
4. If the player's input is a free-text custom action (not one of the choices), interpret it plausibly in-world, then generate the next scene normally and populate narrative/choices/state_updates.
5. Keep choices consequential and clearly different (avoid duplicates).
6. Always assume the player character is present and make scenes appropriate for a mid-tier powerful protagonist (not an invincible god).
7. If the story risks large-scale violence or sexual content, keep it brief and non-graphic; do not produce explicit sexual content.
8. Maintain continuity based on the 'player_state' and 'history' provided in the messages.
9. Use short scene_id strings, e.g., "mansion_hall_01".
""").strip()

def build_messages(pc: Dict, history: List[Dict], player_input: str, is_custom: bool):
    """
    Build Chat API messages with:
    - system: DM_SYSTEM_PROMPT
    - user: current context including pc & history & player input
    """
    # Build a compact history summary to pass to LLM (last ~6 entries)
    hist_excerpt = history[-6:]
    summary_lines = []
    for i, h in enumerate(hist_excerpt, 1):
        entry = h.get("narrative", "")
        chosen = h.get("choice_taken", "")
        summary_lines.append(f"{i}. Scene: {entry}\n   Choice: {chosen}")
    history_block = "\n".join(summary_lines) if summary_lines else "None yet."

    player_state_block = json.dumps(pc, ensure_ascii=False)

    user_instructions = textwrap.dedent(f"""
    Player character (JSON): {player_state_block}

    Recent history (most recent first):
    {history_block}

    The player's most recent input (either a choice label exactly as shown in 'choices' or a free-text custom action):
    {player_input}

    If the input exactly matches one of the previous choices, continue that branch. If it is a custom action, interpret it and continue.
    Remember to respond ONLY with valid JSON as specified by the system prompt.
    """).strip()

    messages = [
        {"role": "system", "content": DM_SYSTEM_PROMPT},
        {"role": "user", "content": user_instructions},
    ]
    return messages

####################
# LLM interaction  #
####################
def call_llm(messages: List[Dict], model: str, temperature: float, max_tokens: int):
    openai.api_key = st.session_state.openai_api_key
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1
        )
        text = resp["choices"][0]["message"]["content"]
        return text
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

def parse_llm_json(text: str):
    # LLM instructed to return only JSON. But sometimes it wraps in markdown or adds stray text.
    # Attempt to locate the first `{` and parse JSON from that.
    try:
        start = text.index("{")
        json_text = text[start:]
        parsed = json.loads(json_text)
        return parsed
    except Exception as e:
        # Fallback try: strip markdown fences
        cleaned = text.strip().strip("```").strip()
        try:
            parsed = json.loads(cleaned)
            return parsed
        except Exception as e2:
            return None

####################
# Story functions  #
####################
def start_new_adventure():
    # Reset story history
    st.session_state.story_history = []
    st.session_state.current_scene = {
        "scene_id": "prologue_01",
        "narrative": f"{st.session_state.pc['name']} arrives at the border of a glittering fae court as twilight bleeds across the sky."
    }
    # We will call LLM for the next "real" scene after the player's initial choice.
    st.experimental_rerun()

def take_choice(choice_text: str, is_custom=False):
    pc = st.session_state.pc
    history = st.session_state.story_history
    messages = build_messages(pc, history, choice_text, is_custom)
    raw = call_llm(messages, st.session_state.model, st.session_state.temperature, st.session_state.max_tokens)
    if raw is None:
        st.error("No response from LLM.")
        return
    parsed = parse_llm_json(raw)
    if parsed is None:
        st.error("Failed to parse LLM output as JSON. Raw output:")
        st.code(raw)
        return
    # Apply state updates if any
    updates = parsed.get("state_updates") or {}
    for k, v in updates.items():
        # simple numeric updates
        if isinstance(v, (int, float)):
            st.session_state.pc[k] = st.session_state.pc.get(k, 0) + v
        else:
            st.session_state.pc[k] = v
    # Push to history
    history_entry = {
        "scene_id": parsed.get("scene_id"),
        "narrative": parsed.get("narrative"),
        "choices": parsed.get("choices"),
        "choice_taken": choice_text,
        "raw_llm": raw
    }
    st.session_state.story_history.append(history_entry)
    st.session_state.current_scene = history_entry
    # Rerun to update UI
    st.experimental_rerun()

####################
# UI Layout         #
####################
init_session_state()

st.title("ACOTAR-style Interactive Novel — Streamlit Demo")
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Settings")
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key (will be stored only in this session)", 
        type="password", 
        value=st.session_state.openai_api_key
    )
    st.session_state.model = st.selectbox("Model", options=["gpt-4o-mini", "gpt-4o", "gpt-4o-mid"], index=0)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.5, st.session_state.temperature, 0.05)
    st.session_state.max_tokens = st.number_input("Max tokens (response)", min_value=120, max_value=2000, value=st.session_state.max_tokens, step=10)
    st.markdown("---")
    st.markdown("**Session quick-controls**")
    if st.button("Restart session (clear story)"):
        st.session_state.story_history = []
        st.session_state.current_scene = None
        st.experimental_rerun()

with col1:
    # Character creation / summary
    if st.session_state.pc is None:
        st.header("Create your character")
        with st.form("pc_form"):
            name = st.text_input("Name", value="Aryn")
            court = st.selectbox("Court Affiliation", ["Night Court (mysterious)", "Dawn Court (stately)", "Spring Court (warmth)", "Autumn Court (wary)", "Independent / None"])
            archetype = st.selectbox("Archetype", ["Wary Survivor", "Scholarly Heir", "Ambitious Outsider", "Reckless Champion"])
            strength = st.slider("Strength (physical prowess)", 1, 10, 5)
            guile = st.slider("Guile (social skill / stealth)", 1, 10, 6)
            magic = st.slider("Magic affinity", 0, 10, 3)
            submit = st.form_submit_button("Create & Start Adventure")
        if submit:
            st.session_state.pc = {
                "name": name,
                "court": court,
                "archetype": archetype,
                "strength": strength,
                "guile": guile,
                "magic": magic,
                "hp": 10,
                "relationship": {},
                "inventory": []
            }
            start_new_adventure()
        else:
            st.info("Fill in the form and press 'Create & Start Adventure' to begin.")
            st.stop()
    else:
        # Show PC summary
        pc = st.session_state.pc
        st.header(f"Player: {pc['name']} — {pc['court']}")
        colpc1, colpc2, colpc3 = st.columns(3)
        colpc1.metric("Strength", pc.get("strength", 0))
        colpc2.metric("Guile", pc.get("guile", 0))
        colpc3.metric("Magic", pc.get("magic", 0))
        st.write("HP:", pc.get("hp"))
        st.write("Inventory:", ", ".join(pc.get("inventory", []) or ["—"]))
        st.markdown("---")

    # Current scene / story area
    st.subheader("Story")
    if st.session_state.current_scene is None:
        st.write("The world waits. Press **Begin** to ask the Narrative Engine for the opening scene.")
        if st.button("Begin Adventure"):
            # Fire off a special "start" prompt that seeds the prologue
            # Use the current pc and an initial prompt like "open with a twilight border scene"
            start_prompt = "Begin the adventure with a twilight border scene: the protagonist approaches a glittering fae court border. Provide narrative and 3 choices."
            messages = build_messages(st.session_state.pc, st.session_state.story_history, start_prompt, is_custom=True)
            raw = call_llm(messages, st.session_state.model, st.session_state.temperature, st.session_state.max_tokens)
            if raw is None:
                st.error("Failed to get opening scene from LLM.")
            else:
                parsed = parse_llm_json(raw)
                if parsed is None:
                    st.error("Failed to parse LLM output for opening scene. Raw output shown:")
                    st.code(raw)
                else:
                    entry = {
                        "scene_id": parsed.get("scene_id"),
                        "narrative": parsed.get("narrative"),
                        "choices": parsed.get("choices"),
                        "choice_taken": "",
                        "raw_llm": raw
                    }
                    st.session_state.story_history.append(entry)
                    st.session_state.current_scene = entry
                    st.experimental_rerun()
        st.stop()
    else:
        scene = st.session_state.current_scene
        st.markdown(f"**Scene ID**: `{scene.get('scene_id', 'unknown')}`")
        st.write(scene.get("narrative", ""))

        # Show history / breadcrumbs
        with st.expander("History (recent)"):
            for i, h in enumerate(reversed(st.session_state.story_history[-10:])):
                st.markdown(f"**{h.get('scene_id','?')}** — {h.get('choice_taken','(start)')}")
                st.write(h.get("narrative"))

        # Present choices as buttons
        st.markdown("---")
        choices = scene.get("choices") or []
        if choices:
            st.subheader("Choices")
            cols = st.columns(len(choices))
            for i, choice in enumerate(choices):
                if cols[i].button(choice):
                    take_choice(choice, is_custom=False)
        else:
            st.info("No choices in this scene. You may type a custom action.")

        st.markdown("**Or type a custom action**")
        with st.form("custom_action_form", clear_on_submit=False):
            custom_action = st.text_input("Describe what you want to do:", "")
            submit_custom = st.form_submit_button("Do custom action")
        if submit_custom and custom_action.strip():
            take_choice(custom_action.strip(), is_custom=True)

        # Small tools
        st.markdown("---")
        st.subheader("Quick tools")
        if st.button("Show raw LLM JSON for this scene"):
            st.code(pretty_json({
                "scene_id": scene.get("scene_id"),
                "narrative": scene.get("narrative"),
                "choices": scene.get("choices")
            }))

        if st.button("Export story (JSON)"):
            export = {
                "player": st.session_state.pc,
                "history": st.session_state.story_history
            }
            st.download_button("Download story JSON", data=json.dumps(export, ensure_ascii=False, indent=2), file_name="adventure_export.json", mime="application/json")

# Footer
st.markdown("---")
st.caption("This demo app is a prototype. You may expand the DM prompt for stricter JSON, add dice/roll logic locally, or build a more complex state machine. Do not publish any fanwork commercially without proper rights.")
