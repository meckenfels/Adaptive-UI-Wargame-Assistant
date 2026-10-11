import gradio as gr
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

MODULES = {
    "01_Setup": {
        "title": "1. Setup & Deployment",
        "novice_text": "### 1. Setup & Deployment (Detailed Mode)\n* **Step 1:** Place the operational map on the table.\n* **Step 2:** Separate Division HQ counters and Combat Units by formation.\n* **Step 3:** Shuffle the Friction Deck and place it face down near the Turn Track.\n* *Tip: Set initial Divisional Morale to 10.*",
        "veteran_text": "**1. Setup Quick-Ref:** Map out -> Units by Formation -> Shuffle Friction Deck -> Init Morale = 10.",
        "keywords": "setup deploy start game board map counters shuffle friction deck initial placement prepare pieces"
    },
    "02_Command": {
        "title": "2. Command & Friction Phase",
        "novice_text": "### 2. Command & Friction Phase (Detailed Mode)\n* **Step 1:** Draw 1 card from the Friction Deck to determine weather and HQ bottlenecks.\n* **Step 2:** Roll 2d6 + HQ Rating to generate Command Points (CP).\n* **Step 3:** Assign CP to activate specific regiments within command radius (4 hexes).",
        "veteran_text": "**2. Command Table:** Draw 1 Friction Card | CP = 2d6 + HQ Rating | Command Radius = 4 Hexes.",
        "keywords": "command points cp headquarters hq friction card draw weather activation radius leader initiative orders"
    },
    "03_Movement": {
        "title": "3. Movement & Terrain",
        "novice_text": "### 3. Movement & Terrain (Detailed Mode)\n* **Mechanized Units:** 6 Movement Points (MP). Clear = 1 MP, Woods/Rough = 2 MP, River = +1 MP.\n* **Infantry Units:** 4 Movement Points (MP). All terrain = 1 MP except Marsh (2 MP).\n* **Zones of Control (ZOC):** Entering an enemy ZOC costs +2 MP and halts standard movement.",
        "veteran_text": "**3. Terrain Chart:** Mech (6 MP): Clear 1, Woods 2, River +1 | Inf (4 MP): All 1 (Marsh 2) | Enemy ZOC = +2 MP & Stop.",
        "keywords": "move movement terrain hexes woods river road mechanized tank infantry speed zone of control zoc halt"
    },
    "04_Combat": {
        "title": "4. Combat & CRT Resolution",
        "novice_text": "### 4. Combat & CRT Resolution (Detailed Mode)\n* **Step 1:** Total Attack Factors vs. Total Defense Factors to find the odds ratio (e.g., 3:1).\n* **Step 2:** Apply DRM (Dice Roll Modifiers): Woods (-1), Flanking (+2), Armor Superiority (+1).\n* **Step 3:** Roll 2d6 and cross-reference on the Combat Results Table (CRT).",
        "veteran_text": "**4. CRT Summary:** Odds (Atk:Def) | DRMs: Woods -1, Flank +2, Armor +1 | Roll 2d6 (2-5=AR, 6-8=EX, 9-10=DR, 11+=DE).",
        "keywords": "attack combat fight battle crt odds ratio dice roll modifier drm armor tank flanking defense casualties retreat"
    },
    "05_Supply": {
        "title": "5. Supply & Logistics",
        "novice_text": "### 5. Supply & Logistics (Detailed Mode)\n* **Tracing Supply:** Every unit must trace an unblocked path of up to 5 hexes to a road connected to a Supply Depot.\n* **Out of Supply (OOS) Penalty:** Halve Attack and Movement factors. Units OOS for 3 consecutive turns are eliminated.",
        "veteran_text": "**5. Supply Check:** 5 hexes to Road/Depot (unblocked by ZOC) | OOS Effect: 1/2 Atk & Move (3 turns = Elim).",
        "keywords": "supply logistics depot road trace path out of supply oos penalty ammo fuel starvation isolated"
    },
    "06_Victory": {
        "title": "6. Victory & Morale Checks",
        "novice_text": "### 6. Victory & Morale Checks (Detailed Mode)\n* **Objective Hexes:** Gain 2 Victory Points (VP) per Town hex controlled at the end of the turn.\n* **Morale Collapse:** If Divisional Morale drops to 0 due to combat losses, the scenario ends in an immediate tactical defeat.",
        "veteran_text": "**6. Victory/Morale:** Town Hex = 2 VP | Morale 0 = Auto-Defeat | Scenario End = Turn 8.",
        "keywords": "victory points vp win lose scenario end morale check collapse town objective score"
    }
}

module_keys = list(MODULES.keys())
corpus = [MODULES[k]["keywords"] + " " + MODULES[k]["title"] for k in module_keys]
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(corpus)

def create_initial_state():
    return {"scores": {k: 0.0 for k in module_keys}, "interactions": 0, "history_log": []}

def process_interaction(query_text, clicked_module_title, adaptive_enabled, density_override, state):
    state["interactions"] += 1
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    selected_key = None
    ai_explanation = ""
    for k in module_keys:
        state["scores"][k] *= 0.85
    if query_text and query_text.strip():
        query_vec = vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        selected_key = module_keys[best_idx]
        for idx, k in enumerate(module_keys):
            state["scores"][k] += float(similarities[idx]) * 2.5
        if best_score == 0:
            state["scores"][selected_key] += 0.5
            ai_explanation = f"No exact keyword match for '{query_text}'. Defaulting to {MODULES[selected_key]['title']}."
        else:
            ai_explanation = f"TF-IDF matched query '{query_text}' to **{MODULES[selected_key]['title']}** (Cosine Similarity: {best_score:.2f})."
        state["history_log"].append(f"[{timestamp}] Query: '{query_text}' -> Matched {MODULES[selected_key]['title']}")
    elif clicked_module_title:
        for k, v in MODULES.items():
            if v["title"] in clicked_module_title:
                selected_key = k
                break
        if not selected_key:
            selected_key = module_keys[0]
        state["scores"][selected_key] += 1.5
        ai_explanation = f"User selected **{MODULES[selected_key]['title']}**. Affinity weight increased (+1.5)."
        state["history_log"].append(f"[{timestamp}] Clicked: {MODULES[selected_key]['title']}")
    else:
        selected_key = module_keys[0]
    if not adaptive_enabled:
        active_density = "Novice (Detailed)"
        ordered_keys = module_keys
        adaptation_banner = "⚪ **STATIC UI MODE:** Interface layout and text density remain fixed regardless of user behavior."
    else:
        ordered_keys = sorted(module_keys, key=lambda k: state["scores"][k], reverse=True)
        if density_override == "Auto (AI-Driven)":
            active_density = "Veteran (Compact Quick-Ref)" if state["interactions"] >= 4 else "Novice (Detailed)"
        else:
            active_density = density_override
        top_predicted = MODULES[ordered_keys[0]]["title"]
        adaptation_banner = (
            f"🟢 **ADAPTIVE UI ACTIVE (Interaction #{state['interactions']}):**\n"
            f"- **AI Intent Engine:** {ai_explanation}\n"
            f"- **Dynamic Layout Reordering:** Promoted **{top_predicted}** to Top Slot (#1) based on usage & semantic similarity.\n"
            f"- **Runtime Density Mode:** `{active_density}` (Automatically switches to Compact Veteran tables after 4 interactions)."
        )
    if "Veteran" in active_density:
        content_display = f"### ⚡ Veteran Compact View (Auto-Adapted)\n{MODULES[selected_key]['veteran_text']}"
    else:
        content_display = MODULES[selected_key]["novice_text"]
    df_telemetry = pd.DataFrame([
        {"Rank": idx + 1, "Module": MODULES[k]["title"], "AI Priority Score": round(state["scores"][k], 3)}
        for idx, k in enumerate(ordered_keys)
    ])
    btn_updates = []
    for idx, k in enumerate(ordered_keys):
        prefix = "⭐ [Top Priority] " if (adaptive_enabled and idx == 0 and state["scores"][k] > 0) else ""
        btn_updates.append(gr.update(value=f"{prefix}{MODULES[k]['title']}"))
    log_text = "\n".join(state["history_log"][-6:])
    return [adaptation_banner, content_display, df_telemetry, log_text, "", state] + btn_updates

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    state = gr.State(create_initial_state())
    gr.Markdown("# 🎲 AI-Based Adaptive UI: Solo Wargame Rules Assistant")
    with gr.Row():
        adaptive_toggle = gr.Checkbox(label="Enable AI Runtime UI Adaptation", value=True)
        density_radio = gr.Radio(choices=["Auto (AI-Driven)", "Novice (Detailed)", "Veteran (Compact Quick-Ref)"], value="Auto (AI-Driven)", label="UI Information Density Mode")
    adaptation_status = gr.Markdown("🟢 **ADAPTIVE UI READY:** Interact via search or buttons below to observe runtime UI mutations.")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🧭 Dynamic Navigation Menu\n*(Reorders at runtime by AI priority)*")
            nav_buttons = [gr.Button(MODULES[k]["title"]) for k in module_keys]
        with gr.Column(scale=2):
            with gr.Row():
                search_box = gr.Textbox(label="Ask a Rules Question (NLP TF-IDF Search)", placeholder="e.g., How do I resolve tank combat odds or supply penalties?", scale=3)
                search_btn = gr.Button("Ask AI Assistant", variant="primary", scale=1)
            content_box = gr.Markdown(MODULES["01_Setup"]["novice_text"])
    with gr.Accordion("📊 View AI User Model Telemetry & Interaction Log", open=True):
        with gr.Row():
            telemetry_table = gr.Dataframe(label="Real-Time Module Priority Scores", interactive=False)
            history_box = gr.Textbox(label="Runtime Interaction Log", lines=5, interactive=False)
    outputs_list = [adaptation_status, content_box, telemetry_table, history_box, search_box, state] + nav_buttons
    search_btn.click(fn=process_interaction, inputs=[search_box, gr.State(""), adaptive_toggle, density_radio, state], outputs=outputs_list)
    search_box.submit(fn=process_interaction, inputs=[search_box, gr.State(""), adaptive_toggle, density_radio, state], outputs=outputs_list)
    for btn in nav_buttons:
        btn.click(fn=process_interaction, inputs=[gr.State(""), btn, adaptive_toggle, density_radio, state], outputs=outputs_list)

if __name__ == "__main__":
    demo.launch(inline=True, share=True)
