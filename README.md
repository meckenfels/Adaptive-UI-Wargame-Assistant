# AI-Based Adaptive UI: Solo Wargame Rules Assistant

## Overview
This repository contains the source code for the AI-Based Adaptive Human-Computer Interaction project.
The application uses Scikit-Learn (TF-IDF Vectorization & Cosine Similarity) and Gradio to dynamically adapt a solo wargame rules interface at runtime based on user queries and navigation behavior.

## Key Adaptive Features
1. **Semantic Intent Prediction (TF-IDF):** Matches natural language rules queries to relevant game modules and updates real-time user affinity scores with temporal decay.
2. **Dynamic Menu Reordering:** Automatically promotes the highest-priority rule modules to the top of the navigation menu during runtime.
3. **Cognitive Load Density Adaptation:** Automatically transitions the interface from Novice Mode (Detailed step-by-step scaffolding) to Veteran Mode (Compact quick-reference tables) once the user reaches 4 interactions.
4. **Static vs. Adaptive Toggle:** Allows direct comparison between the baseline static interface and the AI-driven adaptive interface.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python app.py`
