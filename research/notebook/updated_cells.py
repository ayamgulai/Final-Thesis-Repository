# ============================================================
# UPDATED NOTEBOOK CELLS
# Copy these into your notebook to replace the current cells.
# ============================================================

# ============================================================
# CELL: NEW - Add right after "df.dropna(subset=['unit_kerja']...)" cell
# (After the NaN drop cell, before "Exploratory Insight Before Applying KB")
# 
# SECTION TITLE: "## Preprocess Title and Body"
# ============================================================

# --- Markdown cell ---
# ## Preprocess Title and Body

# --- Code cell ---
# Apply preprocessing to title and body columns, store as new columns
df['title_clean'] = df['title'].astype(str).apply(preprocess)
df['body_clean'] = df['body'].astype(str).apply(preprocess)

# Combine into a single preprocessed text column
df['preprocessed_text'] = df['title_clean'] + ' ' + df['body_clean']

print(f"Preprocessing done for {len(df)} rows")
print(f"Example title_clean: {df['title_clean'].iloc[0]}")
print(f"Example body_clean (first 100 chars): {df['body_clean'].iloc[0][:100]}")


# ============================================================
# NOTE: The preprocess() function definition stays the same, 
# but it must be defined BEFORE this cell (move it up before 
# the exploratory section). Currently it's defined at line 1499-1517
# which is after the exploratory section.
# Move the preprocess() function cell to right after the import cell.
# ============================================================


# ============================================================
# CELL: REPLACE the implement_kb function (currently at lines 1544-1619)
# This version NO LONGER calls preprocess() on row data — 
# it reads from the pre-cleaned columns instead.
# It also pre-processes KB keywords once via a helper.
# ============================================================

def preprocess_kb(kb):
    """Pre-process all keywords in the KB once, so implement_kb doesn't need to."""
    processed_kb = {}
    for unit, levels in kb.items():
        processed_kb[unit] = {}
        for level, items in levels.items():
            processed_items = []
            for item in items:
                processed_item = {
                    "canonical": item["canonical"],
                    "canonical_clean": preprocess(item["canonical"]),
                    "variants_clean": [preprocess(v) for v in item.get("variants", [])],
                }
                processed_items.append(processed_item)
            processed_kb[unit][level] = processed_items
    return processed_kb


def implement_kb(row, kb):
    # Use pre-cleaned text from the DataFrame columns (no preprocessing here)
    full_text = row['preprocessed_text']

    WEIGHT_MAP = {"high": 3, "low": 1}
    normalized_scores = {}
    raw_scores = {}
    match_details = {}

    # --- STEP 1: OVERRIDE CHECK (Skor Otomatis 1.0) ---
    for unit, levels in kb.items():
        if "override" in levels:
            for item in levels["override"]:
                keywords = [item["canonical_clean"]] + item.get("variants_clean", [])

                if any(re.search(rf"\b{re.escape(kw)}\b", full_text) for kw in keywords if kw):
                    return pd.Series([unit, 1.0, 1.0, [item["canonical"]], f"{unit}: Override", f"{unit}: {item['canonical']}", full_text])

    # --- STEP 2: WEIGHTED SCORING (High & Low) ---
    for unit, levels in kb.items():
        unit_raw_score = 0
        total_possible_weight = 0
        matches = []

        for level, items in levels.items():
            if level == "override":
                continue

            weight = WEIGHT_MAP.get(level, 0)
            for item in items:
                keywords = [item["canonical_clean"]] + item.get("variants_clean", [])

                for kw in keywords:
                    if not kw: continue

                    total_possible_weight += weight
                    if re.search(rf"\b{re.escape(kw)}\b", full_text):
                        unit_raw_score += weight
                        matches.append(kw)

        if matches:
            normalized_scores[unit] = unit_raw_score / total_possible_weight if total_possible_weight > 0 else 0
            raw_scores[unit] = unit_raw_score
            match_details[unit] = list(set(matches))

    # --- STEP 3: SELEKSI TERBAIK ---
    if not normalized_scores:
        return pd.Series(["UNKNOWN", 0.0, 0.0, [], "", "", full_text])

    sorted_units = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    best_unit, best_norm_score = sorted_units[0]

    second_norm_score = sorted_units[1][1] if len(sorted_units) > 1 else 0
    confidence = best_norm_score / (best_norm_score + second_norm_score) if (best_norm_score + second_norm_score) > 0 else 1.0

    all_scores_str = ", ".join([f"{u}: {raw_scores[u]}" for u in raw_scores])
    all_keywords_str = ", ".join([f"{u}: {'|'.join(match_details[u])}" for u in match_details])
    
    return pd.Series([
        best_unit,
        round(best_norm_score, 3),
        round(confidence, 3),
        match_details.get(best_unit, []),
        all_scores_str,
        all_keywords_str,
        full_text
    ])


# ============================================================
# CELL: REPLACE the KB loading + apply cells
# Instead of: kb_data = load_kb(...)
# Use:        kb_data = preprocess_kb(load_kb(...))
# 
# And instead of: df.apply(lambda row: implement_kb(row, kb_data), axis=1)
# The apply stays the same, but the function now reads pre-cleaned columns
# ============================================================

# --- First Implementation ---
kb_data = preprocess_kb(load_kb('../utils/kb_20260408_first.json'))

df_kb_label = df.copy()
df_kb_label[['predicted_unit', 'score', 'confidence', 'keywords', 'all_unit_scores', 'all_unit_keywords', 'preprocessed_text']] = df.apply(
    lambda row: implement_kb(row, kb_data), axis=1
)
df_kb_label.to_excel("../data/output/20260408_kb_implement_match_keyword.xlsx", index=False)


# --- Iteration 1 ---
kb_iteration_1 = preprocess_kb(load_kb("../utils/kb_20260409_iteration_1.json"))

df_kb_label_1 = df_kb_label.copy()
df_kb_label_1[['predicted_unit', 'score', 'confidence', 'keywords', 'all_unit_scores', 'all_unit_keywords', 'preprocessed_text']] = df.apply(
    lambda row: implement_kb(row, kb_iteration_1), axis=1
)
df_kb_label_1.to_excel("../data/output/20260410_kb_implement_match_keyword_iteration_1.xlsx", index=False)


# --- Iteration 2 ---
kb_iteration_2 = preprocess_kb(load_kb("../utils/kb_20260414_iteration_2.json"))

df_kb_label_2 = df_kb_label.copy()
df_kb_label_2[['predicted_unit', 'score', 'confidence', 'keywords', 'all_unit_scores', 'all_unit_keywords', 'preprocessed_text']] = df.apply(
    lambda row: implement_kb(row, kb_iteration_2), axis=1
)
df_kb_label_2.to_excel("../data/output/20260415_kb_implement_match_keyword_iteration_2_001.xlsx", index=False)
