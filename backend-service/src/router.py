import re
import html
from rapidfuzz import fuzz
import json
import os

class TicketRouter:
    def __init__(self, kb_data):
        self.kb_data = kb_data
        
        # Read threshold from config.json
        base_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_dir, 'config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.THRESHOLD = float(config.get("matching_score_threshold", 85.0))
        except Exception:
            self.THRESHOLD = 85.0  # Fallback if config is missing or invalid

    # show in appendix
    def _search_in_unit(self, query, unit_name):
        # Bersihkan HTML tag dan unescape karakter HTML sebelum keyword matching
        clean_query = html.unescape(query)
        clean_query = re.sub(r'<[^>]+>', ' ', clean_query)
        
        # Tokenisasi query
        query_tokens = re.findall(r'\b\w+\b', clean_query.lower())
        
        best_match = {
            "score": 0, "unit": unit_name, "issue_category": None, 
            "canonical": None, "sla": None
        }
        
        unit_data = self.kb_data.get(unit_name)
        if not unit_data:
            return best_match
            
        # Mengambil fallback SLA dari unit jika category tidak punya SLA spesifik
        default_sla = unit_data.get("default_sla", {})
        categories = unit_data.get("issue_category", {})

        for issue_name, issue_content in categories.items():
            # Ekstrak SLA dari sub-dict "sla", fallback ke default_sla
            issue_sla = issue_content.get("sla", {})
            sla = {
                "response_time": issue_sla.get("response_time", default_sla.get("response_time")),
                "resolution_time": issue_sla.get("resolution_time", default_sla.get("resolution_time"))
            }

            canonicals = issue_content.get("canonicals",)
            for canonical_item in canonicals:
                canonical_name = canonical_item.get("canonical", "")
                variants = canonical_item.get("variants",)
                
                for variant in variants:
                    variant_clean = variant.lower()
                    # Jangan tokenisasi variant dari knowledge base —
                    # gunakan string aslinya agar karakter khusus tidak hilang.
                    # n dihitung dari jumlah kata (split) untuk menentukan window size.
                    variant_words = variant_clean.split()
                    
                    if not variant_words or not query_tokens:
                        continue
                        
                    n = len(variant_words)
                    max_score = 0
                    
                    # Window size exactly N (Exact token length match)
                    for i in range(len(query_tokens) - n + 1):
                        window_str = " ".join(query_tokens[i:i+n])
                        score = fuzz.ratio(window_str, variant_clean)
                        if score > max_score: max_score = score
                        
                    # Window size N+1 (Menangani typo split word, misal 'myits wifi' -> 'my its wifi')
                    if len(query_tokens) >= n + 1:
                        for i in range(len(query_tokens) - n):
                            window_str = " ".join(query_tokens[i:i+n+1])
                            score = fuzz.ratio(window_str, variant_clean)
                            if score > max_score: max_score = score
                            
                    # Window size N-1 (Menangani typo merged word, misal 'myits wifi' -> 'myitswifi')
                    if n > 1 and len(query_tokens) >= n - 1:
                        for i in range(len(query_tokens) - n + 2):
                            window_str = " ".join(query_tokens[i:i+n-1])
                            score = fuzz.ratio(window_str, variant_clean)
                            if score > max_score: max_score = score

                    # Catatan: tidak ada fallback untuk query yang lebih pendek dari n-1 token.
                    # Jika query terlalu pendek untuk mengisi window manapun (N, N+1, N-1),
                    # max_score tetap 0 — artinya variant ini tidak cocok.
                    # Ini mencegah query pendek (misal: "myits") secara keliru
                    # mendapat skor dari variant panjang (misal: "myits human capital").

                    if max_score > best_match["score"]:
                        best_match.update({
                            "score": max_score,
                            "issue_category": issue_name,
                            "canonical": canonical_name,
                            "sla": sla
                        })
        return best_match

    def assign_ticket_sla(self, query, predicted_unit):
        # Tahap 1: Cari kecocokan keyword di unit yang diprediksi model (Local Search)
        best_match = self._search_in_unit(query, predicted_unit)

        # Tahap 2: Evaluasi hasil
        if best_match["score"] >= self.THRESHOLD:
            # Keyword ditemukan di unit prediksi → gunakan SLA spesifik dari issue category
            return {
                "assigned_unit": best_match["unit"],
                "issue_category": best_match["issue_category"],
                "canonical": best_match["canonical"],
                "assigned_sla": best_match["sla"],
                "keyword_match_score": round(best_match["score"], 2),
                "is_default_sla": False
            }
        else:
            # Keyword tidak ditemukan → percaya prediksi ML, gunakan Default SLA unit tersebut
            unit_data = self.kb_data.get(predicted_unit, {})
            default_sla = unit_data.get("default_sla", {})
            return {
                "assigned_unit": predicted_unit,
                "issue_category": None,
                "canonical": None,
                "assigned_sla": {
                    "response_time": default_sla.get("response_time"),
                    "resolution_time": default_sla.get("resolution_time")
                },
                "keyword_match_score": round(best_match["score"], 2),
                "is_default_sla": True
            }