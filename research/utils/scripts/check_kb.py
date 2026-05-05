import json

def check_missing_keywords(old_json_path, new_json_path):
    # Membaca file lama
    with open(old_json_path, 'r', encoding='utf-8') as f:
        old_kb = json.load(f)
    
    # Membaca file baru
    with open(new_json_path, 'r', encoding='utf-8') as f:
        new_kb = json.load(f)
        
    # Mengumpulkan semua keyword (canonical & variants) dari KB lama
    old_keywords = set()
    for unit, unit_data in old_kb.items():
        # Mengecek format iteration_4 (high, low, override)
        for severity in ['high', 'low', 'override']:
            if severity in unit_data:
                for item in unit_data[severity]:
                    if item.get('canonical'):
                        old_keywords.add(item['canonical'].strip().lower())
                    for v in item.get('variants', []):
                        if v:
                            old_keywords.add(v.strip().lower())

    # Mengumpulkan semua keyword (canonical & variants) dari KB baru
    new_keywords = set()
    for unit, unit_data in new_kb.items():
        for key, items in unit_data.items():
            # Abaikan key root yang bukan merupakan list array issue
            if key not in ['description', 'default_sla', 'full_name']:
                for item in items:
                    if item.get('canonical'):
                        new_keywords.add(item['canonical'].strip().lower())
                    for v in item.get('variants', []):
                        if v:
                            new_keywords.add(v.strip().lower())

    # Mencari selisih (keyword di old tapi tidak ada di new)
    missing_in_new = old_keywords - new_keywords
    
    print(f"Total keyword lama: {len(old_keywords)}")
    print(f"Total keyword baru: {len(new_keywords)}")
    print(f"\nJumlah keyword yang tertinggal: {len(missing_in_new)}")
    
    print("\nDaftar Keyword yang tertinggal:")
    for kw in sorted(missing_in_new):
        print(f"- {kw}")

# Cara penggunaan
if __name__ == "__main__":
    check_missing_keywords('../kb_20260429_iteration_4.json', '../kb_injected_final.json')