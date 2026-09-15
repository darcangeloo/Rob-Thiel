import json

def valida_e_scrivi_batch(testo_batch, path_file):
    scritte = 0
    scartate = 0

    righe = [r for r in testo_batch.strip().split("\n") if r.strip()]

    with open(path_file, "a", encoding="utf-8") as f:
        for riga in righe:  # ogni riga è un oggetto JSON separato (formato JSONL)
            try:
                esempio = json.loads(riga)
            except json.JSONDecodeError as e:
                print(f"Riga scartata (JSON non valido): {e}")
                scartate += 1
                continue

            chiavi_richieste = {"instruction", "input", "output", "category"}
            if not chiavi_richieste.issubset(esempio.keys()):
                print(f"Esempio scartato (chiavi mancanti): {esempio}")
                scartate += 1
                continue

            f.write(json.dumps(esempio, ensure_ascii=False) + "\n")
            scritte += 1

    print(f"Scritte: {scritte}, scartate: {scartate}")
    return scritte, scartate