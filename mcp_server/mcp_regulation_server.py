from fastmcp import FastMCP
import json, csv, unicodedata
from pathlib import Path

mcp = FastMCP("Regulation")

CSV_PATH = Path("../data/xport.csv")  # adjust to your real location
regulation_rows = []
regulation_index = {}

def _normalize_code(raw: str) -> str:
    if not raw:
        return ""
    cleaned = unicodedata.normalize("NFKD", raw)
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    return cleaned.strip().upper()

with CSV_PATH.open("r", encoding="cp1252", newline="") as fh:
    reader = csv.DictReader(fh, delimiter=";")  # or "," depending on the file
    for row in reader:
        key = _normalize_code(row.get("Réference"))
        if not key:
            continue  # ignore rows without a reference
        regulation_rows.append(row)
        regulation_index[key] = row

@mcp.tool()
def lookup_regulation(regulation_code: str) -> str:
    """Fetch the official status and metadata for a regulation code (e.g., EN 62304:2006/AC). Always call this first for regulation queries."""
    key = _normalize_code(regulation_code)
    row = regulation_index.get(key)
    if not row:
        return json.dumps({"regulation_code": regulation_code, "error": "not found"}, ensure_ascii=True)
    return json.dumps({"regulation_code": regulation_code, "regulation_info": row}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8100)