from __future__ import annotations

import csv
import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import urlopen


DB55_XLSX_URL = "https://zlab.wenglab.org/benchmark/Table_BM5.5.xlsx"


@dataclass(frozen=True)
class Db55Row:
    pdbid_1: str
    pdbid_2: str


_PDB4_RE = re.compile(r"([0-9][A-Za-z0-9]{3})")


def _pdb4(value: str) -> Optional[str]:
    if not value:
        return None
    m = _PDB4_RE.search(value.strip())
    return m.group(1).upper() if m else None


def download_db55_xlsx(dest_path: str, url: str = DB55_XLSX_URL) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urlopen(url, timeout=30) as resp:
        data = resp.read()
    with open(dest_path, "wb") as f:
        f.write(data)
    return dest_path


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    # Very small XML parser: pull <t>...</t> values.
    text = raw.decode("utf-8", errors="ignore")
    # sharedStrings can contain rich text: multiple <t> nodes per <si>.
    # We concatenate <t> nodes in each <si>.
    sis = text.split("<si")
    strings: List[str] = []
    for si in sis[1:]:
        parts = re.findall(r"<t[^>]*>(.*?)</t>", si, flags=re.DOTALL)
        s = "".join(parts)
        # unescape minimal entities
        s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        strings.append(s)
    return strings


def _read_sheet_xml(zf: zipfile.ZipFile) -> str:
    # Usually sheet1.xml; if not found, pick the first worksheet.
    candidates = [
        "xl/worksheets/sheet1.xml",
    ]
    for c in candidates:
        if c in zf.namelist():
            return zf.read(c).decode("utf-8", errors="ignore")

    for name in zf.namelist():
        if name.startswith("xl/worksheets/") and name.endswith(".xml"):
            return zf.read(name).decode("utf-8", errors="ignore")

    raise ValueError("No worksheet XML found in xlsx")


def _col_to_index(col_letters: str) -> int:
    # A -> 0, B -> 1, ..., Z -> 25, AA -> 26
    idx = 0
    for ch in col_letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _parse_sheet_cells(sheet_xml: str, shared_strings: List[str]) -> List[List[str]]:
    # Parse rows and cells from worksheet XML.
    # Extract <row r="..."> ... <c r="A1" t="s"><v>0</v></c>
    rows: Dict[int, Dict[int, str]] = {}

    for row_match in re.finditer(r"<row[^>]* r=\"(\d+)\"[^>]*>(.*?)</row>", sheet_xml, flags=re.DOTALL):
        r = int(row_match.group(1))
        row_xml = row_match.group(2)
        cells: Dict[int, str] = {}

        for cell_match in re.finditer(r"<c[^>]* r=\"([A-Z]+)(\d+)\"([^>]*)>(.*?)</c>", row_xml, flags=re.DOTALL):
            col_letters = cell_match.group(1)
            attrs = cell_match.group(3) or ""
            inner = cell_match.group(4) or ""

            col_idx = _col_to_index(col_letters)

            v_match = re.search(r"<v>(.*?)</v>", inner, flags=re.DOTALL)
            if not v_match:
                continue
            v = v_match.group(1).strip()

            # shared string
            if "t=\"s\"" in attrs:
                try:
                    s_idx = int(v)
                    cells[col_idx] = shared_strings[s_idx] if 0 <= s_idx < len(shared_strings) else ""
                except ValueError:
                    cells[col_idx] = ""
            else:
                cells[col_idx] = v

        rows[r] = cells

    if not rows:
        return []

    max_row = max(rows.keys())
    max_col = 0
    for _, cells in rows.items():
        if cells:
            max_col = max(max_col, max(cells.keys()))

    table: List[List[str]] = []
    for r in range(1, max_row + 1):
        cells = rows.get(r, {})
        table.append([cells.get(c, "") for c in range(0, max_col + 1)])

    return table


def extract_pairs_from_xlsx(xlsx_path: str) -> List[Db55Row]:
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_xml = _read_sheet_xml(zf)

    table = _parse_sheet_cells(sheet_xml, shared_strings)
    if not table:
        raise ValueError("Could not parse xlsx sheet into table")

    def norm_cell(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def norm_key(s: str) -> str:
        # Removes spaces/punctuation so header variants match.
        return re.sub(r"[^a-z0-9]", "", (s or "").strip().lower())

    def find_cols_in_row(row: List[str]) -> Optional[Tuple[int, int]]:
        header = [norm_cell(h) for h in row]
        header_keys = [norm_key(h) for h in row]

        def find_col(name: str) -> Optional[int]:
            name_n = norm_cell(name)
            name_k = norm_key(name)
            for i, (h, hk) in enumerate(zip(header, header_keys)):
                if hk == name_k or h == name_n:
                    return i
            for i, (h, hk) in enumerate(zip(header, header_keys)):
                if name_k and name_k in hk:
                    return i
                if name_n and name_n in h:
                    return i
            return None

        c1 = find_col("PDBid 1")
        c2 = find_col("PDBid 2")
        if c1 is None or c2 is None:
            return None
        return int(c1), int(c2)

    header_row_idx = None
    col_pdb1 = None
    col_pdb2 = None
    for i in range(min(60, len(table))):
        cols = find_cols_in_row(table[i])
        if cols:
            header_row_idx = i
            col_pdb1, col_pdb2 = cols
            break

    if header_row_idx is None or col_pdb1 is None or col_pdb2 is None:
        preview = [[norm_cell(c) for c in r[:10]] for r in table[:10]]
        raise ValueError(
            "Could not find header row containing 'PDBid 1' and 'PDBid 2'. "
            f"First rows preview: {preview}"
        )

    pairs: List[Db55Row] = []
    for row in table[header_row_idx + 1 :]:
        if col_pdb1 >= len(row) or col_pdb2 >= len(row):
            continue
        raw1 = str(row[col_pdb1] or "").strip()
        raw2 = str(row[col_pdb2] or "").strip()
        p1 = _pdb4(raw1)
        p2 = _pdb4(raw2)
        if not p1 or not p2:
            continue
        # ignore accidental header repeats
        if p1.lower() == "pdbi" or p2.lower() == "pdbi":
            continue
        pairs.append(Db55Row(pdbid_1=p1, pdbid_2=p2))

    # de-dup while preserving order
    seen = set()
    out: List[Db55Row] = []
    for p in pairs:
        key = (p.pdbid_1, p.pdbid_2)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)

    return out


def write_pairs_csv(rows: Iterable[Db55Row], csv_path: str) -> str:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdb1", "pdb2", "label", "source"])
        for r in rows:
            w.writerow([r.pdbid_1, r.pdbid_2, 1, "db5.5"])
    return csv_path
