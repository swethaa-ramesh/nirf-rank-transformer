import os
import pandas as pd
from bs4 import BeautifulSoup, Tag, NavigableString

# Path to the saved HTML table
HTML_PATH = os.path.join("data", "raw", "nirf_2023_table.html")
CSV_PATH = os.path.join("data", "raw", "nirf_2023.csv")

def parse_nirf_table(html_path, csv_path):
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    
    table = soup.find("table", {"id": "tbl_overall"})
    if not isinstance(table, Tag):
        print("Table with id 'tbl_overall' not found.")
        return
    
    thead = table.find("thead") if isinstance(table, Tag) else None
    if not isinstance(thead, Tag):
        print("No thead found in table.")
        return
    
    headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    sub_headers = ["TLR (100)", "RPC (100)", "GO (100)", "OI (100)", "PERCEPTION (100)"]
    all_headers = headers[:2] + sub_headers + headers[2:]
    
    rows = []
    tbody = table.find("tbody") if isinstance(table, Tag) else None
    if not isinstance(tbody, Tag):
        print("No tbody found in table.")
        return
    
    for idx, tr in enumerate(tbody.find_all("tr")):
        if not isinstance(tr, Tag):
            continue
        
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 6:
            continue
        
        # Extract main table data
        institute_id = cells[0].get_text(strip=True) if len(cells) > 0 else ""
        
        # Extract name (first text content, ignoring the div with buttons)
        name_cell = cells[1] if len(cells) > 1 else None
        if name_cell and isinstance(name_cell, Tag):
            name_text = ""
            for content in name_cell.contents:
                if isinstance(content, NavigableString):
                    name_text = content.strip()
                    break
            name = name_text
        else:
            name = ""
        
        # Extract sub-table data from the hidden div
        hidden_div = name_cell.find("div", {"class": "tbl_hidden"}) if name_cell else None
        tlr = rpc = go = oi = perception = ""
        if hidden_div and isinstance(hidden_div, Tag):
            sub_table = hidden_div.find("table", {"class": "table"})
            if sub_table and isinstance(sub_table, Tag):
                sub_tbody = sub_table.find("tbody")
                if sub_tbody and isinstance(sub_tbody, Tag):
                    sub_row = sub_tbody.find("tr")
                    if sub_row and isinstance(sub_row, Tag):
                        sub_cells = sub_row.find_all("td")
                        if len(sub_cells) >= 5:
                            tlr = sub_cells[0].get_text(strip=True)
                            rpc = sub_cells[1].get_text(strip=True)
                            go = sub_cells[2].get_text(strip=True)
                            oi = sub_cells[3].get_text(strip=True)
                            perception = sub_cells[4].get_text(strip=True)
        
        # Now extract city, state, score, rank from the correct cells
        # The next four cells after name_cell are city, state, score, rank
        city = cells[2].get_text(strip=True) if len(cells) > 2 else ""
        state = cells[3].get_text(strip=True) if len(cells) > 3 else ""
        score = cells[4].get_text(strip=True) if len(cells) > 4 else ""
        rank = cells[5].get_text(strip=True) if len(cells) > 5 else ""
        
        if idx == 0:
            print(f"DEBUG: {institute_id=}, {name=}, {tlr=}, {rpc=}, {go=}, {oi=}, {perception=}, {city=}, {state=}, {score=}, {rank=}")
        
        row = [institute_id, name, tlr, rpc, go, oi, perception, city, state, score, rank]
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=all_headers)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    print(f"Total records: {len(df)}")

if __name__ == "__main__":
    parse_nirf_table(HTML_PATH, CSV_PATH) 