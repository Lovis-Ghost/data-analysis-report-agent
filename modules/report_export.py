import io

from docx import Document


def _is_markdown_table_separator(line):
    cleaned_line = line.replace("|", "").replace("-", "").replace(":", "").strip()
    return cleaned_line == ""


def _split_markdown_table_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _add_markdown_table(document, table_lines):
    table_rows = []

    for line in table_lines:
        if _is_markdown_table_separator(line):
            continue

        row_cells = _split_markdown_table_row(line)
        if row_cells:
            table_rows.append(row_cells)

    if not table_rows:
        return

    column_count = max(len(row) for row in table_rows)
    table = document.add_table(rows=len(table_rows), cols=column_count)
    table.style = "Table Grid"

    for row_index, row_cells in enumerate(table_rows):
        for col_index in range(column_count):
            cell_text = row_cells[col_index] if col_index < len(row_cells) else ""
            table.cell(row_index, col_index).text = cell_text


def create_docx_report_bytes(report_text):
    document = Document()
    lines = report_text.splitlines()
    table_lines = []

    def flush_table():
        nonlocal table_lines
        if table_lines:
            _add_markdown_table(document, table_lines)
            table_lines = []

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("|"):
            table_lines.append(stripped_line)
            continue

        flush_table()

        if not stripped_line:
            continue

        if stripped_line.startswith("# "):
            document.add_heading(stripped_line[2:].strip(), level=0)
        elif stripped_line.startswith("## "):
            document.add_heading(stripped_line[3:].strip(), level=1)
        elif stripped_line.startswith("### "):
            document.add_heading(stripped_line[4:].strip(), level=2)
        elif stripped_line.startswith("- "):
            document.add_paragraph(stripped_line[2:].strip(), style="List Bullet")
        else:
            document.add_paragraph(stripped_line)

    flush_table()

    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
