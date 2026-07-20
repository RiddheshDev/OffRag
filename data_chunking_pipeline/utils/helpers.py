import os
import yaml
import tabulate

def load_config(config_path="data_chunking_pipeline/config.yaml"):
    """
    Loads configuration parameters from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def validate_pdf_path(pdf_path):
    """
    Validates if the provided path is a valid PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Provided path does not exist: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"File is not a PDF: {pdf_path}")
    return True

def format_markdown_table(table_data):
    """
    Converts a grid/table array of rows into a clean markdown table.
    """
    if not table_data or not table_data[0]:
        return ""
    # Clean newlines from cells to avoid breaking markdown formatting
    cleaned_data = []
    for row in table_data:
        cleaned_row = []
        for cell in row:
            val = str(cell or "").strip()
            # Replace newlines with spaces
            val = val.replace("\n", " ").replace("\r", "")
            cleaned_row.append(val)
        cleaned_data.append(cleaned_row)
    
    # Render table using tabulate
    return tabulate.tabulate(cleaned_data, headers="firstrow", tablefmt="github")

def get_context_lines(text, num_lines, direction="last"):
    """
    Extracts the last or first N non-empty lines from the given text block.
    """
    if not text:
        return ""
    # Split and filter out empty lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""
    if direction == "last":
        selected = lines[-num_lines:]
    else:
        selected = lines[:num_lines]
    return "\n".join(selected)
