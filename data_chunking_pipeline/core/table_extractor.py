import os
import sys
import pandas as pd
import pdfplumber
from data_chunking_pipeline.utils.helpers import format_markdown_table

class HybridTableExtractor:
    """
    Combines Camelot (high-accuracy lattice/stream) with PdfPlumber crop
    extractors and evaluations to parse structured tabular data.
    """
    @staticmethod
    def extract_table(pdf_path, page_number, bbox):
        """
        Extracts a table at a specific bounding box on a page.
        
        Args:
            pdf_path (str): Path to the PDF.
            page_number (int): 1-indexed page number.
            bbox (tuple): (x0, top, x1, bottom) top-left coordinates.
            
        Returns:
            str: Markdown formatted table representation.
        """
        # Step 1: Detect lines in table candidate bounding box
        has_lines = False
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_number - 1]
                has_lines = HybridTableExtractor._has_lines(page, bbox)
        except Exception as e:
            print(f"[Table Extractor] Line detection failed on page {page_number}: {e}")

        # Step 2: Try Camelot Parser
        df = None
        flavor = "lattice" if has_lines else "stream"
        
        try:
            import camelot
            
            # Convert bbox coordinates from pdfplumber (top-left) to Camelot (bottom-left)
            camelot_bbox = HybridTableExtractor._convert_bbox(pdf_path, page_number, bbox)
            
            print(f"[Table Extractor] Trying Camelot ({flavor}) on page {page_number} bbox: {camelot_bbox}...")
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_number),
                flavor=flavor,
                table_areas=[camelot_bbox]
            )
            
            if len(tables) > 0:
                table = tables[0]
                parsing_report = table.parsing_report
                result = {
                    "df": table.df,
                    "accuracy": parsing_report.get("accuracy", 0.0),
                    "whitespace": parsing_report.get("whitespace", 100.0),
                    "flavor": flavor
                }
                
                # Step 3: Evaluate Camelot quality
                if HybridTableExtractor._is_good_evaluation(result):
                    df = result["df"]
                    print(f"[Table Extractor] Camelot successfully extracted table (accuracy={result['accuracy']}%).")
                else:
                    print(f"[Table Extractor] Camelot accuracy/whitespace failed evaluation rules.")
            else:
                print(f"[Table Extractor] Camelot returned 0 tables on page {page_number}.")
                
        except Exception as e:
            # Catch ghostscript missing, import errors, or general camelot failures
            print(f"[Table Extractor] Camelot extraction failed (possibly missing Ghostscript): {e}")

        # Step 4: Fallback to PdfPlumber Extractor
        if df is None:
            print(f"[Table Extractor] Falling back to pdfplumber text alignment extraction on page {page_number}...")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[page_number - 1]
                    settings = {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text"
                    }
                    table_matrix = page.crop(bbox).extract_table(settings)
                    if table_matrix:
                        df = pd.DataFrame(table_matrix)
                        print("[Table Extractor] pdfplumber fallback succeeded.")
                    else:
                        print("[Table Extractor] pdfplumber fallback returned None.")
            except Exception as ex:
                print(f"[Table Extractor] pdfplumber fallback failed: {ex}")

        # Step 5: Convert DataFrame to Markdown
        if df is not None and not df.empty:
            # Convert pandas DataFrame to list of lists (matrix)
            matrix = df.values.tolist()
            # Clean empty rows
            matrix = [row for row in matrix if any(str(cell).strip() for cell in row)]
            if matrix:
                return format_markdown_table(matrix)
                
        # Return fallback error indicator if all failed
        print(f"[Table Extractor] [Warning] All extraction methods failed on page {page_number}.")
        return "| Column 1 | Column 2 |\n|---|---|\n| [Extraction Error] | [Empty Table] |"

    @staticmethod
    def _has_lines(page, bbox):
        x0, top, x1, bottom = bbox
        count = 0
        for line in page.lines:
            if x0 <= line["x0"] <= x1 and x0 <= line["x1"] <= x1:
                count += 1
        return count > 8

    @staticmethod
    def _convert_bbox(pdf_path, page_number, bbox):
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            h = page.height
            
        x0, top, x1, bottom = bbox
        # Camelot expects coordinates as "x0,y0,x1,y1" relative to bottom-left
        # Standard bottom-left origin mapping: y0 = h - bottom, y1 = h - top
        y0 = h - bottom
        y1 = h - top
        return f"{x0},{y0},{x1},{y1}"

    @staticmethod
    def _is_good_evaluation(result):
        if result is None:
            return False
        if result.get("accuracy", 0.0) < 90.0:
            return False
        if result.get("whitespace", 100.0) > 30.0:
            return False
            
        df = result.get("df")
        if df is None or df.empty:
            return False
        if df.shape[0] < 2 or df.shape[1] < 2:
            return False
        return True
