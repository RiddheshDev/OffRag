import os
import fitz
import pdfplumber

class DocumentClassifier:
    """
    Classifies a PDF document into scanned/OCR, table-heavy, or text-heavy/mixed.
    """
    def __init__(self, ocr_threshold_chars=100, table_density_threshold=0.5):
        """
        Args:
            ocr_threshold_chars (int): If average characters per page is below this,
                                    the document is classified as 'scanned'.
            table_density_threshold (float): Ratio of pages with tables to total pages.
                                            If ratio is >= this, it is 'table_heavy'.
        """
        self.ocr_threshold_chars = ocr_threshold_chars
        self.table_density_threshold = table_density_threshold

    def classify(self, pdf_path):
        """
        Analyzes the PDF and returns: 'scanned', 'table_heavy', or 'text_heavy_mixed'.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        total_pages = 0
        total_text_len = 0
        pages_with_tables = 0
        total_tables = 0

        # 1. Inspect text using PyMuPDF (fitz)
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            for page in doc:
                text = page.get_text()
                total_text_len += len(text.strip())
            doc.close()
        except Exception as e:
            print(f"[Warning] PyMuPDF failed to count text characters: {e}")

        # 2. Inspect tables using pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # If PyMuPDF failed, use pdfplumber for page count
                if total_pages == 0:
                    total_pages = len(pdf.pages)
                for page in pdf.pages:
                    tables = page.find_tables()
                    if tables:
                        pages_with_tables += 1
                        total_tables += len(tables)
        except Exception as e:
            print(f"[Warning] pdfplumber failed to check tables: {e}")

        # Compute metrics
        avg_chars_per_page = total_text_len / total_pages if total_pages > 0 else 0
        table_page_ratio = pages_with_tables / total_pages if total_pages > 0 else 0

        print(f"[Classifier] PDF Metrics:")
        print(f"  - Total Pages: {total_pages}")
        print(f"  - Avg Chars/Page: {avg_chars_per_page:.2f}")
        print(f"  - Pages w/ Tables: {pages_with_tables} ({table_page_ratio*100:.1f}%)")
        print(f"  - Total Tables: {total_tables}")

        # Routing logic
        if total_pages > 0 and avg_chars_per_page < self.ocr_threshold_chars:
            print("[Classifier] Classified as: scanned")
            return "scanned"
        elif total_pages > 0 and (table_page_ratio >= self.table_density_threshold or total_tables >= total_pages):
            print("[Classifier] Classified as: table_heavy")
            return "table_heavy"
        else:
            print("[Classifier] Classified as: text_heavy_mixed")
            return "text_heavy_mixed"
