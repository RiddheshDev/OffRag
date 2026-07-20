import os
import pdfplumber
from data_chunking_pipeline.utils.helpers import format_markdown_table, get_context_lines
from data_chunking_pipeline.core.ocr_pipeline import PDFImageExtractor, ImagePreprocessor, DocumentParser, normalize_data
from data_chunking_pipeline.utils.helpers import load_config
from data_chunking_pipeline.core.table_extractor import HybridTableExtractor
config = load_config()

class IntelligentExtractor:
    """
    Intelligent Extractor that parses text and tables page-by-page.
    Isolates tables, formats them to markdown, and vertical slices text context.
    """
    def __init__(self, table_preceding_lines=5, table_succeeding_lines=5):
        self.table_preceding_lines = table_preceding_lines
        self.table_succeeding_lines = table_succeeding_lines

    def _get_special_blocks(self, page):
        """
        Finds and returns tables and code blocks on the page, sorted by top coordinate.
        """
        blocks = []
        
        # 1. Find tables
        tables = page.find_tables()
        for t in tables:
            blocks.append({
                'type': 'table',
                'bbox': t.bbox,
                'obj': t
            })
            
        # 2. Find monospace code blocks
        mono_chars = []
        for char in page.chars:
            font = char.get("fontname", "").lower()
            if "mono" in font or "courier" in font or "consolas" in font:
                mono_chars.append(char)
                
        if mono_chars:
            # Group close characters vertically
            sorted_mono = sorted(mono_chars, key=lambda c: c["top"])
            char_groups = []
            current_group = [sorted_mono[0]]
            
            for char in sorted_mono[1:]:
                prev = current_group[-1]
                if char["top"] - prev["bottom"] <= 15:
                    current_group.append(char)
                else:
                    char_groups.append(current_group)
                    current_group = [char]
            char_groups.append(current_group)
            
            # Create code block bounding boxes
            for grp in char_groups:
                if len(grp) < 5:  # Filter out small noise groups
                    continue
                x0 = min(c["x0"] for c in grp)
                top = min(c["top"] for c in grp)
                x1 = max(c["x1"] for c in grp)
                bottom = max(c["bottom"] for c in grp)
                
                # Check for overlap with tables
                overlap = False
                for t in tables:
                    tx0, ttop, tx1, tbottom = t.bbox
                    if not (x1 < tx0 or x0 > tx1 or bottom < ttop or top > tbottom):
                        overlap = True
                        break
                if not overlap:
                    blocks.append({
                        'type': 'code',
                        'bbox': (x0, top, x1, bottom),
                        'obj': grp
                    })
                    
        # Sort all blocks by top coordinate
        return sorted(blocks, key=lambda b: b['bbox'][1])

    def extract_document(self, pdf_path, classification="text_heavy_mixed"):
        """
        Parses the PDF and returns a list of pages.
        Each page is represented as:
        {
            'page_number': int,
            'elements': list of dict (each dict has 'type', 'retrieved_text', and optional fields)
        }
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        extracted_pages = []
        if classification == "scanned":
            print("Executing using OCR")
            try:
                pdf_img_path = config.get('pdf_img_path','pdf_2_img')
                image_paths = PDFImageExtractor().extract(pdf_path,pdf_img_path)
                parser = DocumentParser()
                for idx,image_path  in enumerate(image_paths):
                    ## get the image and preprocess it for ocr by removing any noise
                    image = ImagePreprocessor().preprocess(image_path= image_path)
                    ## process the image using paddleocr 
                    parsed_output = parser.parse_page(image)
                    final_output = normalize_data(parsed_output)
                    extracted_pages.append(
                        {"page_number": idx+1,
                            "elements"  : final_output }
                    )
            
            except Exception as e:
                print(f"Failed on page {idx+1}: {e}")

                extracted_pages.append(
                    {
                        "page_number": idx + 1,
                        "elements": [],
                        "error": str(e) }
                )
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                page_width = page.width
                page_height = page.height

                # Find special blocks (tables and code blocks) on page
                special_blocks = self._get_special_blocks(page)

                if not special_blocks:
                    # Case 1: Simple text-only page
                    text = page.extract_text() or ""
                    page_elements = [{
                        'type': 'text',
                        'retrieved_text': text.strip()
                    }]
                else:
                    # Case 2: Page with tables or code blocks -> Slice vertically
                    # Filter out monospace characters from text slices to avoid duplication
                    def is_mono(fontname):
                        if not fontname:
                            return False
                        fontname = fontname.lower()
                        return "mono" in fontname or "courier" in fontname or "consolas" in fontname
                        
                    filtered_page = page.filter(lambda obj: obj.get("object_type") != "char" or not is_mono(obj.get("fontname")))
                    
                    page_elements = []
                    y_start = 0
                    
                    # We will collect text slices and block contents
                    text_slices = []
                    block_contents = []
                    
                    for idx, block in enumerate(special_blocks):
                        bx0, btop, bx1, bbottom = block['bbox']
                        
                        # Guard coordinates
                        btop = max(0.0, min(btop, page_height))
                        bbottom = max(0.0, min(bbottom, page_height))
                        if btop < y_start:
                            btop = y_start
                        if bbottom < btop:
                            bbottom = btop
                            
                        # Crop text slice above the block (from filtered page)
                        slice_text = ""
                        if btop > y_start:
                            try:
                                crop_box = (0, y_start, page_width, btop)
                                cropped = filtered_page.crop(crop_box, relative=False)
                                slice_text = cropped.extract_text() or ""
                            except Exception as e:
                                print(f"[Warning] Failed to crop page {page_num} at y:{y_start}-{btop}: {e}")
                        text_slices.append(slice_text.strip())
                        
                        # Extract block data
                        if block['type'] == 'table':
                            table_markdown = HybridTableExtractor.extract_table(
                                pdf_path=pdf_path,
                                page_number=page_num,
                                bbox=block['bbox']
                            )
                            block_contents.append({
                                'type': 'table',
                                'markdown': table_markdown
                            })
                        else:
                            # Monospace code block: extract with slight padding from original page
                            try:
                                crop_box = (0, max(0.0, btop - 2), page_width, min(page_height, bbottom + 2))
                                cropped_code = page.crop(crop_box, relative=False)
                                code_text = cropped_code.extract_text() or ""
                                code_markdown = f"```json\n{code_text.strip()}\n```"
                                block_contents.append({
                                    'type': 'code',
                                    'markdown': code_markdown
                                })
                            except Exception as e:
                                print(f"[Warning] Failed to crop code block page {page_num}: {e}")
                                block_contents.append({
                                    'type': 'code',
                                    'markdown': "```\n[Error extracting code block]\n```"
                                })
                                
                        y_start = bbottom
                        
                    # Crop final text slice after the last block
                    final_slice_text = ""
                    if y_start < page_height:
                        try:
                            crop_box = (0, y_start, page_width, page_height)
                            cropped = filtered_page.crop(crop_box, relative=False)
                            final_slice_text = cropped.extract_text() or ""
                        except Exception as e:
                            print(f"[Warning] Failed to crop page {page_num} at y:{y_start}-{page_height}: {e}")
                    text_slices.append(final_slice_text.strip())
                    
                    # Interleave text slices and blocks
                    for i in range(len(block_contents)):
                        if text_slices[i]:
                            page_elements.append({
                                'type': 'text',
                                'retrieved_text': text_slices[i]
                            })
                            
                        preceding = get_context_lines(text_slices[i], self.table_preceding_lines, "last")
                        succeeding = get_context_lines(text_slices[i+1], self.table_succeeding_lines, "first")
                        
                        markdown_content = block_contents[i]['markdown']
                        label = "Table" if block_contents[i]['type'] == 'table' else "Code Block"
                        
                        retrieved_text = f"Context Before:\n{preceding}\n\n{label}:\n{markdown_content}\n\nContext After:\n{succeeding}".strip()
                        
                        page_elements.append({
                            'type': block_contents[i]['type'],
                            'retrieved_text': retrieved_text,
                            'preceding_context': preceding,
                            'succeeding_context': succeeding,
                            'markdown': markdown_content
                        })
                        
                    if text_slices[-1]:
                        page_elements.append({
                            'type': 'text',
                            'retrieved_text': text_slices[-1]
                        })

                extracted_pages.append({
                    'page_number': page_num,
                    'elements': page_elements
                })

        return extracted_pages

