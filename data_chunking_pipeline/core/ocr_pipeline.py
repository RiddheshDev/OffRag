import fitz
from pathlib import Path
import cv2
from paddleocr import PaddleOCRVL
from markdownify import markdownify


class PDFImageExtractor:

    def __init__(self, dpi=300):
        self.dpi = dpi

    def extract(self, pdf_path, output_dir):

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        pdf = fitz.open(pdf_path)

        zoom = self.dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        pages = []

        for page_idx in range(len(pdf)):

            page = pdf[page_idx]

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image_path = output_dir / f"page_{page_idx+1}.png"

            pix.save(str(image_path))

            pages.append(str(image_path))

        return pages
    
class ImagePreprocessor:

    def preprocess(self, image_path):

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        denoised = cv2.fastNlMeansDenoising(
            gray
        )

        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10
        )
        thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
        return thresh

class DocumentParser:

    def __init__(self):
        # self.pipeline = PPStructureV3(
        #     use_doc_orientation_classify=True,
        #     use_doc_unwarping=True
        # )
        self.pipeline = PaddleOCRVL(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True
        )

    def parse_page(self, image):
        result = self.pipeline.predict(image)

        return result

def build_table_chunk(
    html_table,
    current_section,
    before_context,
    after_context,
    embed_format="markdown"
):
    """
    embed_format:
        'markdown' -> embed markdown table
        'html' -> embed html table
    """

    markdown_table = markdownify(html_table)

    table_content = (
        markdown_table
        if embed_format == "markdown"
        else html_table
    )

    retrieved_text = f"""
Section: {current_section}

Context Before:
{' '.join(before_context)}

Table:
{table_content}

Context After:
{' '.join(after_context)}
""".strip()

    return {
        "type": "table",
        "section": current_section,
        "html": html_table,
        "markdown": markdown_table,
        "retrieved_text": retrieved_text,
        "preceding_context":before_context,
        "succeding_context":after_context
    }

def normalize_data(parsed_data, table_format="markdown"):

    blocks = parsed_data[0]["parsing_res_list"]

    normalized_data = []
    current_section = None

    for idx, block in enumerate(blocks):

        if block.label == "paragraph_title":

            current_section = block.content

            normalized_data.append({
                "type": "text",
                "section": current_section,
                "retrieved_text": block.content
            })

        elif block.label == "text":

            normalized_data.append({
                "type": "text",
                "section": current_section,
                "retrieved_text": block.content
            })

        elif block.label == "table":

            # previous 2 text blocks
            before_context = []

            j = idx - 1
            while j >= 0 and len(before_context) < 2:
                if blocks[j].label in ["text", "paragraph_title"]:
                    before_context.insert(0, blocks[j].content)
                j -= 1

            # next 2 text blocks
            after_context = []

            j = idx + 1
            while j < len(blocks) and len(after_context) < 2:
                if blocks[j].label in ["text", "paragraph_title"]:
                    after_context.append(blocks[j].content)
                j += 1

            html_table = block.content
            table_output =  build_table_chunk(html_table,current_section,before_context,after_context)
            normalized_data.append(table_output)
    return normalized_data