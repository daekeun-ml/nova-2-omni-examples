"""
Document Analysis Module
"""
import concurrent.futures
import fitz
from io import BytesIO
from PIL import Image
from .common import call_nova_model

class DocumentAnalyzer:
    """Document Analysis Class"""
    
    def __init__(self):
        self.prompts = {
            "OCR (텍스트 추출)": """Please perform accurate OCR (Optical Character Recognition) on this document image. 
Extract ALL text exactly as it appears, including:
- Headers, titles, and subtitles
- Body text and paragraphs  
- Lists and bullet points
- Table content and data
- Footer information
- Any small text or fine print

Pay special attention to:
- Korean text accuracy (한글 정확도)
- Proper spacing and line breaks
- Maintaining original text formatting
- Not interpreting or changing the original text

Return the extracted text in clean, readable format with proper markdown structure.""",
            
            "핵심 정보 추출": "Extract key information from this document and return it in structured JSON format with clear categories and values.",
            "표 데이터 추출": "Extract all table data from this document and convert it to structured markdown tables or JSON format.",
            "문서 요약": "Summarize the main content of this document in a clear and organized manner with bullet points and sections."
        }
    
    def analyze_pdf_parallel(self, file_bytes, analysis_option, temperature=0.0, max_tokens=2000, top_p=1.0, 
                           pages_per_batch=3, max_workers=10, progress_callback=None):
        """Analyze PDF pages in parallel batches"""
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(pdf_document)
        
        # Group pages into batches
        page_batches = []
        for i in range(0, total_pages, pages_per_batch):
            batch_pages = []
            for page_num in range(i, min(i + pages_per_batch, total_pages)):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x 해상도
                img_data = pix.tobytes("png")
                batch_pages.append((page_num, img_data))
            page_batches.append(batch_pages)
        
        pdf_document.close()
        
        if progress_callback:
            progress_callback(f"📄 총 {total_pages}페이지를 {len(page_batches)}개 배치로 병렬 처리합니다...")
        
        def process_batch(batch_info):
            batch_idx, batch_pages = batch_info
            batch_results = []
            
            for page_num, img_data in batch_pages:
                content_item = {
                    "image": {
                        "format": "png",
                        "source": {"bytes": img_data}
                    }
                }
                
                messages = [{
                    "role": "user",
                    "content": [
                        content_item,
                        {"text": f"Page {page_num + 1}: {self.prompts[analysis_option]}"}
                    ]
                }]
                
                inference_config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "topP": top_p
                }
                
                response = call_nova_model(messages, inference_config)
                
                if response and "output" in response:
                    page_result = response["output"]["message"]["content"][0]["text"]
                    batch_results.append((page_num, page_result))
            
            return batch_results
        
        # Parallel execution
        all_results = []
        completed_batches = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batches
            future_to_batch = {
                executor.submit(process_batch, (i, batch)): i 
                for i, batch in enumerate(page_batches)
            }
            
            # Gather results
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_results = future.result()
                all_results.extend(batch_results)
                completed_batches += 1
                
                if progress_callback:
                    progress_callback(completed_batches / len(page_batches))
        
        # Sort pages in order
        all_results.sort(key=lambda x: x[0])
        
        # Append results
        final_results = []
        for page_num, page_result in all_results:
            final_results.append(f"## 페이지 {page_num + 1}\n\n{page_result}")
        
        return "\n\n---\n\n".join(final_results)
    
    def analyze_image(self, file_bytes, analysis_option, temperature=0.0, max_tokens=2000, top_p=1.0):
        """Analyze single image document"""
        
        image = Image.open(BytesIO(file_bytes))
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        content_item = {
            "image": {
                "format": "png",
                "source": {"bytes": image_bytes}
            }
        }
        
        messages = [{
            "role": "user",
            "content": [
                content_item,
                {"text": self.prompts[analysis_option]}
            ]
        }]
        
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
            "topP": top_p
        }
        
        response = call_nova_model(messages, inference_config)
        
        if response and "output" in response:
            return response["output"]["message"]["content"][0]["text"]
        
        return None
