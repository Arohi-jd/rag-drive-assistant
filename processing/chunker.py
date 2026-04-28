"""
Document chunker for processing PDFs, Word documents, and text files.
"""

import os
import re
import hashlib
from typing import List, Dict
from dotenv import load_dotenv


class DocumentChunker:
    """Chunk documents into smaller text pieces for embedding."""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
        """
        try:
            load_dotenv()
            self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
            self.overlap = overlap or int(os.getenv("CHUNK_OVERLAP", "100"))
            if self.chunk_size <= 0:
                raise ValueError("chunk_size must be greater than 0")
            if self.overlap < 0:
                raise ValueError("overlap must be 0 or greater")
            if self.overlap >= self.chunk_size:
                raise ValueError("overlap must be smaller than chunk_size")

            print(
                f"✅ DocumentChunker initialized with chunk_size={self.chunk_size}, "
                f"overlap={self.overlap}"
            )
        except Exception as e:
            print(f"❌ Error initializing DocumentChunker: {str(e)}")
            raise
    
    def read_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            import fitz

            text = ""
            pdf_document = fitz.open(file_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text += page.get_text() + "\n"
                print(f"📄 Processed page {page_num + 1}/{len(pdf_document)}")
            
            pdf_document.close()
            print(f"✅ Successfully read PDF: {file_path}")
            return text
        except Exception as e:
            print(f"❌ Error reading PDF: {str(e)}")
            raise
    
    def read_docx(self, file_path: str) -> str:
        """Extract text from a Word document."""
        try:
            from docx import Document

            doc = Document(file_path)
            text = ""
            
            for para in doc.paragraphs:
                text += para.text + "\n"
            
            print(f"✅ Successfully read DOCX: {file_path}")
            return text
        except Exception as e:
            print(f"❌ Error reading DOCX: {str(e)}")
            raise
    
    def read_text(self, file_path: str) -> str:
        """Extract text from a plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"✅ Successfully read text file: {file_path}")
            return text
        except Exception as e:
            print(f"❌ Error reading text file: {str(e)}")
            raise
    
    def read_file(self, file_path: str) -> str:
        """Read a file based on its extension."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext == '.pdf':
                return self.read_pdf(file_path)
            elif ext == '.docx':
                return self.read_docx(file_path)
            elif ext in ['.txt', '.md']:
                return self.read_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
            raise

    def _build_doc_id(self, file_path: str) -> str:
        """Create a stable document identifier for metadata."""
        normalized_path = os.path.abspath(file_path)
        return hashlib.sha1(normalized_path.encode('utf-8')).hexdigest()[:16]

    def clean_text(self, text: str) -> str:
        """Normalize extracted text before chunking and embedding."""
        normalized = text.replace('\r\n', '\n').replace('\r', '\n')
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)
        normalized = re.sub(r'[^\w\s\.,;:!\?\-\'"()\[\]/&%\n]', ' ', normalized)
        normalized = re.sub(r'[ \t]+', ' ', normalized)
        normalized = re.sub(r' *\n *', '\n', normalized)
        normalized = re.sub(r'\n{2,}', '\n\n', normalized)
        return normalized.strip()

    def chunk_text(self, text: str, file_path: str) -> List[Dict[str, str]]:
        """
        Split cleaned text into sliding windows using the configured size and overlap.
        """
        try:
            file_name = os.path.basename(file_path)
            doc_id = self._build_doc_id(file_path)
            cleaned_text = self.clean_text(text)

            if not cleaned_text:
                print(f"⚠️  No readable text found in {file_name}")
                return []

            step = self.chunk_size - self.overlap
            chunk_texts: List[str] = []
            start = 0
            text_length = len(cleaned_text)

            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunk_body = cleaned_text[start:end].strip()
                if chunk_body:
                    chunk_texts.append(chunk_body)
                if end >= text_length:
                    break
                start += step

            total_chunks = len(chunk_texts)
            chunks = []
            for chunk_index, chunk_body in enumerate(chunk_texts):
                if not chunk_body:
                    continue
                chunks.append({
                    'text': chunk_body,
                    'doc_id': doc_id,
                    'file_name': file_name,
                    'chunk_index': chunk_index,
                    'total_chunks': total_chunks,
                    'source': file_path,
                    'char_count': len(chunk_body)
                })

            return chunks
        except Exception as e:
            print(f"❌ Error chunking text: {str(e)}")
            raise
    
    def process_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Process a file: read it and chunk the text.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of text chunks with metadata
        """
        try:
            file_name = os.path.basename(file_path)
            print(f"📖 Processing file: {file_name}")
            
            # Read file
            text = self.read_file(file_path)
            
            # Chunk text
            chunks = self.chunk_text(text, file_path)
            
            print(f"✅ Summary for {file_name}: created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            raise
    
    def process_multiple_files(self, file_paths: List[str]) -> List[Dict[str, str]]:
        """Process multiple files and combine chunks."""
        try:
            all_chunks = []
            for file_path in file_paths:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            
            print(f"✅ Successfully processed {len(file_paths)} files with {len(all_chunks)} total chunks")
            return all_chunks
        except Exception as e:
            print(f"❌ Error processing multiple files: {str(e)}")
            raise
