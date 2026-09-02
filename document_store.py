# document_store.py
# Document loading, indexing, and character-position tracking for Verified RAG
# This is CRITICAL - we track exact character positions to make hallucination IMPOSSIBLE

import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib

# For BM25 keyword search (primary search method - LLM reasoning over this)
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("Warning: rank_bm25 not installed. Install with: pip install rank-bm25")

# For PDF support
try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("Warning: PyMuPDF not installed. PDF support disabled. Install with: pip install pymupdf")

# For DOCX support
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("Warning: python-docx not installed. DOCX support disabled.")

from rag_config import get_document_config, get_rag_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    A chunk of text with EXACT character position tracking.
    This is the key to making hallucination impossible - we know exactly 
    where every character came from in the original document.
    """
    text: str
    start_char: int  # Exact start position in original document
    end_char: int    # Exact end position in original document
    document_id: str
    document_name: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Verify positions are valid
        if self.end_char <= self.start_char:
            raise ValueError(f"end_char ({self.end_char}) must be > start_char ({self.start_char})")
        expected_len = self.end_char - self.start_char
        actual_len = len(self.text)
        if actual_len != expected_len:
            # Allow small differences due to text processing
            if abs(actual_len - expected_len) > 5:
                logger.warning(f"Text length mismatch: expected {expected_len}, got {actual_len}")
    
    def contains_position(self, pos: int) -> bool:
        """Check if a character position falls within this chunk"""
        return self.start_char <= pos < self.end_char
    
    def get_text_at_positions(self, start: int, end: int) -> Optional[str]:
        """
        Extract text at given positions IF they fall within this chunk.
        Returns None if positions are outside chunk boundaries.
        """
        if start < self.start_char or end > self.end_char:
            return None
        local_start = start - self.start_char
        local_end = end - self.start_char
        return self.text[local_start:local_end]


@dataclass 
class DocumentRecord:
    """
    Complete record of a loaded document with full text preserved.
    We keep the ENTIRE original text so we can verify ANY quote.
    """
    document_id: str
    name: str
    file_path: str
    full_text: str  # The COMPLETE original text - never modified
    char_count: int
    word_count: int
    chunks: List[TextChunk] = field(default_factory=list)
    load_time: str = ""
    file_hash: str = ""
    
    def __post_init__(self):
        if not self.load_time:
            self.load_time = datetime.now().isoformat()
        if not self.file_hash:
            self.file_hash = hashlib.md5(self.full_text.encode()).hexdigest()
    
    def get_text_at_positions(self, start: int, end: int) -> str:
        """
        THE KEY METHOD: Extract exact text from original document at given positions.
        This is how we verify quotes - we extract from source, not from LLM output.
        """
        if start < 0 or end > self.char_count or start >= end:
            raise ValueError(f"Invalid positions: start={start}, end={end}, doc_length={self.char_count}")
        return self.full_text[start:end]
    
    def find_exact_quote(self, quote: str) -> List[Tuple[int, int]]:
        """
        Find ALL occurrences of an exact quote in the document.
        Returns list of (start, end) positions.
        """
        positions = []
        start = 0
        while True:
            pos = self.full_text.find(quote, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(quote)))
            start = pos + 1
        return positions
    
    def find_fuzzy_quote(self, quote: str, threshold: float = 0.9) -> List[Tuple[int, int, float, str]]:
        """
        Find approximate matches for a quote (for showing user what went wrong).
        Returns list of (start, end, similarity_score, actual_text).
        Uses sliding window approach.
        """
        from difflib import SequenceMatcher
        
        matches = []
        quote_len = len(quote)
        
        # Slide window across document
        for i in range(len(self.full_text) - quote_len + 1):
            window = self.full_text[i:i + quote_len]
            ratio = SequenceMatcher(None, quote.lower(), window.lower()).ratio()
            if ratio >= threshold:
                matches.append((i, i + quote_len, ratio, window))
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:5]  # Return top 5 matches


class DocumentStore:
    """
    The main document storage and retrieval system.
    
    KEY DESIGN PRINCIPLE: Every piece of text we return can be traced back
    to exact character positions in the original document. The LLM CANNOT
    add or modify text - it can only SELECT from what exists.
    """
    
    def __init__(self):
        self.config = get_document_config()
        self.rag_config = get_rag_config()
        self.documents: Dict[str, DocumentRecord] = {}
        self.all_chunks: List[TextChunk] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunk_to_tokens: List[List[str]] = []
        
        logger.info("DocumentStore initialized")
    
    def clear(self):
        """
        Clear all loaded documents and reset the index.
        This method is called when starting a new search session.
        """
        self.documents = {}
        self.all_chunks = []
        self.bm25_index = None
        self.chunk_to_tokens = []
        logger.info("DocumentStore cleared")
    
    def load_document(self, file_path: str) -> DocumentRecord:
        """
        Load a document and create chunks with exact position tracking.
        """
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.config["supported_formats"]:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.config['supported_formats']}")
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.config["max_file_size_mb"]:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB (max: {self.config['max_file_size_mb']}MB)")
        
        # Extract text based on format
        if ext == ".pdf":
            full_text = self._extract_pdf(file_path)
        elif ext == ".docx":
            full_text = self._extract_docx(file_path)
        elif ext == ".html":
            full_text = self._extract_html(file_path)
        else:  # .txt, .md
            with open(file_path, 'r', encoding=self.config["encoding"], errors='ignore') as f:
                full_text = f.read()
        
        # Generate document ID
        doc_id = hashlib.md5(f"{file_path}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        doc_name = os.path.basename(file_path)
        
        # Create document record
        doc = DocumentRecord(
            document_id=doc_id,
            name=doc_name,
            file_path=file_path,
            full_text=full_text,
            char_count=len(full_text),
            word_count=len(full_text.split())
        )
        
        # Create chunks with position tracking
        doc.chunks = self._create_chunks(doc)
        
        # Store document
        self.documents[doc_id] = doc
        
        # Add chunks to global list and rebuild index
        self.all_chunks.extend(doc.chunks)
        self._rebuild_bm25_index()
        
        logger.info(f"Loaded document: {doc_name} ({doc.char_count} chars, {len(doc.chunks)} chunks)")
        return doc
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        if not HAS_PDF:
            raise ImportError("PyMuPDF required for PDF. Install with: pip install pymupdf")
        
        text_parts = []
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        if not HAS_DOCX:
            raise ImportError("python-docx required for DOCX. Install with: pip install python-docx")
        
        doc = DocxDocument(file_path)
        text_parts = [para.text for para in doc.paragraphs]
        return "\n".join(text_parts)
    
    def _extract_html(self, file_path: str) -> str:
        """Extract text from HTML"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("BeautifulSoup required for HTML. Install with: pip install beautifulsoup4")
        
        with open(file_path, 'r', encoding=self.config["encoding"], errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()
        
        return soup.get_text(separator='\n')
    
    def _create_chunks(self, doc: DocumentRecord) -> List[TextChunk]:
        """
        Split document into chunks while preserving EXACT character positions.
        This is critical for quote verification.
        """
        chunks = []
        chunk_size = self.rag_config["chunk_size"]
        overlap = self.rag_config["chunk_overlap"]
        
        text = doc.full_text
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 100 chars of chunk
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i-1] in '.!?\n':
                        end = i
                        break
            
            chunk_text = text[start:end]
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=start,
                end_char=start + len(chunk_text),  # Use actual text length
                document_id=doc.document_id,
                document_name=doc.name,
                chunk_index=chunk_index,
                metadata={"position_ratio": start / max(len(text), 1)}
            ))
            
            chunk_index += 1
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def _rebuild_bm25_index(self):
        """Rebuild the BM25 index after adding documents"""
        if not HAS_BM25:
            logger.warning("BM25 not available - search will be limited")
            return
        
        if not self.all_chunks:
            self.bm25_index = None
            return
        
        # Tokenize all chunks
        self.chunk_to_tokens = []
        for chunk in self.all_chunks:
            tokens = self._tokenize(chunk.text)
            self.chunk_to_tokens.append(tokens)
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(self.chunk_to_tokens)
        logger.info(f"BM25 index rebuilt with {len(self.all_chunks)} chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        return tokens
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[TextChunk, float]]:
        """
        Search for relevant chunks using BM25.
        Returns list of (chunk, score) tuples sorted by relevance.
        """
        if not self.all_chunks:
            return []
        
        if top_k is None:
            top_k = self.rag_config["top_k_chunks"]
        
        if not HAS_BM25 or self.bm25_index is None:
            # Fallback to simple keyword matching
            return self._simple_search(query, top_k)
        
        # BM25 search
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Pair chunks with scores and sort
        chunk_scores = list(zip(self.all_chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum score and return top-k
        min_score = self.rag_config["min_relevance_score"]
        results = [(chunk, score) for chunk, score in chunk_scores[:top_k] if score >= min_score]
        
        return results
    
    def _simple_search(self, query: str, top_k: int) -> List[Tuple[TextChunk, float]]:
        """Fallback search when BM25 is not available"""
        query_words = set(self._tokenize(query))
        
        results = []
        for chunk in self.all_chunks:
            chunk_words = set(self._tokenize(chunk.text))
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                score = overlap / len(query_words) if query_words else 0
                results.append((chunk, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def verify_quote_exact(self, quote: str, document_id: str) -> Tuple[bool, int, int, str]:
        """
        Verify that a quote exists EXACTLY in the specified document.
        Returns (is_valid, start_pos, end_pos, message).
        
        This is THE KEY FUNCTION that makes hallucination impossible:
        - We never trust the LLM's text output
        - We only extract text from the original document at verified positions
        """
        if document_id not in self.documents:
            return False, -1, -1, f"Document {document_id} not found"
        
        doc = self.documents[document_id]
        positions = doc.find_exact_quote(quote)
        
        if positions:
            start, end = positions[0]  # Use first occurrence
            return True, start, end, "Exact match found"
        
        # Try fuzzy matching to help debug
        fuzzy_matches = doc.find_fuzzy_quote(quote, threshold=0.8)
        if fuzzy_matches:
            best = fuzzy_matches[0]
            return False, best[0], best[1], f"No exact match. Closest ({best[2]:.0%}): \"{best[3][:100]}...\""
        
        return False, -1, -1, "Quote not found in document"
    
    def extract_text_at_positions(self, document_id: str, start: int, end: int) -> Tuple[bool, str, str]:
        """
        Extract text from a document at exact positions.
        Returns (success, text, message).
        
        This is used INSTEAD of trusting LLM output - we extract the actual text ourselves.
        """
        if document_id not in self.documents:
            return False, "", f"Document {document_id} not found"
        
        doc = self.documents[document_id]
        
        try:
            text = doc.get_text_at_positions(start, end)
            return True, text, "Text extracted successfully"
        except ValueError as e:
            return False, "", str(e)
    
    def list_documents(self) -> List[Dict]:
        """List all loaded documents"""
        result = []
        for doc_id, doc in self.documents.items():
            result.append({
                "document_id": doc_id,
                "name": doc.name,
                "file_path": doc.file_path,
                "char_count": doc.char_count,
                "word_count": doc.word_count,
                "chunk_count": len(doc.chunks),
                "load_time": doc.load_time
            })
        return result
    
    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Get a document by ID"""
        return self.documents.get(document_id)
    
    def get_all_text(self, document_id: str = None) -> str:
        """Get all text from specified document or all documents"""
        if document_id:
            doc = self.documents.get(document_id)
            return doc.full_text if doc else ""
        
        # Combine all documents
        texts = []
        for doc in self.documents.values():
            texts.append(f"=== {doc.name} ===\n{doc.full_text}")
        return "\n\n".join(texts)


# Testing
if __name__ == "__main__":
    store = DocumentStore()
    
    # Create a test document
    test_text = """This is a test document for the Verified RAG system.
It contains multiple paragraphs to test chunking and quote verification.

The key feature of this system is that quotes are VERIFIED character-by-character.
The LLM cannot hallucinate - it can only select text that actually exists.

This paragraph contains some specific information:
- The system was created in 2024
- It uses LangGraph for orchestration
- It supports multiple document formats

Final paragraph with a conclusion: This verified approach ensures accuracy."""

    # Save test document
    test_path = "/tmp/test_document.txt"
    with open(test_path, 'w') as f:
        f.write(test_text)
    
    # Load and test
    doc = store.load_document(test_path)
    print(f"Loaded: {doc.name}")
    print(f"Chunks: {len(doc.chunks)}")
    
    # Test search
    results = store.search("verified quote character")
    print(f"\nSearch results for 'verified quote character':")
    for chunk, score in results[:3]:
        print(f"  Score {score:.2f}: {chunk.text[:100]}...")
    
    # Test quote verification
    test_quote = "quotes are VERIFIED character-by-character"
    is_valid, start, end, msg = store.verify_quote_exact(test_quote, doc.document_id)
    print(f"\nQuote verification: {is_valid}")
    print(f"Message: {msg}")
    
    # Test extraction
    if is_valid:
        success, text, msg = store.extract_text_at_positions(doc.document_id, start, end)
        print(f"\nExtracted text: '{text}'")
    
    # Test clear
    print(f"\nBefore clear: {len(store.documents)} documents")
    store.clear()
    print(f"After clear: {len(store.documents)} documents")
