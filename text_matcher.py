# text_matcher.py
# Robust text matching for finding LLM-selected quotes in documents
#
# CORE INSIGHT: LLMs are TERRIBLE at counting characters but GOOD at:
# - Copying/quoting text
# - Identifying key phrases
# - Understanding which chunk contains relevant content
#
# This module finds the text the LLM is trying to select using multiple
# strategies, without requiring accurate character positions.

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of attempting to find a quote in text."""
    success: bool
    matched_text: str
    start_position: int
    end_position: int
    match_method: str  # exact, fuzzy, phrase, expanded
    confidence: float  # 0.0 to 1.0
    error_message: str = ""


class TextMatcher:
    """
    Finds text in documents using multiple matching strategies.
    
    Designed to work around LLM limitations - doesn't require accurate
    character positions, instead uses the actual text/quotes the LLM provides.
    """
    
    def __init__(self, fuzzy_threshold: float = 0.8, min_phrase_length: int = 10):
        self.fuzzy_threshold = fuzzy_threshold
        self.min_phrase_length = min_phrase_length
    
    def find_quote_in_chunk(
        self,
        quote_text: str,
        chunk_text: str,
        chunk_start: int,
        key_phrases: List[str] = None,
        expand_to_sentences: bool = True
    ) -> MatchResult:
        """
        Find a quote within a chunk using multiple strategies.
        
        Args:
            quote_text: The text the LLM is trying to select (may be imperfect)
            chunk_text: The full text of the chunk to search in
            chunk_start: The character offset of this chunk in the full document
            key_phrases: Optional distinctive phrases to help locate the text
            expand_to_sentences: Whether to expand match to full sentences
            
        Returns:
            MatchResult with the found text and positions
        """
        if not quote_text or not chunk_text:
            return MatchResult(
                success=False, matched_text="", start_position=0, end_position=0,
                match_method="none", confidence=0.0, error_message="Empty input"
            )
        
        # Clean up the quote text
        quote_clean = self._normalize_text(quote_text)
        chunk_clean = self._normalize_text(chunk_text)
        
        # Strategy 1: Exact match
        # NOTE: exact and normalized matches already return the LLM's complete,
        # verbatim quote, so we do NOT sentence-expand them. Expansion is only
        # used to *complete* the weaker, approximate matches below. This keeps
        # well-copied quotes exactly as the source has them (no heading fragments
        # bleeding in from the previous line, no neighbouring sentence merged in).
        result = self._try_exact_match(quote_text, chunk_text, chunk_start)
        if result.success:
            return result
        
        # Strategy 2: Normalized exact match (ignore whitespace differences)
        result = self._try_normalized_match(quote_clean, chunk_text, chunk_clean, chunk_start)
        if result.success:
            return result
        
        # Strategy 3: Fuzzy match (handles small transcription errors)
        result = self._try_fuzzy_match(quote_text, chunk_text, chunk_start)
        if result.success:
            if expand_to_sentences:
                result = self._expand_to_sentences(result, chunk_text, chunk_start)
            return result
        
        # Strategy 4: Key phrase matching
        if key_phrases:
            result = self._try_phrase_match(key_phrases, chunk_text, chunk_start)
            if result.success:
                if expand_to_sentences:
                    result = self._expand_to_sentences(result, chunk_text, chunk_start)
                return result
        
        # Strategy 5: Longest common substring
        result = self._try_longest_common_substring(quote_text, chunk_text, chunk_start)
        if result.success:
            if expand_to_sentences:
                result = self._expand_to_sentences(result, chunk_text, chunk_start)
            return result
        
        # Strategy 6: Word overlap scoring (find region with most matching words)
        result = self._try_word_overlap_match(quote_text, chunk_text, chunk_start)
        if result.success:
            if expand_to_sentences:
                result = self._expand_to_sentences(result, chunk_text, chunk_start)
            return result
        
        return MatchResult(
            success=False, matched_text="", start_position=0, end_position=0,
            match_method="none", confidence=0.0,
            error_message="Could not find quote using any matching strategy"
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison (collapse whitespace, lowercase)."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def _try_exact_match(self, quote: str, chunk: str, chunk_start: int) -> MatchResult:
        """Try to find exact match of quote in chunk."""
        pos = chunk.find(quote)
        if pos != -1:
            return MatchResult(
                success=True,
                matched_text=quote,
                start_position=chunk_start + pos,
                end_position=chunk_start + pos + len(quote),
                match_method="exact",
                confidence=1.0
            )
        return MatchResult(success=False, matched_text="", start_position=0, 
                          end_position=0, match_method="exact", confidence=0.0)
    
    def _try_normalized_match(self, quote_norm: str, chunk: str, 
                              chunk_norm: str, chunk_start: int) -> MatchResult:
        """Try to find match with normalized whitespace."""
        pos_norm = chunk_norm.find(quote_norm)
        if pos_norm != -1:
            # Map back to original text positions
            # This is approximate but usually works
            original_pos = self._map_normalized_to_original(pos_norm, chunk, chunk_norm)
            quote_len_approx = len(quote_norm)
            
            # Extract the actual text from original
            end_pos = min(original_pos + quote_len_approx + 50, len(chunk))
            extracted = chunk[original_pos:end_pos]
            
            # Find where the normalized quote ends in the extracted text
            for i in range(len(extracted), 0, -1):
                if self._normalize_text(extracted[:i]) == quote_norm:
                    return MatchResult(
                        success=True,
                        matched_text=extracted[:i],
                        start_position=chunk_start + original_pos,
                        end_position=chunk_start + original_pos + i,
                        match_method="normalized",
                        confidence=0.95
                    )
            
            # Fallback: use approximate length
            return MatchResult(
                success=True,
                matched_text=chunk[original_pos:original_pos + quote_len_approx],
                start_position=chunk_start + original_pos,
                end_position=chunk_start + original_pos + quote_len_approx,
                match_method="normalized",
                confidence=0.85
            )
        
        return MatchResult(success=False, matched_text="", start_position=0,
                          end_position=0, match_method="normalized", confidence=0.0)
    
    def _map_normalized_to_original(self, norm_pos: int, original: str, normalized: str) -> int:
        """Map a position in normalized text back to original text."""
        # Walk through original text, counting how many normalized chars we've seen
        norm_count = 0
        in_whitespace = False
        
        for i, char in enumerate(original):
            if norm_count >= norm_pos:
                return i
            
            if char.isspace():
                if not in_whitespace:
                    norm_count += 1  # Collapsed whitespace = 1 char
                    in_whitespace = True
            else:
                norm_count += 1
                in_whitespace = False
        
        return len(original)
    
    def _try_fuzzy_match(self, quote: str, chunk: str, chunk_start: int) -> MatchResult:
        """Try fuzzy matching with sliding window."""
        quote_len = len(quote)
        best_ratio = 0.0
        best_start = 0
        best_text = ""
        
        # Slide a window across the chunk
        window_sizes = [quote_len, int(quote_len * 0.9), int(quote_len * 1.1)]
        
        for window_size in window_sizes:
            if window_size < 20:
                continue
            step = max(1, window_size // 10)  # 10% step for efficiency
            
            for i in range(0, len(chunk) - window_size + 1, step):
                window = chunk[i:i + window_size]
                ratio = SequenceMatcher(None, quote.lower(), window.lower()).ratio()
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i
                    best_text = window
        
        if best_ratio >= self.fuzzy_threshold:
            return MatchResult(
                success=True,
                matched_text=best_text,
                start_position=chunk_start + best_start,
                end_position=chunk_start + best_start + len(best_text),
                match_method="fuzzy",
                confidence=best_ratio
            )
        
        return MatchResult(success=False, matched_text="", start_position=0,
                          end_position=0, match_method="fuzzy", confidence=best_ratio)
    
    def _try_phrase_match(self, phrases: List[str], chunk: str, chunk_start: int) -> MatchResult:
        """Find the region containing the most key phrases."""
        if not phrases:
            return MatchResult(success=False, matched_text="", start_position=0,
                              end_position=0, match_method="phrase", confidence=0.0)
        
        # Find positions of each phrase
        phrase_positions = []
        for phrase in phrases:
            phrase_clean = phrase.strip()
            if len(phrase_clean) < self.min_phrase_length:
                continue
            
            # Try exact match first
            pos = chunk.lower().find(phrase_clean.lower())
            if pos != -1:
                phrase_positions.append((pos, pos + len(phrase_clean), phrase_clean))
        
        if not phrase_positions:
            return MatchResult(success=False, matched_text="", start_position=0,
                              end_position=0, match_method="phrase", confidence=0.0)
        
        # Sort by position
        phrase_positions.sort(key=lambda x: x[0])
        
        # Find the span that contains all found phrases
        min_pos = min(p[0] for p in phrase_positions)
        max_pos = max(p[1] for p in phrase_positions)
        
        # Add some padding
        min_pos = max(0, min_pos - 20)
        max_pos = min(len(chunk), max_pos + 20)
        
        matched_text = chunk[min_pos:max_pos]
        confidence = len(phrase_positions) / len(phrases)
        
        return MatchResult(
            success=True,
            matched_text=matched_text,
            start_position=chunk_start + min_pos,
            end_position=chunk_start + max_pos,
            match_method="phrase",
            confidence=confidence
        )
    
    def _try_longest_common_substring(self, quote: str, chunk: str, 
                                       chunk_start: int) -> MatchResult:
        """Find longest common substring between quote and chunk."""
        if len(quote) < 20:
            return MatchResult(success=False, matched_text="", start_position=0,
                              end_position=0, match_method="lcs", confidence=0.0)
        
        # Use SequenceMatcher to find matching blocks
        matcher = SequenceMatcher(None, quote.lower(), chunk.lower())
        blocks = matcher.get_matching_blocks()
        
        # Find the longest block
        longest_block = max(blocks, key=lambda b: b.size)
        
        if longest_block.size < 30:  # Minimum meaningful match
            return MatchResult(success=False, matched_text="", start_position=0,
                              end_position=0, match_method="lcs", confidence=0.0)
        
        # Extract the matched region from chunk (with some context)
        start = max(0, longest_block.b - 20)
        end = min(len(chunk), longest_block.b + longest_block.size + 20)
        matched_text = chunk[start:end]
        
        confidence = longest_block.size / len(quote)
        
        return MatchResult(
            success=True,
            matched_text=matched_text,
            start_position=chunk_start + start,
            end_position=chunk_start + end,
            match_method="lcs",
            confidence=min(confidence, 0.9)
        )
    
    def _try_word_overlap_match(self, quote: str, chunk: str, chunk_start: int) -> MatchResult:
        """Find region with highest word overlap with quote."""
        # Extract words from quote
        quote_words = set(re.findall(r'\b\w{4,}\b', quote.lower()))
        
        if len(quote_words) < 3:
            return MatchResult(success=False, matched_text="", start_position=0,
                              end_position=0, match_method="word_overlap", confidence=0.0)
        
        # Slide a window and count word overlaps
        window_size = min(len(quote) * 2, 500)
        best_overlap = 0
        best_start = 0
        best_end = 0
        
        for i in range(0, len(chunk) - 100, 50):
            end = min(i + window_size, len(chunk))
            window = chunk[i:end]
            window_words = set(re.findall(r'\b\w{4,}\b', window.lower()))
            
            overlap = len(quote_words & window_words)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_start = i
                best_end = end
        
        if best_overlap >= len(quote_words) * 0.5:  # At least 50% word overlap
            return MatchResult(
                success=True,
                matched_text=chunk[best_start:best_end],
                start_position=chunk_start + best_start,
                end_position=chunk_start + best_end,
                match_method="word_overlap",
                confidence=best_overlap / len(quote_words)
            )
        
        return MatchResult(success=False, matched_text="", start_position=0,
                          end_position=0, match_method="word_overlap", confidence=0.0)
    
    # Abbreviations whose trailing "." must NOT be treated as a sentence end
    # (prevents premature truncation when expanding weak matches).
    _ABBREVIATIONS = {
        "al", "et", "eg", "ie", "vs", "fig", "figs", "no", "nos", "cf", "etc",
        "dr", "mr", "mrs", "ms", "st", "jr", "sr", "vol", "pp", "p", "ca",
        "approx", "ref", "refs", "ed", "eds", "i.e", "e.g",
    }

    def _looks_like_heading_line(self, line: str) -> bool:
        """
        Heuristic: is `line` a heading or a blank line (i.e. a hard boundary the
        quote should not expand across)? A heading is short and does not end like
        a prose sentence. Used so backward expansion never swallows a section
        heading such as 'Results' that sits on the line above the quote.
        """
        s = line.strip()
        if not s:
            return True  # blank line is a boundary
        if len(s) > 80 or len(s.split()) > 9:
            return False
        # Prose lines typically end with sentence/clause punctuation.
        if s[-1] in '.!?;,':
            return False
        return True

    def _is_abbreviation_dot(self, chunk: str, dot_index: int) -> bool:
        """True if the '.' at dot_index belongs to an abbreviation or an initial."""
        # Gather the alphanumeric token immediately preceding the dot.
        j = dot_index - 1
        token_chars = []
        while j >= 0 and (chunk[j].isalnum() or chunk[j] == '.'):
            token_chars.append(chunk[j])
            j -= 1
        token = "".join(reversed(token_chars)).strip('.').lower()
        if not token:
            return False
        if token in self._ABBREVIATIONS:
            return True
        # Single letter -> an initial (e.g. "J." in "Smith J. found")
        if len(token) == 1 and token.isalpha():
            return True
        return False

    def _is_sentence_end(self, chunk: str, i: int) -> bool:
        """True if chunk[i] terminates a sentence (terminator + following space/EOF,
        not part of a decimal number or an abbreviation)."""
        ch = chunk[i]
        if ch not in '.!?':
            return False
        # must be followed by whitespace or end of text
        if i + 1 < len(chunk) and not chunk[i + 1].isspace():
            return False
        if ch == '.':
            # not a decimal point (digit on both sides handled by the next-char
            # check above; here guard the preceding side e.g. "0.")
            if i > 0 and chunk[i - 1].isdigit() and i + 1 < len(chunk) and chunk[i + 1].isdigit():
                return False
            if self._is_abbreviation_dot(chunk, i):
                return False
        return True

    def _expand_to_sentences(self, result: MatchResult, chunk: str,
                             chunk_start: int) -> MatchResult:
        """
        Expand an APPROXIMATE match to the single, complete sentence that the
        match begins in.

        Guarantees that keep this clean and hallucination-safe:
          * The returned text is always a verbatim substring of `chunk` (real
            source text) — we only choose tighter boundaries, never invent text.
          * Backward expansion stops at the previous sentence end, a blank line,
            or a HEADING line — so a section heading on the line above (e.g.
            'Results') is never pulled into the quote.
          * Forward expansion stops at the END of the FIRST sentence — so a
            following sentence (e.g. one that cites another study) is never
            merged in. Decimal points and common abbreviations do not count as
            sentence ends.
        """
        if not result.success:
            return result

        local_start = max(0, min(result.start_position - chunk_start, len(chunk)))
        local_end = max(0, min(result.end_position - chunk_start, len(chunk)))

        # ---- backward to the start of the sentence containing local_start ----
        expanded_start = 0
        i = local_start - 1
        while i >= 0:
            ch = chunk[i]
            if self._is_sentence_end(chunk, i):
                expanded_start = i + 1
                break
            if ch == '\n':
                prev_nl = chunk.rfind('\n', 0, i)
                prev_line = chunk[prev_nl + 1:i] if prev_nl != -1 else chunk[:i]
                if self._looks_like_heading_line(prev_line):
                    expanded_start = i + 1
                    break
                # otherwise a wrapped line of the same sentence — keep scanning
            i -= 1
        # trim leading whitespace
        while expanded_start < len(chunk) and expanded_start < local_end and chunk[expanded_start].isspace():
            expanded_start += 1

        # ---- forward to the end of that FIRST sentence ----
        expanded_end = len(chunk)
        j = max(local_start, expanded_start)
        while j < len(chunk):
            if self._is_sentence_end(chunk, j):
                expanded_end = j + 1
                break
            if chunk[j] == '\n' and j + 1 < len(chunk) and chunk[j + 1] == '\n':
                expanded_end = j
                break
            j += 1

        expanded_text = chunk[expanded_start:expanded_end].strip()
        if not expanded_text:
            # Defensive fallback: keep the original matched region.
            expanded_start = local_start
            expanded_text = chunk[local_start:local_end].strip()

        return MatchResult(
            success=True,
            matched_text=expanded_text,
            start_position=chunk_start + expanded_start,
            end_position=chunk_start + expanded_start + len(expanded_text),
            match_method=result.match_method + "+sentence",
            confidence=result.confidence
        )


def find_quote(quote_text: str, chunk_text: str, chunk_start: int,
               key_phrases: List[str] = None) -> MatchResult:
    """Convenience function to find a quote in a chunk."""
    matcher = TextMatcher()
    return matcher.find_quote_in_chunk(quote_text, chunk_text, chunk_start, key_phrases)
