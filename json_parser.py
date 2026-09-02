# json_parser.py
# Robust JSON parsing for LLM responses
# Handles: malformed JSON, code blocks, partial responses

import re
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class RobustJSONParser:
    """Ultra-robust JSON parser for LLM responses."""
    
    def __init__(self):
        self.json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'(\{[\s\S]*\})',
            r'(\[[\s\S]*\])',
        ]
    
    def parse(self, text: str) -> Optional[Dict]:
        """Main parsing method - tries multiple strategies."""
        if not text or not text.strip():
            return None
        
        # Strategy 1: Direct parse
        result = self._try_direct_parse(text)
        if result:
            return result
        
        # Strategy 2: Code blocks
        result = self._try_code_block_parse(text)
        if result:
            return result
        
        # Strategy 3: JSON extraction
        result = self._try_json_extraction(text)
        if result:
            return result
        
        # Strategy 4: Fix and retry
        result = self._try_fixed_parse(text)
        if result:
            return result
        
        # Strategy 5: Manual extraction
        result = self._try_manual_extraction(text)
        if result:
            return result
        
        return None
    
    def _try_direct_parse(self, text: str) -> Optional[Dict]:
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
            elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                return {'items': result}
        except json.JSONDecodeError:
            pass
        return None
    
    def _try_code_block_parse(self, text: str) -> Optional[Dict]:
        for pattern in self.json_patterns[:2]:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                result = self._try_direct_parse(match)
                if result:
                    return result
                result = self._try_fixed_parse(match)
                if result:
                    return result
        return None
    
    def _try_json_extraction(self, text: str) -> Optional[Dict]:
        # Find { } boundaries
        start = text.find('{')
        if start == -1:
            return None
        
        # Find matching close brace
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    json_str = text[start:i+1]
                    result = self._try_direct_parse(json_str)
                    if result:
                        return result
                    result = self._try_fixed_parse(json_str)
                    if result:
                        return result
                    break
        return None
    
    def _try_fixed_parse(self, text: str) -> Optional[Dict]:
        # Find JSON boundaries
        start = text.find('{')
        end = text.rfind('}')
        
        if start == -1:
            return None
        # If there is no closing brace at all (e.g. a response truncated mid
        # object), keep everything from the first '{' so the balancing pass
        # below can still recover a usable object.
        if end == -1 or end <= start:
            json_str = text[start:]
        else:
            json_str = text[start:end+1]
        
        # Fix common issues
        # Remove trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Remove comments
        json_str = re.sub(r'//[^\n]*', '', json_str)
        json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
        
        # Fix unquoted keys
        json_str = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)

        # Quote bare/invalid scalar values (e.g. `"sample_size": 14 crewmembers
        # (13 males, ...)` -> `"sample_size": "14 crewmembers (13 males, ...)"`).
        # Done before the trailing-comma pass is re-run because wrapping a value
        # can expose a now-trailing comma.
        json_str = self._quote_bare_scalar_values(json_str)
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        # Close an unterminated string and balance any open brackets/braces.
        # This recovers responses that were cut off mid-value (done_reason=length)
        # or that omitted a closing fence/brace. It is string-aware so it never
        # mistakes a '{' or '[' that lives inside a string for real nesting.
        json_str = self._balance_structure(json_str)
        # Wrapping/closing may again leave a trailing comma before a closer.
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return None

    def _is_valid_json_scalar(self, token: str) -> bool:
        """True if `token` (already trimmed) is a valid standalone JSON scalar:
        a number, true, false, or null. (Strings are handled separately.)"""
        if token in ('true', 'false', 'null'):
            return True
        if token == '':
            return False
        try:
            val = json.loads(token)
        except Exception:
            return False
        return isinstance(val, (int, float))

    def _looks_like_next_key(self, s: str, idx: int) -> bool:
        """Given `idx` positioned just AFTER a top-level comma inside an object,
        return True if what follows is a real `"key":` (a new member) rather than
        more text belonging to the current bare value. Used so a comma embedded in
        an unquoted value (e.g. `24 finishers (...), with no NSAID use`) does not
        prematurely terminate that value."""
        n = len(s)
        j = idx
        while j < n and s[j] in ' \t\r\n':
            j += 1
        if j >= n or s[j] != '"':
            return False
        # Scan the quoted key, honouring escapes.
        j += 1
        while j < n:
            if s[j] == '\\':
                j += 2
                continue
            if s[j] == '"':
                j += 1
                break
            j += 1
        else:
            return False
        while j < n and s[j] in ' \t\r\n':
            j += 1
        return j < n and s[j] == ':'

    def _quote_bare_scalar_values(self, s: str) -> str:
        """Walk a JSON-ish string and wrap any bare (unquoted, non-scalar) value
        that follows a ':' in double quotes, so malformed values like
        `"k": 14 crewmembers (a, b; c)` become `"k": "14 crewmembers (a, b; c)"`.

        Real strings, objects, arrays, and valid scalars (numbers/true/false/
        null) are left untouched. When reading a bare value, parentheses are
        treated as grouping so commas inside `(...)` do not prematurely end the
        value. A top-level comma only ends the value when it is followed by a
        real `"key":` (the next member) or a closing bracket; otherwise the comma
        is treated as part of the value (handles `"k": 24 finishers (...), with
        no NSAID use and proper hydration`). This is a best-effort repair; on any
        anomaly it preserves input.
        """
        out = []
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            # Copy complete string literals verbatim (so their ':'/',' are safe).
            if c == '"':
                j = i + 1
                while j < n:
                    if s[j] == '\\':
                        j += 2
                        continue
                    if s[j] == '"':
                        j += 1
                        break
                    j += 1
                out.append(s[i:j])
                i = j
                continue
            if c != ':':
                out.append(c)
                i += 1
                continue

            # c == ':' — a key/value separator. Emit it and any whitespace.
            out.append(c)
            i += 1
            while i < n and s[i] in ' \t\r\n':
                out.append(s[i])
                i += 1
            if i >= n:
                break
            v = s[i]
            # Strings and containers are left for the normal walk to handle.
            if v in '"{[':
                continue
            # Bare value: read to the terminating top-level ',' '}' or ']',
            # treating (), [], {} and quoted spans as grouping so inner commas
            # don't split. A top-level ',' only terminates if the next member
            # looks like a real key (or the object/array closes).
            vstart = i
            depth = 0
            in_str = False
            while i < n:
                ch = s[i]
                if in_str:
                    if ch == '\\':
                        i += 2
                        continue
                    if ch == '"':
                        in_str = False
                    i += 1
                    continue
                if ch == '"':
                    in_str = True
                    i += 1
                    continue
                if ch in '([{':
                    depth += 1
                    i += 1
                    continue
                if ch in ')]}':
                    if depth == 0:
                        break
                    depth -= 1
                    i += 1
                    continue
                if ch == ',' and depth == 0:
                    # Only a real separator if a new key (or a closer) follows.
                    k = i + 1
                    while k < n and s[k] in ' \t\r\n':
                        k += 1
                    if k >= n or s[k] in '}]' or self._looks_like_next_key(s, i + 1):
                        break
                    # Otherwise the comma belongs to the bare value.
                    i += 1
                    continue
                i += 1
            raw = s[vstart:i]
            core = raw.strip()
            if self._is_valid_json_scalar(core):
                out.append(raw)
            else:
                lead = raw[:len(raw) - len(raw.lstrip())]
                trail = raw[len(raw.rstrip()):]
                # json.dumps escapes embedded quotes/backslashes/newlines safely.
                out.append(lead + json.dumps(core) + trail)
            # Loop continues; the terminating ',' '}' ']' is handled normally.
        return ''.join(out)

    def _balance_structure(self, s: str) -> str:
        """Best-effort close of a truncated/unbalanced JSON fragment. Walks the
        string tracking quote state (with escapes) and a stack of '{' '['. If the
        string ends mid-quote, a closing '"' is appended; then any unclosed
        brackets/braces are closed in the correct (reverse) order. Characters
        inside strings are never counted as structure. Well-formed input is
        returned unchanged."""
        stack = []
        in_str = False
        escape = False
        for ch in s:
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in '{[':
                stack.append(ch)
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
        if not in_str and not stack:
            return s
        tail = []
        if in_str:
            tail.append('"')
        for opener in reversed(stack):
            tail.append('}' if opener == '{' else ']')
        return s + ''.join(tail)
    
    def _try_manual_extraction(self, text: str) -> Optional[Dict]:
        """Manually extract known fields from malformed JSON."""
        result = {}
        
        # Extract arrays
        array_patterns = [
            (r'"?selections"?\s*:\s*\[([\s\S]*?)\]', 'selections'),
            (r'"?search_terms"?\s*:\s*\[([\s\S]*?)\]', 'search_terms'),
        ]
        
        for pattern, key in array_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                content = match.group(1)
                if key == 'selections':
                    result[key] = self._extract_selections(content)
                else:
                    items = re.findall(r'"([^"]*)"', content)
                    if items:
                        result[key] = items
        
        # Extract booleans
        bool_patterns = [
            (r'"?is_exhaustive"?\s*:\s*(true|false)', 'is_exhaustive'),
            (r'"?missed_any"?\s*:\s*(true|false)', 'missed_any'),
        ]
        
        for pattern, key in bool_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[key] = match.group(1).lower() == 'true'
        
        # Extract strings
        string_patterns = [
            (r'"?query_type"?\s*:\s*"([^"]*)"', 'query_type'),
            (r'"?identified_claim"?\s*:\s*"([^"]*)"', 'identified_claim'),
            (r'"?reasoning"?\s*:\s*"([^"]*)"', 'reasoning'),
            (r'"?term_reasoning"?\s*:\s*"([^"]*)"', 'term_reasoning'),
            (r'"?selection_strategy"?\s*:\s*"([^"]*)"', 'selection_strategy'),
        ]
        
        for pattern, key in string_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result[key] = match.group(1)
        
        return result if result else None
    
    def _extract_selections(self, content: str) -> List[Dict]:
        """Extract selection objects from array content."""
        selections = []
        
        depth = 0
        current = ""
        
        for char in content:
            if char == '{':
                depth += 1
                current += char
            elif char == '}':
                depth -= 1
                current += char
                if depth == 0 and current.strip():
                    obj = self._parse_selection(current)
                    if obj:
                        selections.append(obj)
                    current = ""
            elif depth > 0:
                current += char
        
        return selections
    
    def _parse_selection(self, text: str) -> Optional[Dict]:
        """Parse a single selection object."""
        result = {}
        
        patterns = [
            (r'"?chunk_index"?\s*:\s*(\d+)', 'chunk_index', int),
            (r'"?quote_text"?\s*:\s*"([\s\S]*?)"(?=\s*[,}])', 'quote_text', str),
            (r'"?relevance_type"?\s*:\s*"([^"]*)"', 'relevance_type', str),
            (r'"?relevance_reasoning"?\s*:\s*"([^"]*)"', 'relevance_reasoning', str),
        ]
        
        for pattern, key, type_func in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    result[key] = type_func(match.group(1))
                except (ValueError, TypeError):
                    result[key] = match.group(1)
        
        # Extract key_phrases
        phrases_match = re.search(r'"?key_phrases"?\s*:\s*\[(.*?)\]', text, re.IGNORECASE)
        if phrases_match:
            phrases = re.findall(r'"([^"]*)"', phrases_match.group(1))
            result['key_phrases'] = phrases
        
        return result if result else None


def parse_json_response(text: str) -> Optional[Dict]:
    """Convenience function."""
    parser = RobustJSONParser()
    return parser.parse(text)
