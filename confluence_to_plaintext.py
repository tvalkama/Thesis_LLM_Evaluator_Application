"""
Confluence MHTML to Plain Text Converter

Converts documents exported from Confluence (MHTML/DOC format) to plain text,
stripping all metadata, images, styles, and Confluence-specific markup.

Usage:
    python confluence_to_plaintext.py input.doc [output.txt]
    
If output is not specified, prints to stdout.
"""

import sys
import re
import email
import quopri
from html.parser import HTMLParser
from pathlib import Path


class HTMLToTextParser(HTMLParser):
    """Parse HTML and extract text content, ignoring styles, scripts, and images."""
    
    # Block elements that should have line breaks
    BLOCK_ELEMENTS = {
        'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'li', 'tr', 'br', 'hr', 'blockquote', 'pre',
        'section', 'article', 'header', 'footer', 'nav', 'aside'
    }
    
    # Elements to completely ignore (including their content)
    # Note: void elements like 'meta', 'link' are handled separately
    IGNORE_ELEMENTS = {'script', 'style', 'head', 'noscript'}
    
    # Void elements (self-closing, no end tag) - these should be skipped but not tracked for depth
    VOID_ELEMENTS = {'meta', 'link', 'br', 'hr', 'img', 'input', 'area', 'base', 'col', 'embed', 'param', 'source', 'track', 'wbr'}
    
    # Inline elements that might need spacing
    INLINE_SPACING = {'td', 'th'}
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.ignore_depth = 0
        self.current_tag = None
        self.in_list = False
        self.list_depth = 0
        
    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        self.current_tag = tag_lower
        
        # Skip void elements entirely (they don't have end tags)
        if tag_lower in self.VOID_ELEMENTS:
            return
        
        # Track elements whose content should be ignored
        if tag_lower in self.IGNORE_ELEMENTS:
            self.ignore_depth += 1
            return
            
        if self.ignore_depth > 0:
            return
            
        # Handle list items
        if tag_lower in ('ul', 'ol'):
            self.in_list = True
            self.list_depth += 1
            self.text_parts.append('\n')
        elif tag_lower == 'li':
            indent = '  ' * (self.list_depth - 1)
            self.text_parts.append(f'\n{indent}• ')
        # Handle block elements
        elif tag_lower in self.BLOCK_ELEMENTS:
            self.text_parts.append('\n')
            # Add header markers
            if tag_lower in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                level = int(tag_lower[1])
                self.text_parts.append('#' * level + ' ')
        elif tag_lower in self.INLINE_SPACING:
            self.text_parts.append(' ')
            
    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        
        # Skip void elements (they shouldn't have end tags, but some malformed HTML might)
        if tag_lower in self.VOID_ELEMENTS:
            return
        
        if tag_lower in self.IGNORE_ELEMENTS:
            self.ignore_depth = max(0, self.ignore_depth - 1)
            return
            
        if self.ignore_depth > 0:
            return
            
        # Handle list endings
        if tag_lower in ('ul', 'ol'):
            self.list_depth = max(0, self.list_depth - 1)
            if self.list_depth == 0:
                self.in_list = False
            self.text_parts.append('\n')
        elif tag_lower in self.BLOCK_ELEMENTS:
            self.text_parts.append('\n')
        elif tag_lower in self.INLINE_SPACING:
            self.text_parts.append(' ')
            
    def handle_data(self, data):
        if self.ignore_depth > 0:
            return
        self.text_parts.append(data)
        
    def get_text(self) -> str:
        """Get the extracted text, cleaned up."""
        raw_text = ''.join(self.text_parts)
        return clean_text(raw_text)


def clean_text(text: str) -> str:
    """Clean up extracted text by normalizing whitespace and removing artifacts."""
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive whitespace within lines
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Collapse multiple spaces to single space
        line = re.sub(r'[ \t]+', ' ', line)
        # Strip leading/trailing whitespace
        line = line.strip()
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace from entire document
    text = text.strip()
    
    return text


def decode_quoted_printable(content: bytes) -> str:
    """Decode quoted-printable encoded content."""
    try:
        decoded = quopri.decodestring(content)
        # Try UTF-8 first, fall back to latin-1
        try:
            return decoded.decode('utf-8')
        except UnicodeDecodeError:
            return decoded.decode('latin-1', errors='replace')
    except Exception as e:
        # If decoding fails, try to return as string
        if isinstance(content, bytes):
            return content.decode('utf-8', errors='replace')
        return str(content)


def extract_html_from_mhtml(mhtml_content: str) -> str:
    """Extract HTML content from MHTML (multipart MIME) document."""
    
    # Parse as email message (MHTML uses MIME format)
    if isinstance(mhtml_content, str):
        mhtml_content = mhtml_content.encode('utf-8', errors='replace')
    
    msg = email.message_from_bytes(mhtml_content)
    
    # If it's multipart, find the HTML part
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/html':
                payload = part.get_payload(decode=False)
                
                # Check transfer encoding
                transfer_encoding = part.get('Content-Transfer-Encoding', '').lower()
                
                if transfer_encoding == 'quoted-printable':
                    # Payload might be string or bytes
                    if isinstance(payload, str):
                        payload = payload.encode('utf-8')
                    return decode_quoted_printable(payload)
                elif transfer_encoding == 'base64':
                    import base64
                    if isinstance(payload, str):
                        payload = payload.encode('utf-8')
                    return base64.b64decode(payload).decode('utf-8', errors='replace')
                else:
                    if isinstance(payload, bytes):
                        return payload.decode('utf-8', errors='replace')
                    return payload
    else:
        # Single part message
        payload = msg.get_payload(decode=False)
        transfer_encoding = msg.get('Content-Transfer-Encoding', '').lower()
        
        if transfer_encoding == 'quoted-printable':
            if isinstance(payload, str):
                payload = payload.encode('utf-8')
            return decode_quoted_printable(payload)
        
        if isinstance(payload, bytes):
            return payload.decode('utf-8', errors='replace')
        return payload
    
    return ""


def html_to_text(html_content: str) -> str:
    """Convert HTML content to plain text."""
    parser = HTMLToTextParser()
    
    try:
        parser.feed(html_content)
    except Exception as e:
        print(f"Warning: HTML parsing error: {e}", file=sys.stderr)
    
    return parser.get_text()


def remove_confluence_artifacts(text: str) -> str:
    """Remove Confluence-specific artifacts from text."""
    
    # Remove $request.contextPath placeholders
    text = re.sub(r'\$request\.contextPath[^\s]*', '', text)
    
    # Remove "Page:" prefix lines
    text = re.sub(r'^Page:\s*$', '', text, flags=re.MULTILINE)
    
    # Remove empty link artifacts
    text = re.sub(r'\[\s*\]\([^)]*\)', '', text)
    
    # Clean up any remaining double spaces
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


def convert_mhtml_to_text(input_path: str) -> str:
    """
    Convert an MHTML/DOC file exported from Confluence to plain text.
    
    Args:
        input_path: Path to the input MHTML/DOC file
        
    Returns:
        Plain text content
    """
    # Read the file
    with open(input_path, 'rb') as f:
        content = f.read()
    
    # Extract HTML from MHTML
    html_content = extract_html_from_mhtml(content)
    
    if not html_content:
        raise ValueError("Could not extract HTML content from file")
    
    # Convert HTML to text
    text = html_to_text(html_content)
    
    # Remove Confluence-specific artifacts
    text = remove_confluence_artifacts(text)
    
    return text


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: Please provide an input file path", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Validate input file exists
    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Convert
        text = convert_mhtml_to_text(input_path)
        
        # Output
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Converted: {input_path} -> {output_path}")
        else:
            print(text)
            
    except Exception as e:
        print(f"Error converting file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
