#!/usr/bin/env python3
"""
GeneReviews NXML Parser

Parses GeneReviews NXML files and extracts sections with metadata.
Designed for RAG indexing pipeline.

Usage:
    python scripts/genereviews_parser.py parse data/gene_NBK1116/hnpcc.nxml
    python scripts/genereviews_parser.py parse-all data/gene_NBK1116.tar.gz --output data/genereviews_chunks.jsonl
"""

import argparse
import json
import re
import sys
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from lxml import etree


@dataclass
class GeneReviewsSection:
    """A parsed section from a GeneReviews article."""
    nbk_id: str  # e.g., "NBK1211"
    shortname: str  # e.g., "hnpcc"
    article_title: str  # e.g., "Lynch Syndrome"
    condition_name: str  # e.g., "Lynch Syndrome" (from metadata, standardized)
    gene_symbols: list[str]  # e.g., ["MLH1", "MSH2", "MSH6", "PMS2"]
    section_id: str  # e.g., "hnpcc.Diagnosis"
    section_title: str  # e.g., "Diagnosis"
    section_path: str  # e.g., "Diagnosis > Suggestive Findings"
    section_type: str  # normalized: diagnosis, clinical_features, management, etc.
    text: str  # plain text content
    chunk_index: int  # 0 for single-chunk sections, 0,1,2... for multi-chunk
    char_count: int
    token_estimate: int
    retired: bool  # True if the chapter is retired/historical (title marker)


# Chunking configuration
MAX_CHUNK_TOKENS = 500  # Target max tokens per chunk
CHUNK_OVERLAP_TOKENS = 50  # Overlap between chunks
CHARS_PER_TOKEN = 4  # Rough estimate


def chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS, overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """
    Split text into chunks at sentence boundaries.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks

    Returns:
        List of text chunks
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    # If text fits in one chunk, return as-is
    if len(text) <= max_chars:
        return [text]

    # Don't split sections containing tables (pipe characters indicate table rows)
    # Tables need to stay together for context
    if " | " in text:
        return [text]

    chunks = []

    # Split into sentences (simple regex - handles most cases)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If single sentence exceeds max, we have to include it anyway
        if sentence_len > max_chars and not current_chunk:
            chunks.append(sentence)
            continue

        # If adding this sentence exceeds max, start new chunk
        if current_length + sentence_len > max_chars and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # Start new chunk with overlap from end of previous
            # Take last few sentences that fit in overlap
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap_chars:
                    overlap_chunk.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break

            current_chunk = overlap_chunk
            current_length = sum(len(s) + 1 for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_len + 1

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


@dataclass
class GeneReviewsMetadata:
    """Metadata for a GeneReviews article from FTP index files."""
    shortname: str
    nbk_id: str
    gene_symbols: list[str]
    condition_name: str


def load_metadata(metadata_path: Path = None) -> dict[str, GeneReviewsMetadata]:
    """
    Load GeneReviews metadata from FTP index file.

    Returns dict mapping shortname -> GeneReviewsMetadata
    """
    if metadata_path is None:
        # Default location
        metadata_path = Path(__file__).parent.parent / "data" / "genereviews" / "GRshortname_NBKid_genesymbol_dzname.txt"

    if not metadata_path.exists():
        print(f"WARNING: Metadata file not found: {metadata_path}", file=sys.stderr)
        return {}

    metadata = {}

    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) < 4:
                continue

            shortname, nbk_id, gene_symbol, condition_name = parts[0], parts[1], parts[2], parts[3]

            # Skip "Not applicable" genes
            if gene_symbol == "Not applicable":
                gene_symbol = None

            if shortname not in metadata:
                metadata[shortname] = GeneReviewsMetadata(
                    shortname=shortname,
                    nbk_id=nbk_id,
                    gene_symbols=[],
                    condition_name=condition_name,
                )

            # Add gene symbol if not already present
            if gene_symbol and gene_symbol not in metadata[shortname].gene_symbols:
                metadata[shortname].gene_symbols.append(gene_symbol)

    print(f"Loaded metadata for {len(metadata)} articles", file=sys.stderr)
    return metadata


# Global metadata cache
_metadata_cache: dict[str, GeneReviewsMetadata] = None


def get_metadata(shortname: str) -> GeneReviewsMetadata | None:
    """Get metadata for an article by shortname."""
    global _metadata_cache
    if _metadata_cache is None:
        _metadata_cache = load_metadata()
    return _metadata_cache.get(shortname)


# Section type normalization mapping
SECTION_TYPE_MAP = {
    "diagnosis": "diagnosis",
    "suggestive_findings": "diagnosis",
    "establishing_the_diagnosis": "diagnosis",
    "clinical_characteristics": "clinical_features",
    "clinical_description": "clinical_features",
    "phenotype": "clinical_features",
    "genotype": "genetics",
    "genotypephenotype": "genetics",
    "penetrance": "genetics",
    "prevalence": "epidemiology",
    "management": "management",
    "treatment": "management",
    "surveillance": "surveillance",
    "prevention": "prevention",
    "genetic_counseling": "genetic_counseling",
    "differential_diagnosis": "differential_diagnosis",
    "molecular_genetics": "molecular_genetics",
    "resources": "resources",
    "references": "references",
    "nomenclature": "other",
}


# Retired GeneReviews chapters carry a marker like
# "... – RETIRED CHAPTER, FOR HISTORICAL REFERENCE ONLY" in the article title.
# The dash before "RETIRED" varies in the source (en-dash, box-drawing char),
# so match on the stable "RETIRED CHAPTER" / "FOR HISTORICAL REFERENCE ONLY"
# text rather than the punctuation.
_RETIRED_MARKER = re.compile(r"retired chapter|for historical reference only", re.IGNORECASE)


def is_retired_title(article_title: str) -> bool:
    """True if an article title marks the chapter as retired/historical."""
    return bool(article_title and _RETIRED_MARKER.search(article_title))


def normalize_section_type(section_id: str) -> str:
    """Map section ID to normalized section type."""
    # Extract the part after the shortname (e.g., "hnpcc.Diagnosis" -> "diagnosis")
    if "." in section_id:
        section_name = section_id.split(".", 1)[1].lower()
    else:
        section_name = section_id.lower()

    # Remove underscores and numbers for matching
    section_name = re.sub(r"[_\d]+", "_", section_name).strip("_")

    for key, value in SECTION_TYPE_MAP.items():
        if key in section_name:
            return value

    return "other"


def extract_table_text(table_el) -> str:
    """Extract table content as formatted text."""
    rows = []

    # Get caption if exists
    caption = table_el.find(".//caption")
    if caption is not None:
        caption_text = "".join(caption.itertext()).strip()
        if caption_text:
            rows.append(f"Table: {caption_text}")
            rows.append("")

    # Get label if exists
    label = table_el.find("label")
    if label is not None:
        label_text = "".join(label.itertext()).strip()
        if label_text and not any(label_text in r for r in rows):
            rows.insert(0, label_text)

    # Find the table element
    table = table_el.find(".//table")
    if table is not None:
        # Process thead and tbody
        for section in [table.find("thead"), table.find("tbody")]:
            if section is None:
                continue
            for tr in section.findall("tr"):
                cells = []
                # Iterate in document order to preserve column structure
                for cell in tr:
                    if cell.tag in ("th", "td"):
                        cell_text = "".join(cell.itertext()).strip()
                        cell_text = re.sub(r"\s+", " ", cell_text)
                        cells.append(cell_text if cell_text else "-")
                if cells:
                    rows.append(" | ".join(cells))

    # If no table element found, try direct tr elements
    if not any(" | " in r for r in rows):
        for tr in table_el.findall(".//tr"):
            cells = []
            for cell in tr:
                if cell.tag in ("th", "td"):
                    cell_text = "".join(cell.itertext()).strip()
                    cell_text = re.sub(r"\s+", " ", cell_text)
                    cells.append(cell_text if cell_text else "-")
            if cells:
                rows.append(" | ".join(cells))

    return "\n".join(rows)


def extract_text(element) -> str:
    """Extract plain text from an XML element, handling nested elements."""
    # Special case: if the element itself is a table-wrap, use extract_table_text
    if element.tag == "table-wrap":
        return extract_table_text(element)

    texts = []

    def walk(el):
        if el.text:
            texts.append(el.text)
        for child in el:
            # Skip certain elements
            if child.tag in ("xref", "ext-link", "sup", "sub"):
                if child.text:
                    texts.append(child.text)
            elif child.tag == "table-wrap":
                # Extract table content with proper formatting
                table_text = extract_table_text(child)
                texts.append(f"\n\n{table_text}\n\n")
            elif child.tag in ("table", "thead", "tbody", "tr", "th", "td"):
                # Skip these - handled by extract_table_text when inside table-wrap
                pass
            elif child.tag == "list":
                texts.append("\n")
                walk(child)
            elif child.tag == "list-item":
                texts.append("• ")
                walk(child)
                texts.append("\n")
            else:
                walk(child)
            if child.tail:
                texts.append(child.tail)

    walk(element)

    # Clean up whitespace
    text = "".join(texts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def extract_title(sec_element) -> str:
    """Extract section title from a sec element."""
    title_el = sec_element.find("title")
    if title_el is not None:
        return extract_text(title_el)
    return ""


def parse_nxml(file_path: Path | str, xml_content: bytes = None) -> Iterator[GeneReviewsSection]:
    """
    Parse a GeneReviews NXML file and yield sections.

    Args:
        file_path: Path to the NXML file (used for metadata)
        xml_content: Optional XML content bytes (if already read from archive)

    Yields:
        GeneReviewsSection objects
    """
    file_path = Path(file_path)
    shortname = file_path.stem  # e.g., "hnpcc"

    if xml_content is None:
        xml_content = file_path.read_bytes()

    # Parse XML
    tree = etree.fromstring(xml_content)

    # Get metadata from FTP index files (more complete gene lists)
    meta = get_metadata(shortname)

    # Extract metadata from XML
    # NBK ID - prefer from metadata file, fallback to XML
    nbk_id = ""
    if meta:
        nbk_id = meta.nbk_id
    else:
        nbk_el = tree.find(".//book-part-id[@book-part-id-type='pmcid']")
        if nbk_el is not None and nbk_el.text:
            nbk_id = f"NBK{nbk_el.text}" if not nbk_el.text.startswith("NBK") else nbk_el.text

    # Article title from XML
    article_title = ""
    title_el = tree.find(".//title-group/title")
    if title_el is not None:
        article_title = extract_text(title_el)

    # Condition name - prefer from metadata (standardized)
    condition_name = meta.condition_name if meta else article_title

    # Retired/historical chapters are marked in the title; flag once here.
    retired = is_retired_title(article_title)

    # Gene symbols - prefer from metadata file (more complete)
    if meta and meta.gene_symbols:
        gene_symbols = meta.gene_symbols
    else:
        # Fallback: extract from XML keywords
        gene_symbols = []
        for kwd in tree.findall(".//kwd-group[@kwd-group-type='gene-symbol']/kwd"):
            if kwd.text:
                gene_symbols.append(kwd.text.strip())

        # Also try to extract from subject keywords
        if not gene_symbols:
            for kwd in tree.findall(".//kwd"):
                text = kwd.text or ""
                # Simple heuristic: uppercase 2-6 letter words that look like gene symbols
                if re.match(r"^[A-Z][A-Z0-9]{1,5}$", text):
                    gene_symbols.append(text)

    # Parse sections
    def process_section(sec_el, parent_path: list[str] = None):
        if parent_path is None:
            parent_path = []

        sec_id = sec_el.get("id", "")
        title = extract_title(sec_el)

        current_path = parent_path + [title] if title else parent_path
        section_path = " > ".join(current_path)

        # Extract text content (excluding nested sections)
        text_parts = []
        for child in sec_el:
            if child.tag == "sec":
                continue  # Skip nested sections, we'll process them separately
            elif child.tag == "title":
                continue  # Skip title, we already have it
            else:
                text_parts.append(extract_text(child))

        text = "\n\n".join(filter(None, text_parts))

        # Only yield if there's meaningful content
        if text and len(text) > 50:
            # Chunk large sections
            chunks = chunk_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                char_count = len(chunk)
                token_estimate = char_count // CHARS_PER_TOKEN

                yield GeneReviewsSection(
                    nbk_id=nbk_id,
                    shortname=shortname,
                    article_title=article_title,
                    condition_name=condition_name,
                    gene_symbols=gene_symbols,
                    section_id=sec_id,
                    section_title=title,
                    section_path=section_path,
                    section_type=normalize_section_type(sec_id),
                    text=chunk,
                    chunk_index=chunk_idx,
                    char_count=char_count,
                    token_estimate=token_estimate,
                    retired=retired,
                )

        # Process nested sections
        for nested_sec in sec_el.findall("sec"):
            yield from process_section(nested_sec, current_path)

    # Find all top-level sections in the body
    for sec in tree.findall(".//body/sec"):
        yield from process_section(sec)


def parse_tarball(tarball_path: Path, output_path: Path = None, quiet: bool = False) -> Iterator[GeneReviewsSection]:
    """
    Parse all NXML files in a GeneReviews tarball.

    Args:
        tarball_path: Path to gene_NBK1116.tar.gz
        output_path: Optional path to write JSONL output
        quiet: If True, suppress per-file output (for use with external progress bar)

    Yields:
        GeneReviewsSection objects
    """
    output_file = None
    if output_path:
        output_file = open(output_path, "w")

    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            nxml_members = [m for m in tar.getmembers() if m.name.endswith(".nxml")]

            for member in nxml_members:
                if not quiet:
                    print(f"Processing: {member.name}", file=sys.stderr)

                f = tar.extractfile(member)
                if f is None:
                    continue

                xml_content = f.read()
                file_path = Path(member.name)

                for section in parse_nxml(file_path, xml_content):
                    if output_file:
                        output_file.write(json.dumps(asdict(section)) + "\n")
                    yield section
    finally:
        if output_file:
            output_file.close()


def main():
    parser = argparse.ArgumentParser(description="Parse GeneReviews NXML files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # parse single file
    parse_cmd = subparsers.add_parser("parse", help="Parse a single NXML file")
    parse_cmd.add_argument("file", type=Path, help="Path to NXML file")
    parse_cmd.add_argument("--json", action="store_true", help="Output as JSON")

    # parse-all from tarball
    parse_all_cmd = subparsers.add_parser("parse-all", help="Parse all NXML files from tarball")
    parse_all_cmd.add_argument("tarball", type=Path, help="Path to gene_NBK1116.tar.gz")
    parse_all_cmd.add_argument("--output", "-o", type=Path, help="Output JSONL file")
    parse_all_cmd.add_argument("--stats", action="store_true", help="Show statistics only")

    args = parser.parse_args()

    if args.command == "parse":
        for section in parse_nxml(args.file):
            if args.json:
                print(json.dumps(asdict(section), indent=2))
            else:
                print(f"\n{'='*60}")
                print(f"Section: {section.section_path}")
                print(f"Type: {section.section_type}")
                print(f"ID: {section.section_id}")
                print(f"Tokens: ~{section.token_estimate}")
                print(f"{'='*60}")
                print(section.text[:500] + "..." if len(section.text) > 500 else section.text)

    elif args.command == "parse-all":
        stats = {
            "total_sections": 0,
            "total_tokens": 0,
            "by_type": {},
            "by_article": {},
        }

        for section in parse_tarball(args.tarball, args.output):
            stats["total_sections"] += 1
            stats["total_tokens"] += section.token_estimate

            # By type
            if section.section_type not in stats["by_type"]:
                stats["by_type"][section.section_type] = {"count": 0, "tokens": 0}
            stats["by_type"][section.section_type]["count"] += 1
            stats["by_type"][section.section_type]["tokens"] += section.token_estimate

            # By article
            if section.article_title not in stats["by_article"]:
                stats["by_article"][section.article_title] = {"count": 0, "tokens": 0}
            stats["by_article"][section.article_title]["count"] += 1
            stats["by_article"][section.article_title]["tokens"] += section.token_estimate

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"PARSING COMPLETE", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Total sections: {stats['total_sections']}", file=sys.stderr)
        print(f"Total tokens: ~{stats['total_tokens']:,}", file=sys.stderr)
        print(f"\nBy section type:", file=sys.stderr)
        for stype, data in sorted(stats["by_type"].items(), key=lambda x: -x[1]["count"]):
            print(f"  {stype}: {data['count']} sections, ~{data['tokens']:,} tokens", file=sys.stderr)
        print(f"\nArticles parsed: {len(stats['by_article'])}", file=sys.stderr)

        if args.output:
            print(f"\nOutput written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
