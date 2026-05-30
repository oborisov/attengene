# GeneReviews Download and Processing

## Overview

- **Source:** NCBI FTP (Literature Archive)
- **Archive:** `gene_NBK1116.tar.gz` (~800MB)
- **Content:** 991 disease chapters in NXML format
- **Text output:** ~57MB concatenated

## Download

```bash
# Get file list
wget ftp://ftp.ncbi.nlm.nih.gov/pub/litarch/file_list.txt

# Download GeneReviews archive (~800MB)
wget ftp://ftp.ncbi.nlm.nih.gov/pub/litarch/ca/84/gene_NBK1116.tar.gz

# Extract
tar -xzf gene_NBK1116.tar.gz
cd gene_NBK1116
```

## Extract Clean Text

```bash
# Create output directory
mkdir -p txt_body

# Extract body text only (removes headers, metadata)
for f in *.nxml; do
    xmllint --xpath 'string(//body)' "$f" 2>/dev/null | \
    sed '/^[[:space:]]*$/d' > "txt_body/${f%.nxml}.txt"
done

# Create filename-to-title mapping
for f in *.nxml; do
    title=$(xmllint --xpath 'string(//title-group/title)' "$f" 2>/dev/null)
    echo -e "${f%.nxml}\t$title"
done > chapter_titles.tsv

# Concatenate all chapters
cat txt_body/*.txt > genereviews_all.txt
```

## Verify

```bash
# Count chapters with content
wc -l txt_body/*.txt | awk '$1 > 100' | wc -l  # ~968

# Check total size
du -h genereviews_all.txt  # ~57MB

# Search example
grep -i "huntington" chapter_titles.tsv
```

## Output

| File | Description |
|------|-------------|
| `txt_body/` | Individual chapter text files |
| `chapter_titles.tsv` | filename → disease name |
| `genereviews_all.txt` | All chapters concatenated |

## Requirements

```bash
sudo apt install libxml2-utils  # for xmllint
```
