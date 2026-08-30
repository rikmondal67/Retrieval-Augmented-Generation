import re
import hashlib

H1_PATTERN = re.compile(r'^#{1,2}\s+(.*)')
H2_PATTERN = re.compile(r'^#{3,6}\s+(.*)')
NUMBERED_PATTERN = re.compile(r'^\d+\.\s*\S')

def classify_structure(markdown_text: str):
    structure = []
    current_top_section = "General"

    for raw_line in markdown_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        h1_match = H1_PATTERN.match(line)
        h2_match = H2_PATTERN.match(line)

        if h1_match:
            current_top_section = h1_match.group(1).strip()
            structure.append({
                "section": current_top_section,
                "title": current_top_section,
                "content": ""
            })
            continue

        if h2_match:
            title = h2_match.group(1).strip()
            structure.append({
                "section": current_top_section,
                "title": title,
                "content": ""
            })
            continue

        if NUMBERED_PATTERN.match(line) and len(line) < 100:
            structure.append({
                "section": current_top_section,
                "title": line,
                "content": ""
            })
            continue

        if structure:
            structure[-1]["content"] += " " + line
        else:
            structure.append({
                "section": current_top_section,
                "title": current_top_section,
                "content": line
            })

    return structure


def prepare_chunks(structure):
    chunks = []
    for item in structure:
        content = item["content"].strip()
        if not content and item["title"] == item["section"]:
            continue
        embedding_text = f"[Section: {item['section']} | Title: {item['title']}] {content}"
        chunks.append({
            "section": item["section"], "title": item["title"],
            "raw_text": content, "embedding_text": embedding_text
        })
    return chunks


def make_chunk_id(person_name: str, chunk: dict) -> str:
    raw = f"{person_name}-{chunk['section']}-{chunk['title']}"
    return hashlib.md5(raw.encode()).hexdigest()