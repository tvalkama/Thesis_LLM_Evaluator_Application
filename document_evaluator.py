"""
A Gradio web GUI for document analysis with rubric evaluation using GitHub Models API (GPT-4o)

Evaluates technical IT documentation against a 14-criteria rubric aligned with ISO/IEC 27001:2022.
"""

import os
import json
import time
import subprocess
import hashlib
from datetime import datetime, timezone
import requests
import gradio as gr
import tiktoken
from openai import OpenAI

# Token estimation setup
ENCODING = tiktoken.encoding_for_model("gpt-4o")


def estimate_tokens(text: str) -> int:
    """Estimate token count for GPT-4o model."""
    if not text:
        return 0
    return len(ENCODING.encode(text))


def get_github_token() -> str | None:
    """Get GitHub token from environment variable or GitHub CLI."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


# Configuration
GITHUB_TOKEN = get_github_token()
client = None

# Read rubric (use script directory for relative path)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUBRIC_PATH = os.path.join(SCRIPT_DIR, "rubric.json")
with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
    RUBRIC_JSON = f.read()

# Calculate rubric hash for reproducibility tracking
with open(RUBRIC_PATH, "rb") as f:
    RUBRIC_SHA256 = hashlib.sha256(f.read()).hexdigest()

SYSTEM_PROMPT = """You are an expert evaluator of technical IT documentation quality. You apply
an analytic rubric of 14 criteria to score a single document on a four-level
scale (0 = Not present, 1 = Partial, 2 = Adequate, 3 = Exemplary).

You are evaluating documentation that is part of, or directly supports, an
information security management system aligned with ISO/IEC 27001:2022. The
documents are internal IT artefacts of a Finnish technology company. Treat
all document content as authentic operational documentation.

For each criterion, do the following:
1. Read the four performance-level descriptors for that criterion.
2. Identify the descriptor whose content most closely matches what the
   document shows. Score the document at that level.
3. Decide whether the criterion is applicable to the document at all.
   A criterion is NOT applicable only when it is structurally inapplicable
   to the document type — for example, an access-control coverage criterion
   applied to a glossary or a contact list. If the criterion applies but
   the document fails to address it, that is a score of 0, not N/A. The
   descriptors for criteria 12, 13, and 14 include explicit N/A guidance
   where relevant; otherwise N/A is rarely appropriate.
4. Write a one-to-two-sentence justification that points to specific
   features of the document supporting the score. Cite the document
   directly where possible (e.g. "the document opens with 'Overview'
   section but does not name an audience"). Justifications must be
   grounded in the document content, not in assumptions about
   surrounding tooling, related documents, or institutional context the
   document does not itself reference.

Output strictly the following JSON object and nothing else — no preamble,
no closing remarks, no markdown fences:

{{
  "scores": [
    {{
      "criterion_number": 1,
      "applicable": true,
      "score": 0,
      "justification": "..."
    }},
    ... 14 entries in order, criterion_number 1 through 14
  ]
}}

Rules:
- "score" is an integer 0, 1, 2, or 3 when "applicable" is true.
- "score" is null when "applicable" is false.
- Always return exactly 14 entries in ascending criterion_number order.
- Do not score outside the rubric; do not invent additional criteria.
- Do not award scores for evidence that lies outside the document body
  (such as Confluence approval workflows, page-history metadata, or
  related-document conventions not referenced in the document itself).

The rubric is provided below.

{rubric_json}"""


USER_PROMPT = """Score the following document against all 14 rubric criteria. Return the
JSON object specified in the system prompt. Do not include any text
outside the JSON object.

---DOCUMENT BEGINS---

{document_text}

---DOCUMENT ENDS---"""


# Load rubric data for text report generation
RUBRIC_DATA = json.loads(RUBRIC_JSON)


def get_client():
    """Initialize OpenAI client for GitHub Models API."""
    global client
    if client is None:
        if not GITHUB_TOKEN:
            raise ValueError("No GitHub token. Run 'gh auth login' or set GITHUB_TOKEN.")
        client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=GITHUB_TOKEN)
    return client


def read_uploaded_file(file_obj) -> str | None:
    """Read content from uploaded file."""
    if file_obj is None:
        return None
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def calculate_token_estimate(file_obj) -> str:
    """Calculate and display token estimate for the uploaded document."""
    if file_obj is None:
        return "Upload a file to see token estimate"
    
    content = read_uploaded_file(file_obj)
    if content is None or content.startswith("Error"):
        return content or "No file uploaded"
    
    # Calculate tokens for each component
    system_prompt_formatted = SYSTEM_PROMPT.format(rubric_json=RUBRIC_JSON)
    user_prompt_formatted = USER_PROMPT.format(document_text=content)
    
    system_tokens = estimate_tokens(system_prompt_formatted)
    user_tokens = estimate_tokens(user_prompt_formatted)
    doc_tokens = estimate_tokens(content)
    total_input = system_tokens + user_tokens
    
    # GPT-4o context window is 128k, but GitHub Models may have lower limits
    max_context = 8000  # Conservative estimate for GitHub Models
    
    warning = ""
    if total_input > max_context:
        warning = f"\n\n**Warning:** Total input ({total_input:,} tokens) exceeds recommended limit ({max_context:,} tokens). Consider using a shorter document."
    elif total_input > max_context * 0.8:
        warning = f"\n\n**Note:** Approaching token limit ({total_input:,} / {max_context:,} tokens)"
    
    return f"""### Token Estimate

| Component | Tokens |
|-----------|--------|
| System Prompt + Rubric | {system_tokens:,} |
| Document Content | {doc_tokens:,} |
| User Prompt Overhead | {user_tokens - doc_tokens:,} |
| **Total Input** | **{total_input:,}** |

*Estimated output: ~1,500-2,500 tokens*{warning}"""


def json_to_text_report(scores_data: dict) -> str:
    """Convert JSON scores to a human-readable text report."""
    if "scores" not in scores_data:
        return "Invalid response format: missing 'scores' key"
    
    scores = scores_data["scores"]
    
    # Group labels
    groups = {"A": "Document Governance", "B": "Content Quality", "C": "Compliance"}
    
    # Build report
    lines = ["# Documentation Evaluation Report\n"]
    
    # Summary statistics
    applicable_scores = [s["score"] for s in scores if s.get("applicable", True) and s["score"] is not None]
    total_possible = len(applicable_scores) * 3
    total_achieved = sum(applicable_scores)
    percentage = (total_achieved / total_possible * 100) if total_possible > 0 else 0
    
    lines.append("## Summary\n")
    lines.append(f"- **Total Score:** {total_achieved} / {total_possible} ({percentage:.1f}%)")
    lines.append(f"- **Applicable Criteria:** {len(applicable_scores)} / 14")
    lines.append(f"- **Average Score:** {(total_achieved / len(applicable_scores)):.2f} / 3.00" if applicable_scores else "- **Average Score:** N/A")
    lines.append("")
    
    # Score distribution
    score_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for s in applicable_scores:
        score_counts[s] = score_counts.get(s, 0) + 1
    
    lines.append("### Score Distribution")
    lines.append(f"- Exemplary (3): {score_counts[3]}")
    lines.append(f"- Adequate (2): {score_counts[2]}")
    lines.append(f"- Partial (1): {score_counts[1]}")
    lines.append(f"- Not Present (0): {score_counts[0]}")
    lines.append("")
    
    # Detailed scores by group
    current_group = None
    
    for score_entry in scores:
        criterion_num = score_entry["criterion_number"]
        
        # Find criterion details from rubric
        criterion_info = next((c for c in RUBRIC_DATA["criteria"] if c["number"] == criterion_num), None)
        if criterion_info is None:
            continue
        
        group = criterion_info["group"]
        name = criterion_info["name"]
        
        # Add group header if changed
        if group != current_group:
            current_group = group
            lines.append(f"\n## Group {group}: {groups.get(group, 'Unknown')}\n")
        
        # Score display
        applicable = score_entry.get("applicable", True)
        score = score_entry.get("score")
        justification = score_entry.get("justification", "No justification provided")
        
        if applicable and score is not None:
            score_label = RUBRIC_DATA["scale"].get(str(score), "Unknown")
            score_display = f"**{score}** ({score_label})"
        else:
            score_display = "**N/A** (Not Applicable)"
        
        lines.append(f"### {criterion_num}. {name}")
        lines.append(f"**Score:** {score_display}")
        lines.append(f"\n> {justification}\n")
    
    return "\n".join(lines)


def analyze_document(file_obj):
    """Analyze uploaded document against the rubric."""
    if file_obj is None:
        return "Please upload a document first", "", "", "", ""
    
    content = read_uploaded_file(file_obj)
    if content is None:
        return "Failed to read file", "", "", "", ""
    if content.startswith("Error"):
        return content, "", "", "", ""
    
    try:
        # Start timing
        start_time = time.time()
        
        # Prepare prompts
        system_prompt_formatted = SYSTEM_PROMPT.format(rubric_json=RUBRIC_JSON)
        user_prompt_formatted = USER_PROMPT.format(document_text=content)
        
        # Call API
        response = get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt_formatted},
                {"role": "user", "content": user_prompt_formatted}
            ],
            temperature=0.0,
            max_tokens=2500,
        )
        
        # End timing
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Extract response
        result_text = response.choices[0].message.content.strip()
        
        # Defensive: remove markdown fences if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        # Parse JSON
        scores_data = json.loads(result_text)
        
        # Add metadata for reproducibility
        scores_data["_metadata"] = {
            "rubric_sha256": RUBRIC_SHA256,
            "model": "gpt-4o",
            "temperature": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }
        
        # Format outputs
        json_output = json.dumps(scores_data, indent=2)
        text_report = json_to_text_report(scores_data)
        
        # Token usage
        usage = response.usage
        token_info = f"""### Token Usage

| Metric | Count |
|--------|-------|
| Input Tokens | {usage.prompt_tokens:,} |
| Output Tokens | {usage.completion_tokens:,} |
| **Total Tokens** | **{usage.total_tokens:,}** |"""
        
        # Timing info
        timing_info = f"⏱️ **Analysis completed in {elapsed:.2f} seconds**"
        
        return timing_info, token_info, json_output, text_report, ""
        
    except json.JSONDecodeError as e:
        return f"Failed to parse API response as JSON: {str(e)}", "", result_text, "", ""
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}", "", "", "", ""


def check_rate_limit():
    """Check GitHub Models API rate limit."""
    if not GITHUB_TOKEN:
        return "No GitHub token available"
    
    try:
        response = requests.post(
            "https://models.inference.ai.azure.com/rate_limit",
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Content-Type": "application/json"},
            json={"model": "gpt-4o"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            lines = ["### GitHub Models API Rate Limit\n"]
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"**{key}:**")
                    for k, v in value.items():
                        lines.append(f"  - {k}: `{v}`")
                else:
                    lines.append(f"**{key}:** `{value}`")
            return "\n".join(lines)
        else:
            return f"Failed to get rate limit (HTTP {response.status_code}): {response.text}"
            
    except Exception as e:
        return f"Error: `{type(e).__name__}: {str(e)}`"


# Build Gradio UI
with gr.Blocks(title="Documentation Rubric Evaluator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Documentation Rubric Evaluator")
    gr.Markdown("Evaluate technical IT documentation against a 14-criteria ISO/IEC 27001:2022 rubric using GPT-4o.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Document")
            file_input = gr.File(
                label="Upload plain text file (.txt)",
                file_types=[".txt", ".md", ".log"],
                type="filepath"
            )
            token_estimate = gr.Markdown(value="Upload a file to see token estimate")
            analyze_btn = gr.Button("Analyze Document", variant="primary", size="lg")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            timing_output = gr.Markdown(label="Timing")
            token_usage = gr.Markdown(label="Token Usage")
    
    gr.Markdown("### Analysis Results")
    
    with gr.Tabs():
        with gr.Tab("Text Report"):
            text_output = gr.Markdown(label="Evaluation Report")
        with gr.Tab("JSON Output"):
            json_output = gr.Code(label="Raw JSON Response", language="json", lines=25)
    
    error_output = gr.Markdown(label="Errors", visible=False)
    
    # Event handlers
    file_input.change(
        fn=calculate_token_estimate,
        inputs=[file_input],
        outputs=[token_estimate]
    )
    
    analyze_btn.click(
        fn=analyze_document,
        inputs=[file_input],
        outputs=[timing_output, token_usage, json_output, text_output, error_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("### GitHub Models API")
    
    with gr.Row():
        rate_limit_btn = gr.Button("Check Rate Limit", variant="secondary")
    
    rate_limit_result = gr.Markdown()
    
    rate_limit_btn.click(
        fn=check_rate_limit,
        inputs=[],
        outputs=[rate_limit_result]
    )


if __name__ == "__main__":
    if not GITHUB_TOKEN:
        print("Warning: No GitHub token found!")
        print("   Run 'gh auth login' or set GITHUB_TOKEN environment variable")
    else:
        print("GitHub token found")
    
    print("\nStarting Documentation Rubric Evaluator...")
    app.launch(server_port=5000)
