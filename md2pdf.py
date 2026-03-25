import markdown
from weasyprint import HTML
import os

md_file = "AAL-SD-Doc/research_summary_and_plan.md"
pdf_file = "AAL-SD-Doc/research_summary_and_plan.pdf"

with open(md_file, "r", encoding="utf-8") as f:
    text = f.read()

# Preprocess image paths for HTML rendering
# Replace relative paths to be absolute or relative to the script execution
text = text.replace("../AAL-SD-Doc/figures/", "file://" + os.path.abspath("AAL-SD-Doc/figures") + "/")

# Convert Markdown to HTML
# Enable extensions like tables, fenced_code
html_content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'codehilite'])

# Add some basic CSS for tables and images
html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.6; padding: 20px; }}
    h1, h2, h3, h4 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    img {{ max-width: 100%; height: auto; }}
    code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
    pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Generate PDF
HTML(string=html_content).write_pdf(pdf_file)
print(f"Successfully generated {pdf_file}")
