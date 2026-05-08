import os
import re
import glob

base_path = r'd:\Content\Animesh\bots\ai_signal_bot\web'
html_files = glob.glob(os.path.join(base_path, 'src', 'pages', '*.html')) + glob.glob(os.path.join(base_path, '*.html'))

for file_path in html_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix relative imports
    content = re.sub(r'href="\.\./styles/', 'href="/web/src/styles/', content)
    content = re.sub(r'src="\.\./scripts/', 'src="/web/src/scripts/', content)
    content = re.sub(r'href="([^"]*\.html(?:\?[^"]*)?(?:#[^"]*)?)"', lambda m: f'href="/web/src/pages/{m.group(1)}"' if not m.group(1).startswith('http') and not m.group(1).startswith('/web/') else m.group(0), content)
    
    # Remove duplicate/malformed modal fragments (specifically looking for the ones found in dashboard and pricing)
    # They look like <div class="labeled-input">\s*<label>Password \*</label> ... </div> </div>
    malformed_pattern = re.compile(r'\s*<div class="labeled-input">\s*<label>Password \*</label>.*?Secure encrypted connection.*?</div>\s*</div>', re.DOTALL)
    while malformed_pattern.search(content):
        content = malformed_pattern.sub('', content)

    # In dashboard.html, it had an extra </div> after script tags. Let's fix that.
    content = re.sub(r'</script>\s*</div>\s*</body>', '</script>\n</body>', content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("HTML files fixed.")
