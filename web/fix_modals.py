import os
import re
import glob

base_path = r'd:\Content\Animesh\bots\ai_signal_bot\web'
index_path = os.path.join(base_path, 'src', 'pages', 'index.html')

with open(index_path, 'r', encoding='utf-8') as f:
    index_content = f.read()

start_marker = '<!-- ========== 3‑STEP ONBOARDING MODAL (with password fields) =========='
end_marker = '<!-- ========== TRIAL COUNTDOWN DISPLAY =========='

start_idx = index_content.find(start_marker)
end_idx = index_content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print('Error finding modal in index.html')
    exit(1)

full_modal = index_content[start_idx:end_idx].strip()

html_files = glob.glob(os.path.join(base_path, 'src', 'pages', '*.html')) + glob.glob(os.path.join(base_path, '*.html'))

for file_path in html_files:
    if os.path.normpath(file_path) == os.path.normpath(index_path):
        continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Remove ANY existing modal content to prevent duplicates or malformed HTML
    # We will remove from <!-- ========== 3‑STEP ONBOARDING MODAL down to the last </div> of the modal
    # A safe way is to remove everything from the start_marker up to the </div> immediately preceding <!-- Shared landing.js
    # or just remove the modal entirely by matching the wrapper <div id="loginModal" class="modal">...</div>
    
    # Remove the malformed part from pricing.html:
    content = re.sub(r'<p>Sign in to access the Sovereign Terminal</p>.*?</div>\s*</div>', '', content, flags=re.DOTALL)
    
    # Remove any existing modal start
    content = re.sub(r'<!-- ========== 3‑STEP ONBOARDING MODAL.*?</div>\s*</div>\s*</div>\s*</div>', '', content, flags=re.DOTALL)
    content = re.sub(r'<!-- ========== 3‑STEP ONBOARDING MODAL.*?</div>\s*</div>', '', content, flags=re.DOTALL)
    
    # Clean up any left over <div id="loginModal"...
    content = re.sub(r'<div id="loginModal"[^>]*>.*?</div>\s*</div>\s*</div>\s*</div>', '', content, flags=re.DOTALL)

    # Remove extra newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 2. Insert the full modal just before the landing.js script or </body>
    insert_pattern = re.compile(r'(<!-- Shared landing\.js.*?<script[^>]*src=[\'"].*?landing\.js[\'"][^>]*>.*?</script>)', re.DOTALL)
    
    if not insert_pattern.search(content):
        insert_pattern = re.compile(r'(<script[^>]*src=[\'"].*?landing\.js[\'"][^>]*>.*?</script>)', re.DOTALL)
        
    if not insert_pattern.search(content):
        insert_pattern = re.compile(r'(</body>)', re.IGNORECASE)

    if insert_pattern.search(content):
        # We replace the matched string with full_modal + '\n' + matched string
        new_content = insert_pattern.sub(lambda m: '\n    ' + full_modal.replace('\\', '\\\\') + '\n\n    ' + m.group(1), content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed {os.path.basename(file_path)}')
    else:
        print(f'Skipped {os.path.basename(file_path)} - insertion point not found')
