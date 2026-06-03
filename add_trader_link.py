import os
import glob

pages_dir = 'web/src/pages'
html_files = glob.glob(os.path.join(pages_dir, '*.html'))

for file in html_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Nav link desktop
    content = content.replace(
        '<a href="/web/src/pages/track-record.html" class="nav-link">Track Record</a>',
        '<a href="/web/src/pages/track-record.html" class="nav-link">Live Record</a>\n            <a href="/web/src/pages/trader-track-record.html" class="nav-link">Trader Record</a>'
    )
    
    # Nav link mobile
    content = content.replace(
        '<a href="/web/src/pages/track-record.html" class="nav-mobile-link"><i class="fas fa-chart-bar"></i> Track Record</a>',
        '<a href="/web/src/pages/track-record.html" class="nav-mobile-link"><i class="fas fa-chart-bar"></i> Live Record</a>\n        <a href="/web/src/pages/trader-track-record.html" class="nav-mobile-link"><i class="fas fa-chart-line"></i> Trader Record</a>'
    )

    # Specific for active pages
    content = content.replace(
        '<a href="/web/src/pages/track-record.html" class="nav-link active">Track Record</a>',
        '<a href="/web/src/pages/track-record.html" class="nav-link active">Live Record</a>\n            <a href="/web/src/pages/trader-track-record.html" class="nav-link">Trader Record</a>'
    )
    content = content.replace(
        '<a href="/web/src/pages/track-record.html" class="nav-mobile-link mobile-active"><i class="fas fa-chart-bar"></i> Track Record</a>',
        '<a href="/web/src/pages/track-record.html" class="nav-mobile-link mobile-active"><i class="fas fa-chart-bar"></i> Live Record</a>\n        <a href="/web/src/pages/trader-track-record.html" class="nav-mobile-link"><i class="fas fa-chart-line"></i> Trader Record</a>'
    )

    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)

print(f"Updated {len(html_files)} files.")
