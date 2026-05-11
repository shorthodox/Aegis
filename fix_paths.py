import re

# Fix index.html
with open('web/src/pages/index.html', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace('../styles/', '/web/src/styles/')
c = c.replace('../scripts/', '/web/src/scripts/')
c = c.replace('../stores/', '/web/src/stores/')
c = c.replace('../websocket/', '/web/src/websocket/')

# Replace href="something.html" with href="/web/src/pages/something.html"
# But ignore cases that are already absolute or start with #
c = re.sub(r'href="(?!/|http|#)([\w-]+\.html)"', r'href="/web/src/pages/\1"', c)

with open('web/src/pages/index.html', 'w', encoding='utf-8') as f:
    f.write(c)

# Fix dashboard.html
with open('web/src/pages/dashboard.html', 'r', encoding='utf-8') as f:
    c2 = f.read()

c2 = c2.replace('../styles/', '/web/src/styles/')
c2 = c2.replace('../scripts/', '/web/src/scripts/')
c2 = c2.replace('../auth/', '/web/src/auth/')
c2 = re.sub(r'href="(?!/|http|#)([\w-]+\.html)"', r'href="/web/src/pages/\1"', c2)

with open('web/src/pages/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(c2)

print("Done")
