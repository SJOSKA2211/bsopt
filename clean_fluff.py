import os
import re

# Emojis and other "fluff" to remove
FLUFF_PATTERNS = [
    r'[\u2600-\u26FF\u2700-\u27BF\U0001f300-\U0001f5ff\U0001f600-\U0001f64f\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]', # Emojis
    r'', r'', r'', r'', r'️', r'', r'', r'', r'', r'️', r'', r'', r'', r'', r'️'
]

# ASCII Art patterns (simplistic)
ASCII_ART_PATTERNS = [
    r'/\*', r'\*/', r'=====', r'-----', r'#####', r'\*\*\*\*\*', r'_____'
]

def clean_file(file_path):
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remove emojis
        for pattern in FLUFF_PATTERNS:
            content = re.sub(pattern, '', content)
        
        # Remove repetitive decorative lines (only if they are more than 10 chars of same symbol)
        content = re.sub(r'={10,}', '==', content)
        content = re.sub(r'-{10,}', '--', content)
        content = re.sub(r'#{10,}', '##', content)
        content = re.sub(r'\*{10,}', '**', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
    return False

def main():
    count = 0
    for root, dirs, files in os.walk('.'):
        if any(ignored in root for ignored in ['.git', '.venv', '__pycache__', '.pytest_cache']):
            continue
        for file in files:
            if file.endswith(('.py', '.sh', '.proto', '.yml', '.yaml', '.md', 'Dockerfile')):
                if clean_file(os.path.join(root, file)):
                    count += 1
    print(f"Cleaned {count} files.")

if __name__ == "__main__":
    main()
