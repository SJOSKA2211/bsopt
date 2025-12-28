import os
import re

def convert_assertions(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add import if not present
    if 'from tests.test_utils import assert_equal' not in content:
        import_line = 'from tests.test_utils import assert_equal\n'
        # Try to insert after other imports
        lines = content.split('\n')
        inserted = False
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                continue
            elif i > 0:
                lines.insert(i, import_line)
                inserted = True
                break
        if not inserted:
            lines.insert(0, import_line)
        content = '\n'.join(lines)

    # Basic assert actual == expected -> assert_equal(actual, expected)
    content = re.sub(r'assert\s+(.*?)\s*==\s*(.*)', r'assert_equal(\1, \2)', content)
    
    # Basic assert actual != expected -> assert actual != expected (leave as is or implement assert_not_equal) 
    
    # assert abs(a - b) < tol -> assert_equal(a, b, tolerance=tol)
    content = re.sub(r'assert\s+abs\((.*?)\s*-\s*(.*?)\)\s*<\s*(.*)', r'assert_equal(\1, \2, tolerance=\3)', content)

    with open(file_path, 'w') as f:
        f.write(content)

test_files = [f for f in os.listdir('tests') if f.startswith('test_') and f.endswith('.py')]
for f in test_files:
    print(f'Converting {f}...')
    convert_assertions(os.path.join('tests', f))
