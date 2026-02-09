import ast
try:
    with open('tests/mock_all.py', 'r') as f:
        source = f.read()
    ast.parse(source)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax Error in {e.filename} line {e.lineno}, offset {e.offset}:")
    print(e.text)
    print(" " * (e.offset - 1) + "^")
    print(e)
except Exception as e:
    print(f"Error: {e}")
