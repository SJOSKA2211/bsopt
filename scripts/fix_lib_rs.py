import re

with open("src/math_kernel/rust-core/src/lib.rs") as f:
    content = f.read()

# Replace .as_array() with .as_slice().unwrap() for all 1D arrays
# We can do a regex replacement for variables that are 1D arrays
content = re.sub(r'(\w+)\.as_array\(\)', r'\1.as_slice().unwrap()', content)

# Fix the ambiguity in exp() calls by making literals f64
content = content.replace('(-0.5 * d1 * d1).exp()', '(-0.5_f64 * d1 * d1).exp()')

with open("src/math_kernel/rust-core/src/lib.rs", "w") as f:
    f.write(content)
