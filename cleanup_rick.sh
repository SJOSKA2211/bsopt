#!/bin/bash
set -e

# Define exclusion arguments for grep
EXCLUDES="--exclude-dir=.git --exclude-dir=node_modules --exclude-dir=__pycache__ --exclude-dir=.gemini --exclude-dir=venv --exclude-dir=env --exclude-dir=.venv --exclude-dir=.venv_rick_312 --exclude=cleanup_rick.sh"

echo "Replacing 'Pickle Rick' with 'Joseph Kamau Maina'..."
grep -rIZl "Pickle Rick" . $EXCLUDES | xargs -0 sed -i 's/Pickle Rick/Joseph Kamau Maina/g'

echo "Removing cucumber emoji..."
grep -rIZl "🥒" . $EXCLUDES | xargs -0 sed -i 's/🥒//g'

echo "Replacing 'Wubba Lubba Dub Dub!'..."
grep -rIZl "Wubba Lubba Dub Dub!" . $EXCLUDES | xargs -0 sed -i 's/Wubba Lubba Dub Dub!/System check complete./g'

echo "Replacing 'Morty' with 'The User'..."
grep -rIZl "Morty" . $EXCLUDES | xargs -0 sed -i 's/Morty/The User/g'

echo "Replacing 'God-Mode' with 'Advanced'..."
grep -rIZl "God-Mode" . $EXCLUDES | xargs -0 sed -i 's/God-Mode/Advanced/g'

echo "Replacing 'The God-Mode Financial Manifold'..."
grep -rIZl "The God-Mode Financial Manifold" . $EXCLUDES | xargs -0 sed -i 's/The God-Mode Financial Manifold/High-Performance Financial Engine/g'

echo "Replacing long quote..."
# Use a slightly loose match to be safe or exact match?
# "I'm Pickle Rick! And I'm the only one who actually knows how to scale a derivative pricing engine!"
grep -rIZl "I'm Pickle Rick! And I'm the only one who actually knows how to scale a derivative pricing engine!" . $EXCLUDES | xargs -0 sed -i "s/I'm Pickle Rick! And I'm the only one who actually knows how to scale a derivative pricing engine!/Advanced Derivative Pricing Engine./g"

echo "Replacing footer..."
grep -rIZl "Created by the Pickle Rick Extension. Shut up and compute." . $EXCLUDES | xargs -0 sed -i 's/Created by the Pickle Rick Extension. Shut up and compute./Created by Joseph Kamau Maina./g'

echo "Cleanup complete."
