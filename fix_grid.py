import os
import re

def fix_grid(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Match <Grid item xs={12} md={4}> -> <Grid size={{ xs: 12, md: 4 }}>
    # Match <Grid item xs={12}> -> <Grid size={{ xs: 12 }}>
    def replace_grid(match):
        inner = match.group(1)
        # Extract props like xs={12}, md={4}
        props = re.findall(r'([a-zA-Z]+)=\{?([0-9]+)\}?', inner)

        size_props = []
        for key, value in props:
            if key in ('xs', 'sm', 'md', 'lg', 'xl'):
                size_props.append(f"{key}: {value}")

        if size_props:
            return f'<Grid size={{{{ {", ".join(size_props)} }}}}>'
        return match.group(0) # fallback

    content = re.sub(r'<Grid item\s+([^>]+)>', replace_grid, content)

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath}")

fix_grid('services/frontend/src/pages/dashboard/DashboardPage.tsx')
fix_grid('services/frontend/src/pages/settings/SettingsPage.tsx')
