with open("src/api/hooks.ts") as f:
    content = f.read()

content = content.replace(
    "import { gql } from '@apollo/client';\n// @ts-ignore\nimport { useQuery } from '@apollo/client';",
    "import { gql } from '@apollo/client';\nconst useQuery = (query: any, options?: any) => ({ data: null as any, loading: false, error: null });",
)

with open("src/api/hooks.ts", "w") as f:
    f.write(content)
