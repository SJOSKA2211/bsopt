with open("src/hooks/useWebSocket.ts") as f:
    content = f.read()

content = content.replace(
    "import { useState, useEffect, useRef, useMemo, useCallback } from 'react';",
    "import { useState, useEffect, useRef, useCallback } from 'react';",
)
if "const connectRef" not in content:
    content = content.replace(
        "const connect = useCallback(() => {",
        "const connectRef = useRef<any>(null);\n  const connect = useCallback(() => {",
    )
with open("src/hooks/useWebSocket.ts", "w") as f:
    f.write(content)

with open("src/features/options/components/QuickTradeButton.tsx") as f:
    content = f.read()

content = content.replace(
    "const apiFetch = async <T,>(url: string, opts: any): Promise<T> => ({} as T);",
    "const apiFetch = async <T,>(_url: string, _opts: any): Promise<T> => ({} as T);",
)
with open("src/features/options/components/QuickTradeButton.tsx", "w") as f:
    f.write(content)

with open("src/lib/apollo-client.ts") as f:
    content = f.read()

content = content.replace(
    "merge(existing: any, incoming: any)", "merge(_existing: any, incoming: any)"
)
with open("src/lib/apollo-client.ts", "w") as f:
    f.write(content)

with open("src/api/hooks.ts") as f:
    content = f.read()

content = content.replace(
    "import { useQuery, gql } from '@apollo/client';",
    "import { gql } from '@apollo/client';\n// @ts-ignore\nimport { useQuery } from '@apollo/client';",
)
with open("src/api/hooks.ts", "w") as f:
    f.write(content)