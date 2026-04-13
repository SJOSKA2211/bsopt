with open("src/hooks/useWebSocket.ts") as f:
    content = f.read()

content = content.replace(
    "connectRef.current = connect;", "if (connectRef) connectRef.current = connect;"
)
with open("src/hooks/useWebSocket.ts", "w") as f:
    f.write(content)

with open("src/features/options/components/QuickTradeButton.tsx") as f:
    content = f.read()

content = content.replace(
    "const apiFetch = async (url: string, opts: any) => ({message: 'ok'});",
    "const apiFetch = async <T,>(url: string, opts: any): Promise<T> => ({} as T);",
)
with open("src/features/options/components/QuickTradeButton.tsx", "w") as f:
    f.write(content)