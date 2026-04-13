with open("src/features/options/components/GreeksHeatmap.tsx") as f:
    content = f.read()
content = content.replace(
    "const { data: gqlData, loading: isLoading, error } = useOptionsChain(symbol);",
    "const { data: _gqlData, loading: isLoading, error } = useOptionsChain(symbol);\n  const gqlData: any = _gqlData;",
)
with open("src/features/options/components/GreeksHeatmap.tsx", "w") as f:
    f.write(content)


with open("src/features/options/components/MLPredictions.tsx") as f:
    content = f.read()
content = content.replace(
    "const { data, loading: isLoading, error } = useMLInference(symbol);",
    "const { data: _data, loading: isLoading, error } = useMLInference(symbol);\n  const data: any = _data?.mlPrediction || _data || {};",
)
with open("src/features/options/components/MLPredictions.tsx", "w") as f:
    f.write(content)