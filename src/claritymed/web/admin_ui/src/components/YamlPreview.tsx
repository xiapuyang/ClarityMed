import { Code, ScrollArea } from "@mantine/core";

// Minimal YAML-ish read-only renderer. We pretty-print as JSON to avoid
// pulling a JS YAML lib; operators care about structure + values, not
// strict YAML formatting in the preview.
export function YamlPreview({ data }: { data: unknown }) {
  return (
    <ScrollArea h={400}>
      <Code block>{JSON.stringify(data, null, 2)}</Code>
    </ScrollArea>
  );
}
