import { Badge, Group, Stack, Text } from "@mantine/core";

import { YamlPreview } from "./YamlPreview";

interface DiseaseEntry {
  id?: string;
  primary_model_id?: string;
  flow?: string[];
  enabled?: boolean;
  checkpoints?: Record<string, unknown>;
  [k: string]: unknown;
}

// Read-only Versions & Flow renderer for catalogs whose YAML uses the
// `primary_model_id` + `flow: [fallbacks]` shape (vision.yaml,
// medical_clip.yaml, symptoms.yaml). Full drag-to-reorder editing
// will land as a polish PR; for now the panel surfaces the structure
// so an operator can sanity-check what's on disk.
export function VersionsFlowPanel({
  data,
}: {
  data: Record<string, unknown>;
}) {
  const diseases = (data?.diseases ?? data?.entries ?? []) as DiseaseEntry[];

  if (!Array.isArray(diseases) || diseases.length === 0) {
    return (
      <Stack gap="xs">
        <Text c="dimmed">No diseases / versions configured.</Text>
        <YamlPreview data={data} />
      </Stack>
    );
  }

  return (
    <Stack gap="md">
      {diseases.map((d, i) => (
        <Stack key={d.id ?? i} gap={4}>
          <Group gap="xs">
            <Text fw={500}>{d.id ?? `disease ${i + 1}`}</Text>
            {d.enabled === false ? (
              <Badge color="gray">disabled</Badge>
            ) : (
              <Badge color="green">enabled</Badge>
            )}
          </Group>
          {d.primary_model_id ? (
            <Group gap="xs">
              <Text size="xs" c="dimmed">
                primary
              </Text>
              <Badge variant="filled">{d.primary_model_id}</Badge>
            </Group>
          ) : null}
          {Array.isArray(d.flow) && d.flow.length > 0 ? (
            <Group gap={4}>
              <Text size="xs" c="dimmed">
                flow
              </Text>
              {d.flow.map((m, j) => (
                <Badge key={`${m}-${j}`} variant="light">
                  {m}
                </Badge>
              ))}
            </Group>
          ) : null}
        </Stack>
      ))}
    </Stack>
  );
}
