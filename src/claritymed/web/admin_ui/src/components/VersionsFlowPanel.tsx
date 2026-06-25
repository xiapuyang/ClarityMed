import { Badge, Group, Stack, Text } from "@mantine/core";

import { YamlPreview } from "./YamlPreview";

interface DiseaseEntry {
  id?: string;
  primary_model_id?: string;
  flow?: string[];
  enabled?: boolean;
  [k: string]: unknown;
}

interface DatasetEntry {
  id?: string;
  enabled?: boolean;
  model_ids?: string[];
  model_selection?: string;
  [k: string]: unknown;
}

// Read-only Versions & Flow renderer. Three catalog shapes are
// supported because the three model catalogs each chose their own
// top-level key:
//   - vision.yaml      → `diseases:` (primary + fallback flow)
//   - symptoms.yaml    → `datasets:` (model_ids[] with selection mode)
//   - medical_clip.yaml→ `model:` + `tasks:` (single model, many tasks)
// Anything else falls through to the raw YAML preview so the operator
// can still see what's on disk.
export function VersionsFlowPanel({
  data,
}: {
  data: Record<string, unknown>;
}) {
  const diseases = data?.diseases as DiseaseEntry[] | undefined;
  if (Array.isArray(diseases) && diseases.length > 0) {
    return <DiseasesView diseases={diseases} />;
  }

  const datasets = data?.datasets as DatasetEntry[] | undefined;
  if (Array.isArray(datasets) && datasets.length > 0) {
    return <DatasetsView datasets={datasets} />;
  }

  const model = data?.model as Record<string, unknown> | undefined;
  const tasks = data?.tasks as Record<string, unknown> | undefined;
  if (model || tasks) {
    return <SingleModelView model={model} tasks={tasks} server={data?.server as Record<string, unknown> | undefined} />;
  }

  return (
    <Stack gap="xs">
      <Text c="dimmed">No diseases / versions configured.</Text>
      <YamlPreview data={data} />
    </Stack>
  );
}

function DiseasesView({ diseases }: { diseases: DiseaseEntry[] }) {
  return (
    <Stack gap="md">
      {diseases.map((d, i) => (
        <Stack key={d.id ?? i} gap={4}>
          <Group gap="xs">
            <Text fw={500}>{d.id ?? `disease ${i + 1}`}</Text>
            <EnabledBadge enabled={d.enabled} />
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

function DatasetsView({ datasets }: { datasets: DatasetEntry[] }) {
  return (
    <Stack gap="md">
      {datasets.map((d, i) => {
        const models = Array.isArray(d.model_ids) ? d.model_ids : [];
        const [primary, ...fallbacks] = models;
        return (
          <Stack key={d.id ?? i} gap={4}>
            <Group gap="xs">
              <Text fw={500}>{d.id ?? `dataset ${i + 1}`}</Text>
              <EnabledBadge enabled={d.enabled} />
              {d.model_selection ? (
                <Badge color="blue" variant="light">
                  selection: {d.model_selection}
                </Badge>
              ) : null}
            </Group>
            {primary ? (
              <Group gap="xs">
                <Text size="xs" c="dimmed">
                  primary
                </Text>
                <Badge variant="filled">{primary}</Badge>
              </Group>
            ) : null}
            {fallbacks.length > 0 ? (
              <Group gap={4}>
                <Text size="xs" c="dimmed">
                  flow
                </Text>
                {fallbacks.map((m, j) => (
                  <Badge key={`${m}-${j}`} variant="light">
                    {m}
                  </Badge>
                ))}
              </Group>
            ) : null}
          </Stack>
        );
      })}
    </Stack>
  );
}

function SingleModelView({
  model,
  tasks,
  server,
}: {
  model?: Record<string, unknown>;
  tasks?: Record<string, unknown>;
  server?: Record<string, unknown>;
}) {
  const modelId =
    (model?.model_id as string | undefined) ?? "(unspecified)";
  const device = model?.device as string | undefined;
  const revision = model?.revision as string | undefined;
  const baseUrl = server?.base_url as string | undefined;
  const taskNames = tasks ? Object.keys(tasks) : [];

  return (
    <Stack gap="md">
      <Stack gap={4}>
        <Group gap="xs">
          <Text fw={500}>model</Text>
          <Badge variant="filled">{modelId}</Badge>
          {device ? <Badge variant="light">device: {device}</Badge> : null}
          {revision ? (
            <Badge variant="light">rev: {revision}</Badge>
          ) : null}
        </Group>
        {baseUrl ? (
          <Group gap="xs">
            <Text size="xs" c="dimmed">
              server
            </Text>
            <Text size="xs" ff="monospace">
              {baseUrl}
            </Text>
          </Group>
        ) : null}
      </Stack>
      {taskNames.length > 0 ? (
        <Stack gap={4}>
          <Text fw={500}>tasks</Text>
          <Group gap={4}>
            {taskNames.map((name) => (
              <Badge key={name} variant="light">
                {name}
              </Badge>
            ))}
          </Group>
        </Stack>
      ) : null}
    </Stack>
  );
}

function EnabledBadge({ enabled }: { enabled?: boolean }) {
  return enabled === false ? (
    <Badge color="gray">disabled</Badge>
  ) : (
    <Badge color="green">enabled</Badge>
  );
}
