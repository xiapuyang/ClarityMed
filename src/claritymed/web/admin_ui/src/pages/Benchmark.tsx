import {
  Alert,
  Badge,
  Button,
  Code,
  Group,
  Loader,
  Modal,
  NumberInput,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { notifications } from "@mantine/notifications";
import { IconPlus } from "@tabler/icons-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  useAdminBenchmarkRuns,
  useTriggerBenchmark,
} from "../hooks/useAdminBenchmark";

export function Benchmark() {
  const { t } = useTranslation();
  const { data, isLoading, error } = useAdminBenchmarkRuns();
  const trigger = useTriggerBenchmark();
  const [opened, { open, close }] = useDisclosure(false);
  const [dataset, setDataset] = useState("");
  const [providerId, setProviderId] = useState("");
  const [sampleSize, setSampleSize] = useState<number | "">(50);

  if (isLoading) return <Loader />;
  if (error)
    return <Alert color="red">{(error as Error).message ?? t("app.error")}</Alert>;

  const submit = async () => {
    try {
      const spec = await trigger.mutateAsync({
        dataset: dataset || undefined,
        provider_id: providerId || undefined,
        sample_size: typeof sampleSize === "number" ? sampleSize : undefined,
      });
      notifications.show({
        title: t("benchmark.title"),
        message: `Job ${spec.id.slice(0, 8)} queued.`,
        color: "blue",
      });
      close();
    } catch (e) {
      notifications.show({
        title: t("app.error"),
        message: (e as Error).message,
        color: "red",
      });
    }
  };

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Title order={2}>{t("benchmark.title")}</Title>
        <Button leftSection={<IconPlus size={16} />} onClick={open}>
          New run
        </Button>
      </Group>
      <Table striped highlightOnHover withTableBorder>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Run ID</Table.Th>
            <Table.Th>State</Table.Th>
            <Table.Th>Summary</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {(data?.items ?? []).map((r) => (
            <Table.Tr key={r.run_id}>
              <Table.Td>
                <Text size="sm" ff="monospace">
                  {r.run_id}
                </Text>
              </Table.Td>
              <Table.Td>
                {r.complete ? (
                  <Badge color="green">complete</Badge>
                ) : (
                  <Badge color="orange">incomplete</Badge>
                )}
              </Table.Td>
              <Table.Td>
                <Code block style={{ maxWidth: 600, overflow: "auto" }}>
                  {r.summary ? JSON.stringify(r.summary).slice(0, 200) : "—"}
                </Code>
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      <Modal opened={opened} onClose={close} title="New benchmark run">
        <Stack gap="sm">
          <TextInput
            autoComplete="off"
            label="Dataset"
            value={dataset}
            onChange={(e) => setDataset(e.currentTarget.value)}
            placeholder="medqa, mmlu_med, ..."
          />
          <TextInput
            autoComplete="off"
            label="Provider ID"
            value={providerId}
            onChange={(e) => setProviderId(e.currentTarget.value)}
            placeholder="e.g. openai-gpt-4o"
          />
          <NumberInput
            autoComplete="off"
            label="Sample size"
            value={sampleSize}
            onChange={(v) => setSampleSize(typeof v === "number" ? v : "")}
            min={1}
          />
          <Group justify="flex-end">
            <Button variant="default" onClick={close}>
              {t("app.cancel")}
            </Button>
            <Button onClick={submit} loading={trigger.isPending}>
              {t("app.confirm")}
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
}
