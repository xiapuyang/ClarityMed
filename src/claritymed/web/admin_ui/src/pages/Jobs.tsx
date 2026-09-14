import {
  ActionIcon,
  Alert,
  Group,
  Loader,
  Stack,
  Table,
  Text,
  Title,
  Tooltip,
} from "@mantine/core";
import { IconX } from "@tabler/icons-react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import { JobStatusChip } from "../components/JobStatusChip";
import {
  cancelAdminJob,
  useAdminJobs,
  type JobSpec,
} from "../hooks/useAdminJobs";

export function Jobs() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useAdminJobs();

  const handleCancel = async (jobId: string) => {
    await cancelAdminJob(jobId);
    await queryClient.invalidateQueries({ queryKey: ["admin", "jobs"] });
  };

  if (isLoading) return <Loader />;
  if (error)
    return <Alert color="red">{(error as Error).message ?? t("app.error")}</Alert>;

  const items = data?.items ?? [];

  return (
    <Stack gap="md">
      <Title order={2}>{t("jobs.title")}</Title>
      {items.length === 0 ? (
        <Text c="dimmed">No jobs yet.</Text>
      ) : (
        <Table striped highlightOnHover withTableBorder>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>ID</Table.Th>
              <Table.Th>Kind</Table.Th>
              <Table.Th>State</Table.Th>
              <Table.Th>Progress</Table.Th>
              <Table.Th>Started</Table.Th>
              <Table.Th> </Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {items.map((j: JobSpec) => (
              <Table.Tr key={j.id}>
                <Table.Td>
                  <Text size="xs" c="dimmed" ff="monospace">
                    {j.id.slice(0, 8)}
                  </Text>
                </Table.Td>
                <Table.Td>{j.kind}</Table.Td>
                <Table.Td>
                  <JobStatusChip state={j.state} />
                </Table.Td>
                <Table.Td>{j.progress || "—"}</Table.Td>
                <Table.Td>
                  {j.started_at
                    ? new Date(j.started_at * 1000).toLocaleTimeString()
                    : "—"}
                </Table.Td>
                <Table.Td>
                  {(j.state === "queued" || j.state === "running") &&
                  j.cancellable ? (
                    <Tooltip label={t("app.cancel")}>
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        onClick={() => handleCancel(j.id)}
                        aria-label={`cancel-${j.id}`}
                      >
                        <IconX size={16} />
                      </ActionIcon>
                    </Tooltip>
                  ) : null}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      )}
      <Group justify="space-between">
        <Group gap="xs">
          {(["queued", "running", "done", "failed", "cancelled", "crashed"] as const).map(
            (s) => (
              <Group key={s} gap={4}>
                <JobStatusChip state={s} />
                <Text size="xs" c="dimmed">
                  {items.filter((j) => j.state === s).length}
                </Text>
              </Group>
            ),
          )}
        </Group>
      </Group>
    </Stack>
  );
}
