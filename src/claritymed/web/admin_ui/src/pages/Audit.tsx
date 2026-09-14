import {
  ActionIcon,
  Alert,
  Button,
  Code,
  Group,
  Loader,
  Pagination,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { IconRefresh } from "@tabler/icons-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { useAdminAudit } from "../hooks/useAdminAudit";

const PAGE_SIZE = 50;
const DEFAULT_ACTOR = "default";

export function Audit() {
  const { t } = useTranslation();
  const [page, setPage] = useState(1);
  const [kindFilter, setKindFilter] = useState("");
  const [actorFilter, setActorFilter] = useState<string | null>(DEFAULT_ACTOR);
  const [requestIdFilter, setRequestIdFilter] = useState("");
  const offset = (page - 1) * PAGE_SIZE;

  const { data, isLoading, error, refetch, isFetching } = useAdminAudit({
    offset,
    limit: PAGE_SIZE,
    kinds: kindFilter
      ? kindFilter
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean)
      : undefined,
    actor: actorFilter || undefined,
    request_id: requestIdFilter.trim() || undefined,
  });

  // Carry forward known actors so the Select keeps its options
  // populated across loading states (react-query returns `undefined`
  // while a refetch is in flight).
  const actorOptions = useMemo(() => {
    const seen = new Set<string>(data?.distinct_actors ?? []);
    if (actorFilter) seen.add(actorFilter);
    return Array.from(seen).sort();
  }, [data?.distinct_actors, actorFilter]);

  const totalPages = data ? Math.ceil(Math.max(data.total_count, 1) / PAGE_SIZE) : 1;

  return (
    <Stack gap="md">
      <Title order={2}>{t("audit.title")}</Title>
      <Group gap="sm" align="end">
        <Select
          label="user"
          value={actorFilter}
          onChange={(value) => {
            setActorFilter(value);
            setPage(1);
          }}
          data={actorOptions}
          searchable
          clearable
          nothingFoundMessage="no actors yet"
          w={200}
        />
        <TextInput
          autoComplete="off"
          label="request_id"
          value={requestIdFilter}
          onChange={(e) => {
            setRequestIdFilter(e.currentTarget.value);
            setPage(1);
          }}
          placeholder="20260607213555D327D87D"
          w={260}
        />
        <TextInput
          autoComplete="off"
          label="kind (comma-sep)"
          value={kindFilter}
          onChange={(e) => {
            setKindFilter(e.currentTarget.value);
            setPage(1);
          }}
          placeholder="admin.config.write,admin.job.completed"
          flex={1}
        />
        <ActionIcon
          variant="default"
          size="lg"
          onClick={() => refetch()}
          loading={isFetching}
          aria-label={t("app.retry")}
        >
          <IconRefresh size={18} />
        </ActionIcon>
      </Group>
      {isLoading ? (
        <Loader />
      ) : error ? (
        <Alert color="red">{(error as Error).message ?? t("app.error")}</Alert>
      ) : (
        <>
          <Table striped highlightOnHover withTableBorder>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>When</Table.Th>
                <Table.Th>User</Table.Th>
                <Table.Th>request_id</Table.Th>
                <Table.Th>Kind</Table.Th>
                <Table.Th>Payload</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {(data?.items ?? []).map((ev, i) => (
                <Table.Tr key={`${ev.request_id}-${i}`}>
                  <Table.Td>
                    <Text size="xs" c="dimmed">
                      {new Date(ev.created_at).toLocaleString()}
                    </Text>
                  </Table.Td>
                  <Table.Td>{ev.user_id}</Table.Td>
                  <Table.Td>
                    <Text size="xs" ff="monospace" c="dimmed">
                      {ev.request_id}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" ff="monospace">
                      {ev.kind}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Code block style={{ maxWidth: 500, overflow: "auto" }}>
                      {JSON.stringify(ev.payload)}
                    </Code>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
          <Group justify="space-between">
            <Text size="xs" c="dimmed">
              {data?.total_count ?? 0} events
            </Text>
            <Pagination
              total={totalPages}
              value={page}
              onChange={setPage}
              size="sm"
            />
          </Group>
          {(data?.items ?? []).length === 0 ? (
            <Button
              variant="subtle"
              onClick={() => {
                setKindFilter("");
                setActorFilter(null);
                setRequestIdFilter("");
                setPage(1);
              }}
            >
              Clear filters
            </Button>
          ) : null}
        </>
      )}
    </Stack>
  );
}
