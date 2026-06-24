import {
  Alert,
  Anchor,
  Badge,
  Card,
  Grid,
  Group,
  Loader,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";

import { JobStatusChip } from "../components/JobStatusChip";
import { useAdminOverview } from "../hooks/useAdminOverview";

export function Overview() {
  const { t } = useTranslation();
  const { data, isLoading, error } = useAdminOverview();

  if (isLoading) return <Loader />;
  if (error)
    return <Alert color="red">{(error as Error).message ?? t("app.error")}</Alert>;

  return (
    <Stack gap="md">
      <Title order={2}>{t("overview.title")}</Title>
      <Grid>
        <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
          <Card withBorder padding="md">
            <Stack gap={6}>
              <Text size="xs" c="dimmed">
                Providers
              </Text>
              <Title order={2}>{data?.providers.count ?? 0}</Title>
              <Text size="sm" c="dimmed">
                Default: {data?.providers.default ?? "—"}
              </Text>
              <Anchor component={Link} to="/models" size="xs">
                Models →
              </Anchor>
            </Stack>
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
          <Card withBorder padding="md">
            <Stack gap={6}>
              <Text size="xs" c="dimmed">
                Users
              </Text>
              <Title order={2}>{data?.users.count ?? 0}</Title>
              <Text size="sm" c="dimmed">
                {data?.users.admin_count ?? 0} admins
              </Text>
              <Anchor component={Link} to="/users" size="xs">
                Users →
              </Anchor>
            </Stack>
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
          <Card withBorder padding="md">
            <Stack gap={6}>
              <Group justify="space-between">
                <Text size="xs" c="dimmed">
                  Recent jobs
                </Text>
                {data?.recent_jobs.active ? (
                  <Badge color="blue" variant="dot">
                    active
                  </Badge>
                ) : null}
              </Group>
              {(data?.recent_jobs.items ?? []).length === 0 ? (
                <Text c="dimmed">None.</Text>
              ) : (
                <Stack gap={4}>
                  {(data?.recent_jobs.items ?? []).slice(0, 3).map((j) => (
                    <Group key={j.id} gap="xs">
                      <JobStatusChip state={j.state} />
                      <Text size="xs" ff="monospace" truncate>
                        {j.kind}
                      </Text>
                    </Group>
                  ))}
                </Stack>
              )}
              <Anchor component={Link} to="/jobs" size="xs">
                Jobs →
              </Anchor>
            </Stack>
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, sm: 6, md: 3 }}>
          <Card withBorder padding="md">
            <Stack gap={6}>
              <Text size="xs" c="dimmed">
                Servers
              </Text>
              {data?.servers.ready ? (
                (() => {
                  const total = data.servers.nodes.length;
                  const up = data.servers.nodes.filter(
                    (n) => n.status === "up",
                  ).length;
                  const allUp = up === total && total > 0;
                  return (
                    <>
                      <Title order={2} c={allUp ? undefined : "red"}>
                        {up}/{total} up
                      </Title>
                      <Text size="sm" c="dimmed">
                        {allUp
                          ? "all servers reachable"
                          : `${total - up} not reachable`}
                      </Text>
                    </>
                  );
                })()
              ) : (
                <>
                  <Title order={2}>—</Title>
                  <Text size="sm" c="dimmed">
                    Probing…
                  </Text>
                </>
              )}
              <Anchor component={Link} to="/servers" size="xs">
                Servers →
              </Anchor>
            </Stack>
          </Card>
        </Grid.Col>
      </Grid>
      <Card withBorder padding="md">
        <Stack gap={6}>
          <Group justify="space-between">
            <Text fw={500}>Recent audit events</Text>
            <Anchor component={Link} to="/audit" size="xs">
              Audit log →
            </Anchor>
          </Group>
          {(data?.audit_tail.items ?? []).length === 0 ? (
            <Text c="dimmed">No events.</Text>
          ) : (
            <Stack gap={4}>
              {(data?.audit_tail.items ?? []).slice(0, 10).map((ev, i) => (
                <Group key={`${ev.request_id}-${i}`} gap="sm" wrap="nowrap">
                  <Text size="xs" c="dimmed" w={150}>
                    {new Date(ev.created_at).toLocaleTimeString()}
                  </Text>
                  <Text size="sm" ff="monospace">
                    {ev.kind}
                  </Text>
                  <Text size="xs" c="dimmed" truncate>
                    {ev.user_id}
                  </Text>
                </Group>
              ))}
            </Stack>
          )}
        </Stack>
      </Card>
    </Stack>
  );
}
