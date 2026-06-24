import {
  Alert,
  Badge,
  Card,
  Grid,
  Group,
  Loader,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { useTranslation } from "react-i18next";

import { ServersGraph } from "../components/ServersGraph";
import { useAdminServers, type ServerStatus } from "../hooks/useAdminServers";

const STATUS_COLOR: Record<ServerStatus, string> = {
  up: "green",
  down: "red",
  timeout: "yellow",
  "n/a": "gray",
};

export function Servers() {
  const { t } = useTranslation();
  const { data, isLoading, error } = useAdminServers();
  if (isLoading) return <Loader />;
  if (error)
    return <Alert color="red">{(error as Error).message ?? t("app.error")}</Alert>;

  const processNodes = (data?.nodes ?? []).filter((n) => n.kind === "process");

  return (
    <Stack gap="md">
      <Title order={2}>{t("servers.title")}</Title>
      <Card withBorder padding="md">
        <ServersGraph nodes={data?.nodes ?? []} edges={data?.edges ?? []} />
      </Card>
      <Grid>
        {processNodes.map((n) => (
          <Grid.Col key={n.id} span={{ base: 12, sm: 6, md: 4 }}>
            <Card withBorder padding="md">
              <Stack gap={4}>
                <Group justify="space-between">
                  <Text fw={500}>{n.label}</Text>
                  <Badge color={STATUS_COLOR[n.status]}>{n.status}</Badge>
                </Group>
                <Text size="xs" c="dimmed">
                  port {n.port ?? "—"} · pid {n.pid ?? "—"}
                </Text>
                {n.uptime_s !== undefined ? (
                  <Text size="xs" c="dimmed">
                    uptime {n.uptime_s.toFixed(0)}s
                  </Text>
                ) : null}
              </Stack>
            </Card>
          </Grid.Col>
        ))}
      </Grid>
    </Stack>
  );
}
