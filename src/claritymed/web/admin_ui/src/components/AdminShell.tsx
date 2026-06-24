import {
  AppShell,
  Burger,
  Group,
  NavLink,
  ScrollArea,
  SegmentedControl,
  Skeleton,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useTranslation } from "react-i18next";
import { NavLink as RouterNavLink, Outlet } from "react-router-dom";

import { useMe } from "../hooks/useMe";

// Locked admin nav order (per plan). Each link points to a top-level
// route inside the admin SPA; the foundation ships placeholder pages
// behind them and later units replace them.
const NAV_KEYS: { key: string; to: string }[] = [
  { key: "overview", to: "/" },
  { key: "rag", to: "/rag" },
  { key: "benchmark", to: "/benchmark" },
  { key: "models", to: "/models" },
  { key: "servers", to: "/servers" },
  { key: "users", to: "/users" },
  { key: "config", to: "/config" },
  { key: "i18n", to: "/i18n" },
  { key: "audit", to: "/audit" },
  { key: "jobs", to: "/jobs" },
];

export function AdminShell() {
  const [opened, { toggle }] = useDisclosure();
  const { t, i18n } = useTranslation();
  const me = useMe();

  return (
    <AppShell
      header={{ height: 56 }}
      navbar={{ width: 240, breakpoint: "sm", collapsed: { mobile: !opened } }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group gap="sm">
            <Burger
              opened={opened}
              onClick={toggle}
              hiddenFrom="sm"
              size="sm"
            />
            <Title order={4}>{t("app.title")}</Title>
          </Group>
          <Group gap="sm">
            {me.isLoading ? (
              <Skeleton height={20} width={120} />
            ) : me.data ? (
              <Text size="sm" c="dimmed">
                {me.data.display_name} · {me.data.role}
              </Text>
            ) : null}
            <SegmentedControl
              size="xs"
              value={i18n.language.startsWith("zh") ? "zh" : "en"}
              onChange={(v) => void i18n.changeLanguage(v)}
              data={[
                { value: "en", label: t("lang.en") },
                { value: "zh", label: t("lang.zh") },
              ]}
            />
          </Group>
        </Group>
      </AppShell.Header>
      <AppShell.Navbar p="md">
        <ScrollArea>
          <Stack gap={4}>
            {NAV_KEYS.map(({ key, to }) => (
              <NavLink
                key={key}
                label={t(`nav.${key}`)}
                component={RouterNavLink}
                to={to}
                end={to === "/"}
              />
            ))}
          </Stack>
        </ScrollArea>
      </AppShell.Navbar>
      <AppShell.Main>
        <Outlet />
      </AppShell.Main>
    </AppShell>
  );
}
