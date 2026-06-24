import { Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function Servers() {
  const { t } = useTranslation();
  return (
    <Stack gap="md">
      <Title order={2}>{t("servers.title")}</Title>
      <Text c="dimmed">{t("servers.placeholder")}</Text>
    </Stack>
  );
}
