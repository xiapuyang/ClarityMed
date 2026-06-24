import { Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function SystemConfig() {
  const { t } = useTranslation();
  return (
    <Stack gap="md">
      <Title order={2}>{t("config.title")}</Title>
      <Text c="dimmed">{t("config.placeholder")}</Text>
    </Stack>
  );
}
