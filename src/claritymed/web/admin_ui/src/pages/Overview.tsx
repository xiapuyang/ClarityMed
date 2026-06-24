import { Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function Overview() {
  const { t } = useTranslation();
  return (
    <Stack gap="md">
      <Title order={2}>{t("overview.title")}</Title>
      <Text c="dimmed">{t("overview.placeholder")}</Text>
    </Stack>
  );
}
