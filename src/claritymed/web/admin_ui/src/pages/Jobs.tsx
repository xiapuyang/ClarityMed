import { Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function Jobs() {
  const { t } = useTranslation();
  return (
    <Stack gap="md">
      <Title order={2}>{t("jobs.title")}</Title>
      <Text c="dimmed">{t("jobs.placeholder")}</Text>
    </Stack>
  );
}
