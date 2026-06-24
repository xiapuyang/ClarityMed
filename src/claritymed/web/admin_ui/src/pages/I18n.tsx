import { Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function I18nPage() {
  const { t } = useTranslation();
  return (
    <Stack gap="md">
      <Title order={2}>{t("i18nPage.title")}</Title>
      <Text c="dimmed">{t("i18nPage.placeholder")}</Text>
    </Stack>
  );
}
