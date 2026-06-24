import { Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function Models() {
  const { t } = useTranslation();
  return (
    <Stack gap="md">
      <Title order={2}>{t("models.title")}</Title>
      <Text c="dimmed">{t("models.placeholder")}</Text>
    </Stack>
  );
}
