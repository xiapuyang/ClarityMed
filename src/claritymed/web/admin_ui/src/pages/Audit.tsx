import { Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function Audit() {
  const { t } = useTranslation();
  return (
    <Stack gap="md">
      <Title order={2}>{t("audit.title")}</Title>
      <Text c="dimmed">{t("audit.placeholder")}</Text>
    </Stack>
  );
}
