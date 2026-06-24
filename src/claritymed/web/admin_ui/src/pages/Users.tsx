import { Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function Users() {
  const { t } = useTranslation();
  return (
    <Stack gap="md">
      <Title order={2}>{t("users.title")}</Title>
      <Text c="dimmed">{t("users.placeholder")}</Text>
    </Stack>
  );
}
