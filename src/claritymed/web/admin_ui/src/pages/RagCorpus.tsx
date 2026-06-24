import { Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function RagCorpus() {
  const { t } = useTranslation();
  return (
    <Stack gap="md">
      <Title order={2}>{t("rag.title")}</Title>
      <Text c="dimmed">{t("rag.placeholder")}</Text>
    </Stack>
  );
}
