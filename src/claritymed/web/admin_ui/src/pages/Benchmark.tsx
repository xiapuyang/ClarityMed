import { Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function Benchmark() {
  const { t } = useTranslation();
  return (
    <Stack gap="md">
      <Title order={2}>{t("benchmark.title")}</Title>
      <Text c="dimmed">{t("benchmark.placeholder")}</Text>
    </Stack>
  );
}
