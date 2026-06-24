import { Anchor, Container, Stack, Text, Title } from "@mantine/core";
import { useTranslation } from "react-i18next";

export function Forbidden() {
  const { t } = useTranslation();
  return (
    <Container size="sm" pt="xl">
      <Stack gap="md">
        <Title order={2}>{t("forbidden.title")}</Title>
        <Text>{t("forbidden.body")}</Text>
        <Anchor href="/">/</Anchor>
      </Stack>
    </Container>
  );
}
