import {
  Alert,
  Button,
  Group,
  Loader,
  SegmentedControl,
  Stack,
  Table,
  Text,
  Textarea,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  flattenStrings,
  makeNested,
  useAdminI18n,
  usePatchAdminI18n,
  type I18nLang,
  type I18nSurface,
} from "../hooks/useAdminI18n";

export function I18nPage() {
  const { t } = useTranslation();
  const [surface, setSurface] = useState<I18nSurface>("backend");
  const [lang, setLang] = useState<I18nLang>("en");
  const { data, isLoading, error } = useAdminI18n(surface, lang);
  const patch = usePatchAdminI18n(surface, lang);
  const [edits, setEdits] = useState<Record<string, string>>({});

  useEffect(() => {
    setEdits({});
  }, [surface, lang]);

  if (isLoading) return <Loader />;
  if (error)
    return <Alert color="red">{(error as Error).message ?? t("app.error")}</Alert>;

  const rows = data ? flattenStrings(data.data) : [];

  const save = async (path: string) => {
    const value = edits[path];
    if (value === undefined) return;
    try {
      const res = await patch.mutateAsync(makeNested(path, value));
      notifications.show({
        title: t("i18nPage.title"),
        message: res.needs_rebuild
          ? `${path} updated. Run \`make admin-ui\` to apply.`
          : `${path} updated.`,
        color: res.needs_rebuild ? "yellow" : "green",
        autoClose: res.needs_rebuild ? false : 4000,
      });
      const next = { ...edits };
      delete next[path];
      setEdits(next);
    } catch (e) {
      notifications.show({
        title: t("app.error"),
        message: (e as Error).message,
        color: "red",
      });
    }
  };

  return (
    <Stack gap="md">
      <Title order={2}>{t("i18nPage.title")}</Title>
      <Group gap="md">
        <SegmentedControl
          value={surface}
          onChange={(v) => setSurface(v as I18nSurface)}
          data={[
            { value: "backend", label: "Backend" },
            { value: "admin", label: "Admin UI" },
          ]}
        />
        <SegmentedControl
          value={lang}
          onChange={(v) => setLang(v as I18nLang)}
          data={[
            { value: "en", label: t("lang.en") },
            { value: "zh", label: t("lang.zh") },
          ]}
        />
        {surface === "admin" ? (
          <Text size="xs" c="dimmed">
            Edits require `make admin-ui` (or Vite HMR in dev).
          </Text>
        ) : null}
      </Group>
      <Table striped highlightOnHover withTableBorder>
        <Table.Thead>
          <Table.Tr>
            <Table.Th w={300}>Key</Table.Th>
            <Table.Th>Value</Table.Th>
            <Table.Th w={100}> </Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map((r) => (
            <Table.Tr key={r.path}>
              <Table.Td>
                <Text size="xs" ff="monospace">
                  {r.path}
                </Text>
              </Table.Td>
              <Table.Td>
                <Textarea
                  autoComplete="off"
                  minRows={1}
                  autosize
                  value={edits[r.path] ?? r.value}
                  onChange={(e) =>
                    setEdits({ ...edits, [r.path]: e.currentTarget.value })
                  }
                />
              </Table.Td>
              <Table.Td>
                {edits[r.path] !== undefined &&
                edits[r.path] !== r.value ? (
                  <Button
                    size="xs"
                    variant="default"
                    onClick={() => save(r.path)}
                    loading={patch.isPending}
                  >
                    {t("app.save")}
                  </Button>
                ) : null}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Stack>
  );
}
