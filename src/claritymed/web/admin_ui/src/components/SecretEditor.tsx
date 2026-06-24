import {
  Alert,
  Badge,
  Button,
  Group,
  PasswordInput,
  Stack,
  Text,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  useAdminSecrets,
  usePatchAdminSecrets,
} from "../hooks/useAdminModels";

export function SecretEditor() {
  const { t } = useTranslation();
  const { data, isLoading } = useAdminSecrets();
  const patch = usePatchAdminSecrets();
  const [pending, setPending] = useState<Record<string, string>>({});

  if (isLoading || !data) return null;

  const onSave = async () => {
    if (Object.keys(pending).length === 0) return;
    try {
      const res = await patch.mutateAsync(pending);
      setPending({});
      notifications.show({
        title: t("models.tabs.secrets"),
        message: `${res.keys_changed.length} key(s) updated. ${t("app.restart_required")}`,
        color: "yellow",
        autoClose: false,
      });
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
      <Alert color="yellow">{t("app.restart_required")}</Alert>
      {data.status.map((row) => (
        <Group key={row.key} align="end" gap="sm" wrap="nowrap">
          <Stack gap={2} w={260}>
            <Text size="sm" fw={500}>
              {row.label}
            </Text>
            <Text size="xs" c="dimmed">
              {row.hint}
            </Text>
          </Stack>
          <Badge
            color={row.is_set ? "green" : row.required ? "red" : "gray"}
            variant="light"
          >
            {row.source}
          </Badge>
          <PasswordInput
            autoComplete="new-password"
            placeholder={
              row.is_set ? "•••••• (leave blank to keep)" : "Enter new value"
            }
            value={pending[row.key] ?? ""}
            onChange={(e) => {
              const next = { ...pending };
              const v = e.currentTarget.value;
              if (v.length === 0) {
                delete next[row.key];
              } else {
                next[row.key] = v;
              }
              setPending(next);
            }}
            flex={1}
          />
        </Group>
      ))}
      <Group justify="flex-end">
        <Button
          onClick={onSave}
          disabled={Object.keys(pending).length === 0}
          loading={patch.isPending}
        >
          {t("app.save")} ({Object.keys(pending).length})
        </Button>
      </Group>
    </Stack>
  );
}
