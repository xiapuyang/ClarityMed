import {
  ActionIcon,
  Alert,
  Button,
  Group,
  Loader,
  Modal,
  PasswordInput,
  Select,
  Stack,
  Table,
  Text,
  Title,
  Tooltip,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { notifications } from "@mantine/notifications";
import { IconKey } from "@tabler/icons-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import {
  useAdminUsers,
  usePatchAdminUser,
  useResetAdminUserPassword,
  type AdminUserSummary,
} from "../hooks/useAdminUsers";

type Role = "admin" | "user";
type Lang = "en" | "zh";
type Draft = { role: Role; language: Lang };

export function Users() {
  const { t } = useTranslation();
  const { data, isLoading, error } = useAdminUsers();
  const patch = usePatchAdminUser();
  const reset = useResetAdminUserPassword();
  const [resetTarget, setResetTarget] = useState<AdminUserSummary | null>(null);
  const [newPassword, setNewPassword] = useState("");
  const [resetOpened, { open: openReset, close: closeReset }] = useDisclosure(false);
  // Per-user draft edits — applied only when the row's Save button fires.
  const [drafts, setDrafts] = useState<Record<string, Partial<Draft>>>({});

  if (isLoading) return <Loader />;
  if (error)
    return <Alert color="red">{(error as Error).message ?? t("app.error")}</Alert>;

  const items = data?.items ?? [];

  const draftFor = (u: AdminUserSummary): Draft => ({
    role: (drafts[u.user_id]?.role ?? u.role) as Role,
    language: (drafts[u.user_id]?.language ?? u.language) as Lang,
  });

  const isDirty = (u: AdminUserSummary): boolean => {
    const d = draftFor(u);
    return d.role !== u.role || d.language !== u.language;
  };

  const setDraft = (userId: string, patch: Partial<Draft>) => {
    setDrafts((prev) => ({ ...prev, [userId]: { ...prev[userId], ...patch } }));
  };

  const saveRow = async (u: AdminUserSummary) => {
    const d = draftFor(u);
    const body: Partial<Draft> = {};
    if (d.role !== u.role) body.role = d.role;
    if (d.language !== u.language) body.language = d.language;
    if (Object.keys(body).length === 0) return;
    try {
      await patch.mutateAsync({ userId: u.user_id, patch: body });
      notifications.show({
        title: t("users.title"),
        message: `${u.user_id} updated`,
        color: "green",
      });
      setDrafts((prev) => {
        const next = { ...prev };
        delete next[u.user_id];
        return next;
      });
    } catch (e) {
      notifications.show({
        title: t("app.error"),
        message: (e as Error).message,
        color: "red",
      });
    }
  };

  const submitResetPassword = async () => {
    if (!resetTarget) return;
    try {
      await reset.mutateAsync({
        userId: resetTarget.user_id,
        newPassword,
      });
      notifications.show({
        title: t("users.title"),
        message: `Password reset for ${resetTarget.user_id}`,
        color: "green",
      });
      closeReset();
      setNewPassword("");
      setResetTarget(null);
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
      <Title order={2}>{t("users.title")}</Title>
      <Table striped highlightOnHover withTableBorder>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>User ID</Table.Th>
            <Table.Th>Role</Table.Th>
            <Table.Th>Language</Table.Th>
            <Table.Th>Provider</Table.Th>
            <Table.Th> </Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {items.map((u) => {
            const d = draftFor(u);
            const dirty = isDirty(u);
            return (
              <Table.Tr key={u.user_id}>
                <Table.Td>
                  <Text ff="monospace">{u.user_id}</Text>
                </Table.Td>
                <Table.Td>
                  <Select
                    value={d.role}
                    data={[
                      { value: "admin", label: "admin" },
                      { value: "user", label: "user" },
                    ]}
                    onChange={(v) =>
                      setDraft(u.user_id, { role: (v as Role) ?? u.role })
                    }
                  />
                </Table.Td>
                <Table.Td>
                  <Select
                    value={d.language}
                    data={[
                      { value: "en", label: t("lang.en") },
                      { value: "zh", label: t("lang.zh") },
                    ]}
                    onChange={(v) =>
                      setDraft(u.user_id, { language: (v as Lang) ?? u.language })
                    }
                  />
                </Table.Td>
                <Table.Td>{u.provider_id ?? "—"}</Table.Td>
                <Table.Td>
                  <Group gap="xs">
                    <Button
                      size="xs"
                      variant="default"
                      disabled={!dirty}
                      loading={patch.isPending}
                      onClick={() => saveRow(u)}
                    >
                      {t("app.save")}
                    </Button>
                    <Tooltip label="Reset password">
                      <ActionIcon
                        variant="subtle"
                        onClick={() => {
                          setResetTarget(u);
                          setNewPassword("");
                          openReset();
                        }}
                      >
                        <IconKey size={16} />
                      </ActionIcon>
                    </Tooltip>
                  </Group>
                </Table.Td>
              </Table.Tr>
            );
          })}
        </Table.Tbody>
      </Table>
      <Modal
        opened={resetOpened}
        onClose={closeReset}
        title={`Reset password — ${resetTarget?.user_id ?? ""}`}
      >
        <Stack gap="sm">
          <Text size="sm" c="dimmed">
            The new password takes effect immediately. Communicate it to the
            user out of band.
          </Text>
          <PasswordInput
            autoComplete="new-password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.currentTarget.value)}
            label="New password"
          />
          <Group justify="flex-end">
            <Button variant="default" onClick={closeReset}>
              {t("app.cancel")}
            </Button>
            <Button
              disabled={newPassword.length < 4}
              onClick={submitResetPassword}
              loading={reset.isPending}
            >
              {t("app.confirm")}
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  );
}
