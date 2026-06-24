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
  TextInput,
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

export function Users() {
  const { t } = useTranslation();
  const { data, isLoading, error } = useAdminUsers();
  const patch = usePatchAdminUser();
  const reset = useResetAdminUserPassword();
  const [resetTarget, setResetTarget] = useState<AdminUserSummary | null>(null);
  const [newPassword, setNewPassword] = useState("");
  const [resetOpened, { open: openReset, close: closeReset }] = useDisclosure(false);

  if (isLoading) return <Loader />;
  if (error)
    return <Alert color="red">{(error as Error).message ?? t("app.error")}</Alert>;

  const items = data?.items ?? [];

  const handleRoleChange = async (
    user: AdminUserSummary,
    nextRole: "admin" | "user",
  ) => {
    if (nextRole === user.role) return;
    try {
      await patch.mutateAsync({ userId: user.user_id, patch: { role: nextRole } });
      notifications.show({
        title: t("users.title"),
        message: `${user.user_id}: ${user.role} → ${nextRole}`,
        color: "green",
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
            <Table.Th>Display name</Table.Th>
            <Table.Th>Role</Table.Th>
            <Table.Th>Language</Table.Th>
            <Table.Th>Provider</Table.Th>
            <Table.Th> </Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {items.map((u) => (
            <Table.Tr key={u.user_id}>
              <Table.Td>
                <Text ff="monospace">{u.user_id}</Text>
              </Table.Td>
              <Table.Td>
                <TextInput
                  autoComplete="off"
                  defaultValue={u.display_name}
                  onBlur={(e) => {
                    const v = e.currentTarget.value;
                    if (v !== u.display_name && v.length > 0) {
                      patch.mutateAsync({
                        userId: u.user_id,
                        patch: { display_name: v },
                      });
                    }
                  }}
                />
              </Table.Td>
              <Table.Td>
                <Select
                  value={u.role}
                  data={[
                    { value: "admin", label: "admin" },
                    { value: "user", label: "user" },
                  ]}
                  onChange={(v) => handleRoleChange(u, (v as "admin" | "user") ?? u.role)}
                />
              </Table.Td>
              <Table.Td>
                <Select
                  value={u.language}
                  data={[
                    { value: "en", label: t("lang.en") },
                    { value: "zh", label: t("lang.zh") },
                  ]}
                  onChange={(v) =>
                    patch.mutateAsync({
                      userId: u.user_id,
                      patch: { language: (v as "en" | "zh") ?? u.language },
                    })
                  }
                />
              </Table.Td>
              <Table.Td>{u.provider_id ?? "—"}</Table.Td>
              <Table.Td>
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
              </Table.Td>
            </Table.Tr>
          ))}
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
