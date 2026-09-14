import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Group,
  Loader,
  Modal,
  Stack,
  Table,
  Text,
  Title,
  Tooltip,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { notifications } from "@mantine/notifications";
import {
  IconPlus,
  IconRefresh,
  IconTrash,
  IconUpload,
} from "@tabler/icons-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { RagUpsertModal } from "../components/RagUpsertModal";
import {
  useAdminRagCollections,
  useDeleteRagCollection,
  type RagCollection,
} from "../hooks/useAdminRag";

export function RagCorpus() {
  const { t } = useTranslation();
  const { data, isLoading, error, refetch, isFetching } = useAdminRagCollections();
  const del = useDeleteRagCollection();
  const [confirm, setConfirm] = useState<RagCollection | null>(null);
  const [modalOpened, { open: openModal, close: closeModal }] = useDisclosure(false);
  const [upsertTarget, setUpsertTarget] = useState<RagCollection | null>(null);
  const [upsertOpened, { open: openUpsert, close: closeUpsert }] =
    useDisclosure(false);

  if (isLoading) return <Loader />;
  if (error)
    return <Alert color="red">{(error as Error).message ?? t("app.error")}</Alert>;

  const handleDelete = async () => {
    if (!confirm) return;
    try {
      await del.mutateAsync(confirm.name);
      notifications.show({
        title: t("rag.title"),
        message: `${confirm.name} deleted.`,
        color: "green",
      });
      closeModal();
      setConfirm(null);
    } catch (e) {
      notifications.show({
        title: t("app.error"),
        message: (e as Error).message,
        color: "red",
      });
    }
  };

  const handleAppendClick = (c: RagCollection) => {
    setUpsertTarget(c);
    openUpsert();
  };

  const handleCreateClick = () => {
    setUpsertTarget(null);
    openUpsert();
  };

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Title order={2}>{t("rag.title")}</Title>
        <Group gap="sm">
          <ActionIcon
            variant="default"
            onClick={() => refetch()}
            loading={isFetching}
            aria-label={t("app.retry")}
          >
            <IconRefresh size={18} />
          </ActionIcon>
          <Button leftSection={<IconPlus size={16} />} onClick={handleCreateClick}>
            Create collection
          </Button>
        </Group>
      </Group>
      <Table striped highlightOnHover withTableBorder>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Name</Table.Th>
            <Table.Th>Language</Table.Th>
            <Table.Th>Authority</Table.Th>
            <Table.Th>Chunks</Table.Th>
            <Table.Th>Topics</Table.Th>
            <Table.Th> </Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {(data?.items ?? []).map((c) => (
            <Table.Tr key={c.name}>
              <Table.Td>
                <Text size="sm" ff="monospace">
                  {c.name}
                </Text>
              </Table.Td>
              <Table.Td>{c.language}</Table.Td>
              <Table.Td>
                <Badge variant="light">tier {c.authority_tier}</Badge>
              </Table.Td>
              <Table.Td>{c.chunk_count ?? "—"}</Table.Td>
              <Table.Td>
                <Text size="xs" c="dimmed" truncate>
                  {c.topics.join(", ")}
                </Text>
              </Table.Td>
              <Table.Td>
                <Group gap={4} justify="flex-end" wrap="nowrap">
                  <Tooltip label="Append files">
                    <ActionIcon
                      variant="subtle"
                      onClick={() => handleAppendClick(c)}
                    >
                      <IconUpload size={16} />
                    </ActionIcon>
                  </Tooltip>
                  <Tooltip label={t("app.delete")}>
                    <ActionIcon
                      variant="subtle"
                      color="red"
                      onClick={() => {
                        setConfirm(c);
                        openModal();
                      }}
                    >
                      <IconTrash size={16} />
                    </ActionIcon>
                  </Tooltip>
                </Group>
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      <Modal
        opened={modalOpened}
        onClose={closeModal}
        title={`Delete ${confirm?.name ?? ""}?`}
      >
        <Stack gap="sm">
          <Text>
            All chunks in this collection will be removed. This cannot be
            undone.
          </Text>
          <Group justify="flex-end">
            <Button variant="default" onClick={closeModal}>
              {t("app.cancel")}
            </Button>
            <Button color="red" onClick={handleDelete} loading={del.isPending}>
              {t("app.confirm")}
            </Button>
          </Group>
        </Stack>
      </Modal>
      <RagUpsertModal
        opened={upsertOpened}
        onClose={closeUpsert}
        target={upsertTarget}
      />
    </Stack>
  );
}
