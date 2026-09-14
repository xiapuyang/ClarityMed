import {
  Alert,
  Button,
  Checkbox,
  FileInput,
  Group,
  Modal,
  NumberInput,
  Select,
  Stack,
  TagsInput,
  Text,
  TextInput,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useEffect, useState } from "react";

import { pollJobUntilTerminal } from "../hooks/useAdminJobs";
import {
  useUpsertRagCollection,
  type RagCollection,
  type RagUpsertMetadata,
} from "../hooks/useAdminRag";

const NAME_RE = /^[a-z][a-z0-9_]{0,63}$/;

interface Props {
  opened: boolean;
  onClose: () => void;
  // When ``target`` is set, the modal is in "append files" mode for an
  // existing collection: name is locked + metadata fields are hidden
  // (inherited from yaml on the backend).
  target: RagCollection | null;
}

// Modal for both create-new and append-files flows. The same backend
// endpoint (POST /admin/rag/collections/upsert) handles both — the
// resolver inherits metadata from retrieval.yaml when the name already
// exists, so the SPA can hide those fields in append mode rather than
// asking the operator to retype them.
export function RagUpsertModal({ opened, onClose, target }: Props) {
  const upsert = useUpsertRagCollection();
  const isAppend = target !== null;

  const [name, setName] = useState("");
  const [language, setLanguage] = useState<string>("en");
  const [authorityTier, setAuthorityTier] = useState<string | number>(2);
  const [crossLingual, setCrossLingual] = useState(false);
  const [license, setLicense] = useState("");
  const [topics, setTopics] = useState<string[]>([]);
  const [dedupeThreshold, setDedupeThreshold] = useState<string | number>(0);
  const [files, setFiles] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);

  // Reset every time the modal opens so a previous draft doesn't bleed
  // into a fresh use. Pre-fills the name field in append mode.
  useEffect(() => {
    if (!opened) return;
    setName(target?.name ?? "");
    setLanguage(target?.language ?? "en");
    setAuthorityTier(target?.authority_tier ?? 2);
    setCrossLingual(false);
    setLicense(target?.license ?? "");
    setTopics(target?.topics ?? []);
    setDedupeThreshold(0);
    setFiles([]);
  }, [opened, target]);

  const nameError =
    !isAppend && name.length > 0 && !NAME_RE.test(name)
      ? "must start with a lowercase letter; only [a-z0-9_] allowed"
      : null;
  const canSubmit =
    files.length > 0 && name.length > 0 && !nameError && !submitting;

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      // In append mode, omit metadata fields so the backend inherits
      // them from yaml rather than the SPA pre-filling and accidentally
      // mutating live config when the operator changed a value.
      const metadata: RagUpsertMetadata = isAppend
        ? { name }
        : {
            name,
            language,
            cross_lingual: crossLingual,
            authority_tier: Number(authorityTier) || undefined,
            topics,
            license: license.trim() ? license.trim() : undefined,
            dedupe_cosine_threshold: Number(dedupeThreshold) || 0,
          };

      const spec = await upsert.mutateAsync({ files, metadata });
      notifications.show({
        title: isAppend ? "Append started" : "Create started",
        message: `Job ${spec.id.slice(0, 8)} — ingesting ${files.length} file(s)…`,
        color: "blue",
      });
      onClose();

      // Poll until the job hits a terminal state, then summarise.
      try {
        const final = await pollJobUntilTerminal(spec.id);
        const tail = final.stdout_tail.slice(-3).join(" / ") || final.progress;
        const colour =
          final.state === "done"
            ? "green"
            : final.state === "cancelled"
              ? "yellow"
              : "red";
        notifications.show({
          title: `Ingest ${final.state}`,
          message: tail || `Job ${spec.id.slice(0, 8)} finished as ${final.state}.`,
          color: colour,
          autoClose: 8000,
        });
      } catch (pollErr) {
        notifications.show({
          title: "Job polling failed",
          message: (pollErr as Error).message,
          color: "yellow",
        });
      }
    } catch (e) {
      notifications.show({
        title: "Upload failed",
        message: (e as Error).message,
        color: "red",
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      size="lg"
      title={isAppend ? `Append files to ${target?.name}` : "Create RAG collection"}
    >
      <Stack gap="sm">
        {isAppend ? (
          <Alert color="blue" variant="light">
            <Text size="sm">
              Files will be appended to <Text span fw={500}>{target?.name}</Text>.
              Metadata (language, topics, authority tier, license) is inherited
              from the existing entry in <code>retrieval.yaml</code>.
            </Text>
          </Alert>
        ) : (
          <Alert color="gray" variant="light">
            <Text size="sm">
              A new entry will be appended to{" "}
              <code>configs/retrieval.yaml</code> automatically after the
              first successful ingest. Files run through the same OCR →
              chunk → embed → dedup pipeline as the <code>/upload</code>{" "}
              flow.
            </Text>
          </Alert>
        )}

        <TextInput
          label="Collection name"
          placeholder="ats_idsa_pneumonia_en"
          required
          value={name}
          onChange={(e) => setName(e.currentTarget.value)}
          error={nameError}
          disabled={isAppend}
          description={
            isAppend
              ? "Locked in append mode."
              : "Lowercase letters, digits, underscores. Cannot be renamed later."
          }
        />

        {!isAppend && (
          <>
            <Group grow>
              <Select
                label="Language"
                data={[
                  { value: "en", label: "English" },
                  { value: "zh", label: "Chinese" },
                ]}
                value={language}
                onChange={(v) => setLanguage(v ?? "en")}
              />
              <Select
                label="Authority tier"
                data={[
                  { value: "1", label: "1 — guideline" },
                  { value: "2", label: "2 — textbook" },
                  { value: "3", label: "3 — other" },
                ]}
                value={String(authorityTier)}
                onChange={(v) => setAuthorityTier(v ?? "2")}
              />
            </Group>

            <TagsInput
              label="Topics"
              description="Operator-readable labels; the centroid router ignores topics for scoring."
              value={topics}
              onChange={setTopics}
              placeholder="press Enter to add"
            />

            <TextInput
              label="License"
              placeholder="e.g. Creative Commons / ATS-IDSA educational"
              value={license}
              onChange={(e) => setLicense(e.currentTarget.value)}
            />

            <Group grow>
              <Checkbox
                label="Cross-lingual"
                description="Allow router to consider this corpus for queries in other languages."
                checked={crossLingual}
                onChange={(e) => setCrossLingual(e.currentTarget.checked)}
              />
              <NumberInput
                label="Dedup threshold"
                description="0 disables. 0.92–0.95 typical for overlapping editions."
                min={0}
                max={1}
                step={0.01}
                decimalScale={2}
                value={dedupeThreshold}
                onChange={setDedupeThreshold}
              />
            </Group>
          </>
        )}

        <FileInput
          label="Files"
          required
          multiple
          clearable
          placeholder="Select PDFs / text / images"
          value={files}
          onChange={setFiles}
          description={
            files.length === 0
              ? "Hold ⌘/Ctrl to pick multiple."
              : `${files.length} file(s) selected`
          }
        />

        <Group justify="flex-end" mt="sm">
          <Button variant="default" onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} loading={submitting} disabled={!canSubmit}>
            {isAppend ? "Append" : "Create"}
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}
