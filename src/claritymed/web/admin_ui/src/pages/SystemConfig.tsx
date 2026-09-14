import {
  Alert,
  Box,
  Button,
  Grid,
  Group,
  Loader,
  NavLink,
  Stack,
  Switch,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { YamlPreview } from "../components/YamlPreview";
import {
  getDotted,
  useAdminConfig,
  useAdminConfigCatalog,
  usePatchAdminConfig,
} from "../hooks/useAdminConfigs";

export function SystemConfig() {
  const { t } = useTranslation();
  const catalog = useAdminConfigCatalog();
  const [selected, setSelected] = useState<string | null>(null);
  const detail = useAdminConfig(selected);
  const patch = usePatchAdminConfig();

  useEffect(() => {
    if (!selected && catalog.data?.configs.length) {
      setSelected(catalog.data.configs[0]);
    }
  }, [catalog.data, selected]);

  if (catalog.isLoading) return <Loader />;
  if (catalog.error)
    return (
      <Alert color="red">
        {(catalog.error as Error).message ?? t("app.error")}
      </Alert>
    );

  const submit = async (path: string, value: unknown) => {
    if (!selected) return;
    try {
      await patch.mutateAsync({ name: selected, path, value });
      notifications.show({
        message: `${selected} · ${path} updated`,
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

  return (
    <Stack gap="md">
      <Title order={2}>{t("config.title")}</Title>
      <Grid>
        <Grid.Col span={3}>
          <Stack gap={4}>
            {(catalog.data?.configs ?? []).map((name) => (
              <NavLink
                key={name}
                label={name}
                active={selected === name}
                onClick={() => setSelected(name)}
              />
            ))}
          </Stack>
        </Grid.Col>
        <Grid.Col span={9}>
          {detail.isLoading ? (
            <Loader />
          ) : detail.data ? (
            <Stack gap="md">
              <Box>
                <Title order={4}>{detail.data.name}</Title>
                <YamlPreview data={detail.data.data} />
              </Box>
              <Stack gap="sm">
                <Title order={5}>Editable fields</Title>
                {detail.data.editable_keys.length === 0 ? (
                  <Text c="dimmed">No editable keys.</Text>
                ) : (
                  detail.data.editable_keys.map((key) => (
                    <EditableRow
                      key={key}
                      path={key}
                      current={getDotted(detail.data!.data, key)}
                      onSave={submit}
                    />
                  ))
                )}
              </Stack>
            </Stack>
          ) : null}
        </Grid.Col>
      </Grid>
    </Stack>
  );
}

function EditableRow({
  path,
  current,
  onSave,
}: {
  path: string;
  current: unknown;
  onSave: (path: string, value: unknown) => Promise<void>;
}) {
  // null + undefined both render as empty (placeholder fields like
  // ``abstain.threshold: null`` would otherwise show the string "null").
  const display = (v: unknown) =>
    v === undefined || v === null ? "" : String(v);
  const [text, setText] = useState(display(current));
  const [draftBool, setDraftBool] = useState<boolean>(
    typeof current === "boolean" ? current : false,
  );
  useEffect(() => {
    setText(display(current));
    if (typeof current === "boolean") setDraftBool(current);
  }, [current]);

  // Booleans get a draft Switch + Save button; everything else a draft
  // TextInput + Save button. Neither variant auto-saves on change — the
  // edit only lands once the user clicks Save. The backend re-validates
  // via the Pydantic schema for the YAML, so a typo lands as 422.
  if (typeof current === "boolean") {
    return (
      <Group gap="sm">
        <Text size="sm" ff="monospace" w={250}>
          {path}
        </Text>
        <Switch
          checked={draftBool}
          onChange={(e) => setDraftBool(e.currentTarget.checked)}
        />
        <Button
          variant="default"
          onClick={() => onSave(path, draftBool)}
          disabled={draftBool === current}
        >
          Save
        </Button>
      </Group>
    );
  }

  return (
    <Group gap="sm">
      <Text size="sm" ff="monospace" w={250}>
        {path}
      </Text>
      <TextInput
        autoComplete="off"
        value={text}
        onChange={(e) => setText(e.currentTarget.value)}
        w={300}
      />
      <Button
        variant="default"
        onClick={() => onSave(path, coerce(text, current))}
        disabled={text === display(current)}
      >
        Save
      </Button>
    </Group>
  );
}

// Heuristic coercion based on the field's current type. The backend
// will re-validate; this is just so a numeric field doesn't land as a
// string.
function coerce(text: string, current: unknown): unknown {
  if (typeof current === "number") {
    const n = Number(text);
    return Number.isFinite(n) ? n : text;
  }
  return text;
}
