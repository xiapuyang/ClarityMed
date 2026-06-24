import {
  Alert,
  Loader,
  NavLink,
  Stack,
  Tabs,
  Text,
  Title,
} from "@mantine/core";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { SecretEditor } from "../components/SecretEditor";
import { VersionsFlowPanel } from "../components/VersionsFlowPanel";
import { YamlPreview } from "../components/YamlPreview";
import {
  useAdminCatalog,
  useAdminCatalogs,
} from "../hooks/useAdminModels";

const VERSIONS_FLOW_CATALOGS = new Set([
  "vision.yaml",
  "medical_clip.yaml",
  "symptoms.yaml",
]);

export function Models() {
  const { t } = useTranslation();
  const catalogs = useAdminCatalogs();
  const [selected, setSelected] = useState<string | null>(null);
  const detail = useAdminCatalog(selected);

  useEffect(() => {
    if (!selected && catalogs.data?.catalogs.length) {
      setSelected(catalogs.data.catalogs[0]);
    }
  }, [catalogs.data, selected]);

  if (catalogs.isLoading) return <Loader />;
  if (catalogs.error)
    return (
      <Alert color="red">
        {(catalogs.error as Error).message ?? t("app.error")}
      </Alert>
    );

  return (
    <Stack gap="md">
      <Title order={2}>{t("models.title")}</Title>
      <Tabs defaultValue="catalogs">
        <Tabs.List>
          <Tabs.Tab value="catalogs">{t("models.tabs.catalogs")}</Tabs.Tab>
          <Tabs.Tab value="versions">{t("models.tabs.versions")}</Tabs.Tab>
          <Tabs.Tab value="secrets">{t("models.tabs.secrets")}</Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="catalogs" pt="md">
          <Stack gap="sm" style={{ flexDirection: "row" }}>
            <Stack gap={4} w={220}>
              {(catalogs.data?.catalogs ?? []).map((name) => (
                <NavLink
                  key={name}
                  label={name}
                  active={selected === name}
                  onClick={() => setSelected(name)}
                />
              ))}
            </Stack>
            <Stack flex={1}>
              {detail.isLoading ? (
                <Loader />
              ) : detail.data ? (
                <YamlPreview data={detail.data.data} />
              ) : null}
            </Stack>
          </Stack>
        </Tabs.Panel>
        <Tabs.Panel value="versions" pt="md">
          <Stack gap="sm">
            {selected && VERSIONS_FLOW_CATALOGS.has(selected) ? (
              detail.data ? (
                <VersionsFlowPanel data={detail.data.data} />
              ) : (
                <Loader />
              )
            ) : (
              <Text c="dimmed">
                Pick a catalog with multi-version flow (vision, medical_clip,
                symptoms).
              </Text>
            )}
          </Stack>
        </Tabs.Panel>
        <Tabs.Panel value="secrets" pt="md">
          <SecretEditor />
        </Tabs.Panel>
      </Tabs>
    </Stack>
  );
}
