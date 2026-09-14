import { Route, Routes } from "react-router-dom";

import { AdminShell } from "./components/AdminShell";
import { Audit } from "./pages/Audit";
import { Benchmark } from "./pages/Benchmark";
import { Forbidden } from "./pages/Forbidden";
import { I18nPage } from "./pages/I18n";
import { Jobs } from "./pages/Jobs";
import { Models } from "./pages/Models";
import { Overview } from "./pages/Overview";
import { RagCorpus } from "./pages/RagCorpus";
import { Servers } from "./pages/Servers";
import { SystemConfig } from "./pages/SystemConfig";
import { Users } from "./pages/Users";

export default function App() {
  return (
    <Routes>
      <Route element={<AdminShell />}>
        <Route index element={<Overview />} />
        <Route path="rag" element={<RagCorpus />} />
        <Route path="benchmark" element={<Benchmark />} />
        <Route path="models" element={<Models />} />
        <Route path="servers" element={<Servers />} />
        <Route path="users" element={<Users />} />
        <Route path="config" element={<SystemConfig />} />
        <Route path="i18n" element={<I18nPage />} />
        <Route path="audit" element={<Audit />} />
        <Route path="jobs" element={<Jobs />} />
        <Route path="forbidden" element={<Forbidden />} />
      </Route>
    </Routes>
  );
}
