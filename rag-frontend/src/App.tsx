import { Route, Routes } from "react-router-dom";
import { Sidebar } from "@/components/Sidebar";
import { QueryPage } from "@/pages/Query";
import { LibraryPage } from "@/pages/Library";
import { EvalPage } from "@/pages/Eval";
import { SettingsPage } from "@/pages/Settings";

export default function App() {
  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<QueryPage />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/eval" element={<EvalPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}
