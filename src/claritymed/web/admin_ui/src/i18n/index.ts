import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "./en.json";
import zh from "./zh.json";

// Initial language follows the chat SPA: we read /api/v1/me on app boot
// (see useMe) and call i18n.changeLanguage with the account's `language`
// once the response lands. Until then, the default below is used.

void i18n.use(initReactI18next).init({
  fallbackLng: "en",
  lng: "en",
  resources: {
    en: { translation: en },
    zh: { translation: zh },
  },
  interpolation: { escapeValue: false },
});

export default i18n;
