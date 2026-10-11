import { useI18n } from "../i18n.jsx";

const OPTIONS = [
  { id: "light", icon: "☀" },
  { id: "dark", icon: "☾" },
  { id: "system", icon: "⌘" },
];

export default function ThemeToggle({ mode, onChange }) {
  const { t } = useI18n();
  return (
    <div className="theme" role="group" aria-label={t("theme.aria")}>
      {OPTIONS.map((o) => (
        <button key={o.id} type="button" className={mode === o.id ? "on" : ""} aria-pressed={mode === o.id}
                title={t("theme.title", { name: t(`theme.${o.id}`) })} onClick={() => onChange(o.id)}>
          <span aria-hidden="true">{o.icon}</span> {t(`theme.${o.id}`)}
        </button>
      ))}
    </div>
  );
}
