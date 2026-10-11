// Copies the JSON the dashboard reads from ../web/data into public/data (the model/pipeline write to web/data).
import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "..", "web", "data");
const dest = join(root, "public", "data");
if (!existsSync(src)) {
  console.warn("sync-data: ../web/data not found, keeping existing public/data");
} else {
  mkdirSync(dest, { recursive: true });
  cpSync(src, dest, { recursive: true });
  console.log("sync-data: copied", src, "->", dest);
}
