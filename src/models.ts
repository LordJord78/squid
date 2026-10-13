// The model shelf: find every model in LM Studio's folders and describe it the way `lms ls` does.
// Squid shares LM Studio's folders, so nothing is downloaded twice.
import { readdirSync, readFileSync, statSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { readGgufHeader, describeModel } from "./gguf.ts";

export type ShelfModel = {
  key: string; // the name you load it by, e.g. "qwen/qwen3.5-9b"
  type: "llm" | "embedding";
  publisher: string;
  params?: string; // "9B", "30B-A3B"
  architecture: string;
  quantization?: string;
  contextLength?: number;
  sizeBytes: number; // every .gguf in the folder, vision add-on included
  vision: boolean;
  catalog: boolean; // named by an LM Studio catalog entry
  variants: number; // model files in the folder (e.g. a Q4 and a Q8 of the same model)
  path: string; // the main .gguf file
};

const LMSTUDIO = join(homedir(), ".lmstudio");

// LM Studio's download folder (its settings.json can move it, here to D:\LMStudio\models),
// plus the models it ships with.
export function modelRoots(): string[] {
  let downloads = join(LMSTUDIO, "models");
  try {
    const settings = JSON.parse(readFileSync(join(LMSTUDIO, "settings.json"), "utf8"));
    if (typeof settings.downloadsFolder === "string") downloads = settings.downloadsFolder;
  } catch {
    // no settings file: use the default folder
  }
  return [downloads, join(LMSTUDIO, ".internal", "bundled-models")].filter((dir) => existsSync(dir));
}

// LM Studio's catalog gives some download folders a short name. Each entry's manifest.json lists
// the Hugging Face repos it covers: lmstudio-community/Qwen3.5-9B-GGUF → qwen/qwen3.5-9b.
// The entry's own folder (manifest, model.yaml, README, thumbnail) counts toward the model's size.
export type CatalogEntry = { key: string; dir: string };

export function catalogNames(hubDir = join(LMSTUDIO, "hub", "models")): Map<string, CatalogEntry> {
  const names = new Map<string, CatalogEntry>();
  for (const owner of subfolders(hubDir)) {
    for (const name of subfolders(join(hubDir, owner))) {
      const dir = join(hubDir, owner, name);
      try {
        const manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
        for (const dependency of manifest.dependencies ?? []) {
          for (const source of dependency.sources ?? []) {
            names.set(`${source.user}/${source.repo}`.toLowerCase(), { key: `${manifest.owner}/${manifest.name}`, dir });
          }
        }
      } catch {
        // not a catalog entry we understand: skip it
      }
    }
  }
  return names;
}

// Models live at <root>/<publisher>/<repo>/<file>.gguf, one model per repo folder.
export function listModels(roots = modelRoots(), catalog = catalogNames()): ShelfModel[] {
  const shelf: ShelfModel[] = [];
  for (const root of roots) {
    for (const publisher of subfolders(root)) {
      for (const repo of subfolders(join(root, publisher))) {
        const folder = join(root, publisher, repo);
        const files = readdirSync(folder).filter((f) => f.toLowerCase().endsWith(".gguf")).map((f) => join(folder, f));
        const headers = files.map((file) => ({ file, header: readGgufHeader(file) }));
        // A vision add-on (mmproj) is an image encoder whose architecture is "clip". It belongs to the model.
        const isAddOn = (h: (typeof headers)[number]) => describeModel(h.header).architecture === "clip";
        const variants = headers.filter((h) => !isAddOn(h));
        const main = variants[0];
        if (!main) continue;
        const info = describeModel(main.header);
        const plainKey = repo.toLowerCase().replace("-gguf", "");
        const entry = catalog.get(`${publisher}/${repo}`.toLowerCase());
        const counted = entry ? [...files, ...filesIn(entry.dir)] : files;
        shelf.push({
          key: entry?.key ?? (info.embedding ? `text-embedding-${plainKey}` : plainKey),
          type: info.embedding ? "embedding" : "llm",
          publisher: entry ? entry.key.split("/")[0] : publisher,
          params: info.sizeLabel,
          architecture: info.architecture,
          quantization: info.quantization,
          contextLength: info.contextLength,
          sizeBytes: counted.reduce((sum, file) => sum + statSync(file).size, 0),
          vision: headers.some(isAddOn),
          catalog: entry !== undefined,
          variants: variants.length,
          path: main.file,
        });
      }
    }
  }
  // Plain character order, as LM Studio sorts: "qwen/…" comes before "qwen3…" because "/" < "3".
  return shelf.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
}

function filesIn(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).filter((d) => d.isFile()).map((d) => join(dir, d.name));
}

function subfolders(dir: string): string[] {
  try {
    return readdirSync(dir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name);
  } catch {
    return [];
  }
}
