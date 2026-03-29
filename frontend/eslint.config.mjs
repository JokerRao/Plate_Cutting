import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
  {
    files: ["src/app/**/*.tsx", "src/components/**/*.tsx"],
    rules: {
      // 与 Supabase/API 载荷字段对齐成本高；保留类型收紧为后续重构项
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
];

export default eslintConfig;
