// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightLlmsTxt from "starlight-llms-txt";
import mermaid from "astro-mermaid";

// https://astro.build/config
export default defineConfig({
  site: "https://karma.eka.care/",
  integrations: [
    starlight({
      customCss: [
        './src/styles/global.css',
      ],
      title: "KARMA",
      components: {
        PageTitle: './src/components/overrides/PageTitle.astro',
      },
      plugins: [
        starlightLlmsTxt({
          projectName: "KARMA OpenMedEvalKit",
          description:
            "KARMA-OpenMedEvalKit is a toolkit to evaluate Multimodal Large Language Models (LLMs) across diverse datasets. Currently KARMA supports 10+ datasets focusing on HealthCare/Medical domain spanning text, image and audio modalities. This toolkit can easily be extended to include more datasets and models.",
        }),
      ],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/eka-care/KARMA-OpenMedEvalKit.git",
        },
        {
          icon: "add-document",
          label: "LLMs Results",
          href: "https://karma.eka.care/llms-full.txt",
        },
      ],
      sidebar: [
        {
          label: "Getting Started",
          items: [
            { label: "Overview", slug: "index" },
            { label: "Installation Guide", slug: "user-guide/installation" },
            { label: "CLI Basics", slug: "user-guide/cli-basics" },
            {
              label: "Running evaluations",
              slug: "user-guide/running-evaluations",
            },
          ],
        },
        { label: "Supported Datasets and Models", slug: "supported-resources" },
        { label: "Building with LLMs", slug: "building-with-llms" },
        {
          label: "User Guide",
          items: [
            {
              label: "Using KARMA as a package",
              slug: "user-guide/using-through-api",
            },
            {
              label: "Metrics Overview",
              slug: "user-guide/metrics/metrics_overview",
            },
            {
              label: "Processors Overview",
              slug: "user-guide/processors/processors_overview",
            },
            { label: "Registry", slug: "user-guide/registry/registries" },
            {
              label: "Models",
              items: [
                {
                  label: "Built-in Models",
                  slug: "user-guide/models/built-in-models",
                },
                {
                  label: "Model Configuration",
                  slug: "user-guide/models/model-configuration",
                },
              ],
            },
            {
              label: "Datasets",
              items: [
                {
                  label: "Datasets Overview",
                  slug: "user-guide/datasets/datasets_overview",
                },
                  {
                  label: "Data Loader Iterable",
                  slug: "user-guide/datasets/data-loader-iterable",
                },
              ],
            },
            {
              label: "Add Your Own",
              // auto-generate
              autogenerate: { directory: 'user-guide/add-your-own' },
            },
               {
              label: "Commands",
              // auto-generate
              autogenerate: { directory: 'commands' },
            },
          ],
        },

        { label: "Architecture", slug: "architecture" },
        { label: "Caching", slug: "caching" },
        {
              label: "Sanity Benchmark",
              slug: "benchmark/benchmark_overview",
        },
      ],
    }),
    mermaid({
      theme: "forest",
      autoTheme: true,
    }),
  ],
});
