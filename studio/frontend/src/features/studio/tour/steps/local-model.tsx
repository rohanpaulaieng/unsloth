// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { ReadMore, type TourStep } from "@/features/tour";

export const studioLocalModelStep: TourStep = {
  id: "local-model",
  target: "studio-local-model",
  title: "Local model path",
  body: (
    <>
      Use this if you already downloaded weights locally (eg{" "}
      <span className="font-mono">./models/...</span>) to avoid re-downloading.
      Folder should look like a Hugging Face model (config + tokenizer + weights).{" "}
      <ReadMore href="https://docs.unsloth.ai/basics/fine-tuning-llms-guide" />
    </>
  ),
};
