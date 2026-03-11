// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { RouterProvider } from "@tanstack/react-router";
import { router } from "./router";

export function App() {
  return <RouterProvider router={router} />;
}
