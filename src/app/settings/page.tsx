"use client";

import Link from "next/link";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarSeparator,
  useSidebar,
} from "@/components/ui/sidebar";

const API_URL_KEY = "agro-tech:chat-api-url";
const DEFAULT_API_URL = process.env.NEXT_PUBLIC_CHAT_API_URL ?? "";

function isValidApiUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function MobileSidebarToggle() {
  const { toggleSidebar } = useSidebar();

  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      className="h-8 md:hidden"
      onClick={toggleSidebar}
    >
      Menu
    </Button>
  );
}

export default function SettingsPage() {
  const [apiUrl, setApiUrl] = useState(() => {
    if (typeof window === "undefined") return DEFAULT_API_URL;
    const saved = window.localStorage.getItem(API_URL_KEY);
    if (saved && !DEFAULT_API_URL) return saved;
    return DEFAULT_API_URL;
  });
  const [status, setStatus] = useState<string>("");

  const onSave = () => {
    const trimmed = apiUrl.trim();

    if (trimmed.length > 0 && !isValidApiUrl(trimmed)) {
      setStatus("API URL must be a valid http or https URL.");
      return;
    }

    if (trimmed.length === 0) {
      window.localStorage.removeItem(API_URL_KEY);
      setStatus("Saved. Built in OpenAI fallback is active.");
      return;
    }

    window.localStorage.setItem(API_URL_KEY, trimmed);
    setStatus("Saved.");
  };

  const onReset = () => {
    setApiUrl("");
    window.localStorage.removeItem(API_URL_KEY);
    setStatus("Reset. Built in OpenAI fallback is active.");
  };

  return (
    <SidebarProvider defaultOpen>
      <Sidebar
        collapsible="offcanvas"
        variant="sidebar"
        className="border-r border-sidebar-border bg-sidebar"
      >
        <SidebarHeader className="gap-2 px-3 py-3">
          <Button
            variant="default"
            size="default"
            className="h-9 justify-start bg-sidebar-primary text-sidebar-primary-foreground hover:bg-sidebar-primary/90"
            render={<Link href="/" />}
          >
            New chat
          </Button>
        </SidebarHeader>

        <SidebarSeparator />

        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Navigation</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                <SidebarMenuItem>
                  <SidebarMenuButton render={<Link href="/" />}>
                    <span>Chat</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
                <SidebarMenuItem>
                  <SidebarMenuButton isActive render={<Link href="/settings" />}>
                    <span>Settings</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>

        <SidebarFooter className="px-3 pb-3 pt-0">
          <Button
            type="button"
            variant="outline"
            size="default"
            className="h-9 justify-start"
            render={<Link href="/" />}
          >
            Back to chat
          </Button>
        </SidebarFooter>

        <SidebarRail />
      </Sidebar>

      <SidebarInset className="bg-background">
        <main className="flex min-h-screen flex-col">
          <header className="sticky top-0 z-20 border-b border-border bg-background/95 backdrop-blur">
            <div className="mx-auto flex w-full max-w-3xl items-center gap-3 px-4 py-3">
              <MobileSidebarToggle />
              <div className="min-w-0 flex-1">
                <p className="truncate text-lg font-semibold text-foreground">Settings</p>
              </div>
            </div>
          </header>

          <section className="mx-auto w-full max-w-3xl px-4 py-6">
            <div className="rounded-md border border-border bg-card p-5">
              <p className="text-base font-semibold text-foreground">API Endpoint</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Configure your custom API endpoint for chat requests.
              </p>

              <div className="mt-6">
                <label
                  htmlFor="api-url"
                  className="text-xs uppercase tracking-[0.12em] text-muted-foreground"
                >
                  Custom API URL optional
                </label>
                <Input
                  id="api-url"
                  name="api-url"
                  value={apiUrl}
                  onChange={(event) => {
                    setApiUrl(event.target.value);
                    if (status) setStatus("");
                  }}
                  placeholder="https://your-api.example.com/chat"
                  autoComplete="off"
                  spellCheck={false}
                  className="mt-2 h-9 bg-background"
                />
                <p className="mt-2 text-xs text-muted-foreground">
                  Leave empty to use built in OpenAI fallback at /api/chat
                </p>
              </div>

              <div className="mt-5 flex items-center gap-2">
                <Button
                  type="button"
                  size="default"
                  className="h-9"
                  onClick={onSave}
                >
                  Save
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="default"
                  className="h-9"
                  onClick={onReset}
                >
                  Use fallback
                </Button>
              </div>

              <p className="mt-3 text-xs text-muted-foreground">
                {status || "Changes apply immediately in chat."}
              </p>
            </div>
          </section>
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}
