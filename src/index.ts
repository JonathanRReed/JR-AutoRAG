import { serve } from "bun";
import index from "./index.html";

const apiBaseUrl = (
  process.env.BUN_PUBLIC_API_BASE_URL ||
  process.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const proxyApiRequest = async (req: Request, prefix: string) => {
  const url = new URL(req.url);
  const path = prefix && url.pathname.startsWith(prefix)
    ? url.pathname.slice(prefix.length) || "/"
    : url.pathname;
  try {
    return await fetch(apiBaseUrl + path + url.search, {
      method: req.method,
      headers: req.headers,
      body: req.body,
    });
  } catch {
    return new Response("Backend execution failed", { status: 502 });
  }
};

const server = serve({
  idleTimeout: 60,
  routes: {
    // Serve index.html for all unmatched routes.
    "/*": index,

    "/HWC-Icon.png": Bun.file(new URL("./HWC-Icon.png", import.meta.url)),

    // Same-origin proxy for browser clients. The prefix is stripped before the
    // request reaches FastAPI, so /__api/config maps to /config.
    "/__api/*": (req) => proxyApiRequest(req, "/__api"),

    // Preserve the legacy /api proxy for backend routes that already live under
    // FastAPI's /api prefix, such as /api/artifacts/status.
    "/api/*": (req) => proxyApiRequest(req, ""),

    "/api/hello": {
      async GET(req) {
        return Response.json({
          message: "Hello, world!",
          method: "GET",
        });
      },
      async PUT(req) {
        return Response.json({
          message: "Hello, world!",
          method: "PUT",
        });
      },
    },

    "/api/hello/:name": async req => {
      const name = req.params.name;
      return Response.json({
        message: `Hello, ${name}!`,
      });
    },
  },

  development: process.env.NODE_ENV !== "production" && {
    // Enable browser hot reloading in development
    hmr: true,

    // Echo console logs from the browser to the server
    console: true,
  },
});

console.log(`Server running at ${server.url}`);
