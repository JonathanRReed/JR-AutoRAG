import { serve } from "bun";
import index from "./index.html";

const apiBaseUrl = (
  process.env.BUN_PUBLIC_API_BASE_URL ||
  process.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const proxyApiRequest = async (req: Request) => {
  const url = new URL(req.url);
  try {
    return await fetch(apiBaseUrl + url.pathname + url.search, {
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

    // Do not expose bare FastAPI routes through the web server. The UI should
    // call the configured browser API origin directly instead.
    "/__api/*": () => new Response("API proxy disabled", { status: 404 }),

    // Preserve the legacy /api proxy only for backend routes that already live
    // under FastAPI's /api prefix, such as /api/artifacts/status. Bare backend
    // routes must be reached through the configured browser API origin instead
    // of being exposed through the web server.
    "/api/*": (req) => proxyApiRequest(req),

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
