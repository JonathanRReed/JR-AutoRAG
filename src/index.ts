import { serve } from "bun";
import index from "./index.html";

const server = serve({
  idleTimeout: 60,
  routes: {
    // Serve index.html for all unmatched routes.
    "/*": index,

    "/HWC-Icon.png": Bun.file(new URL("./HWC-Icon.png", import.meta.url)),

    // Proxy API requests to Python backend
    "/api/*": async (req) => {
      const url = new URL(req.url);
      try {
        return await fetch("http://127.0.0.1:8000" + url.pathname + url.search, {
          method: req.method,
          headers: req.headers,
          body: req.body,
        });
      } catch (err) {
        return new Response("Backend execution failed", { status: 502 });
      }
    },

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
