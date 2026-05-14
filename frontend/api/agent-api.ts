type QueryValue = string | string[] | undefined;

type VercelRequest = {
  method?: string;
  url?: string;
  headers: Record<string, QueryValue>;
  query: Record<string, QueryValue>;
  body?: unknown;
};

type VercelResponse = {
  status: (code: number) => VercelResponse;
  setHeader: (name: string, value: string) => void;
  send: (body: Buffer | string) => void;
};

const SKIP_RESPONSE_HEADERS = new Set([
  "connection",
  "content-encoding",
  "content-length",
  "keep-alive",
  "transfer-encoding",
]);

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const spaceUrl = process.env.HF_SPACE_URL;
  if (!spaceUrl) {
    res.status(500).send("HF_SPACE_URL is not configured.");
    return;
  }

  const upstreamUrl = buildUpstreamUrl(spaceUrl, req);
  const headers = new Headers();
  const contentType = firstHeader(req.headers["content-type"]);
  if (contentType) {
    headers.set("Content-Type", contentType);
  }
  if (process.env.HF_TOKEN) {
    headers.set("Authorization", `Bearer ${process.env.HF_TOKEN}`);
  }

  const method = req.method ?? "GET";
  const body =
    method === "GET" || method === "HEAD"
      ? undefined
      : typeof req.body === "string" || req.body instanceof Buffer
        ? req.body
        : JSON.stringify(req.body ?? {});

  const upstream = await fetch(upstreamUrl, { method, headers, body });
  upstream.headers.forEach((value, key) => {
    if (!SKIP_RESPONSE_HEADERS.has(key.toLowerCase())) {
      res.setHeader(key, value);
    }
  });

  const payload = Buffer.from(await upstream.arrayBuffer());
  res.status(upstream.status).send(payload);
}

function buildUpstreamUrl(spaceUrl: string, req: VercelRequest): string {
  const base = new URL(spaceUrl);
  const routePath = routePathFromQuery(req.query.path);
  const requestUrl = new URL(req.url ?? "/", "http://localhost");
  base.pathname = joinPath(base.pathname, routePath);
  base.search = requestUrl.searchParams.toString();
  base.searchParams.delete("path");
  return base.toString();
}

function routePathFromQuery(value: QueryValue): string {
  if (Array.isArray(value)) {
    return value.join("/");
  }
  return value ?? "";
}

function joinPath(basePath: string, routePath: string): string {
  const left = basePath.replace(/\/+$/, "");
  const right = routePath.replace(/^\/+/, "");
  return `${left}/${right}`.replace(/\/{2,}/g, "/");
}

function firstHeader(value: QueryValue): string | null {
  if (Array.isArray(value)) {
    return value[0] ?? null;
  }
  return value ?? null;
}
