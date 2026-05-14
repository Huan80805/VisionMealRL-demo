# VisionMealRL Demo Frontend

Local Vite + React frontend for the interactive agent demo.

The setup flow defines user preference by selecting several example meal
templates. The frontend sends those `meal_idx` values as
`preference_meal_indices`; the backend turns them into the same three-component
preference representation used by agent training.

## API wiring

Browser code calls `/agent-api` by default. During local development,
`vite.config.ts` proxies `/agent-api/*` to `HF_SPACE_URL` when it is set,
or to `http://127.0.0.1:8000` otherwise. When `HF_TOKEN` is present, the
proxy attaches:

```text
Authorization: Bearer $HF_TOKEN
```

The proxy reads `HF_TOKEN` from the repo-root `.env` first, so the token
is not exposed in browser JavaScript.

To use the private Hugging Face Space during local development, put this in
the repo-root `.env` or `frontend/.env.local`:

```text
HF_SPACE_URL=https://your-private-space.hf.space
HF_TOKEN=hf_your_private_space_token
```

To use a local backend instead, create `frontend/.env.local`:

```text
VITE_AGENT_API_BASE=http://127.0.0.1:8000
```

## Vercel deployment

The repo includes `frontend/vercel.json` and `frontend/api/agent-api.ts`.
On Vercel, `/agent-api/*` is rewritten to that serverless proxy. The proxy
forwards requests to Hugging Face and attaches the private token server-side.

Set these Vercel project environment variables:

```text
HF_SPACE_URL=https://your-space-name.hf.space
HF_TOKEN=hf_your_private_space_token
```

Do not set `VITE_AGENT_API_BASE` for the Vercel deployment unless the Space is
public and you intentionally want the browser to call the Space directly. Do not
put `HF_TOKEN` in any `VITE_*` variable.

Recommended Vercel project settings:

```text
Framework Preset: Vite
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
Install Command: npm install
```

## Run

```bash
cd frontend
npm install
npm run dev
```

Then open the Vite URL, usually `http://127.0.0.1:5173`.
