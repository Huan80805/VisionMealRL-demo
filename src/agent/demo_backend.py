from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from agent.catalog import MealCatalog, MealTemplate
from agent.config import AgentConfig
from agent.env import MealPlanningEnv
from agent.profiles import TRAIN_STYLES, EVAL_STYLES, make_style_template_lists
from agent.user import SimulatedUser


DEFAULT_CATALOG_DIR = Path("artifacts/catalog/three_component/train")
DEFAULT_RUN_ROOT = Path("runs/agent_three_component_2")
DEFAULT_IMAGE_DIR = Path("artifacts/catalog_demo_images")
SUPPORTED_HORIZONS = (1, 3, 7, 21)
NUTRITION_KEYS = ("calories", "protein", "carbs", "fat")


@dataclass(frozen=True)
class HorizonRuntime:
    num_days: int
    cfg: AgentConfig
    model: Any


@dataclass
class DemoSession:
    session_id: str
    runtime: HorizonRuntime
    env: MealPlanningEnv
    obs: np.ndarray
    preference_meal_indices: tuple[int, ...] = ()
    selected_meals: list[dict[str, Any]] = field(default_factory=list)
    current_day_nutrition: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )
    episode_nutrition: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )
    completed_day_deficits: list[np.ndarray] = field(default_factory=list)
    done: bool = False


class DemoBackend:
    """In-memory backend for the interactive meal-planning demo."""

    def __init__(
        self,
        run_root: Path,
        catalog_dir: Path,
        image_dir: Path,
        default_top_k: int = 4,
    ) -> None:
        self.run_root = Path(run_root)
        self.catalog_dir = Path(catalog_dir)
        self.image_dir = Path(image_dir)
        self.default_top_k = int(default_top_k)
        self.catalog = MealCatalog.load_from_artifact(
            manifest_path=self.catalog_dir / "catalog_manifest.csv",
        )
        self.runtimes = self._load_runtimes()
        self.default_num_days = 1
        style_seed = self.runtimes[self.default_num_days].cfg.seed
        self.style_lists = make_style_template_lists(
            self.catalog,
            style_names=tuple(TRAIN_STYLES) + tuple(EVAL_STYLES),
            per_style=max(
                1,
                self.catalog.num_meals // (len(TRAIN_STYLES) + len(EVAL_STYLES)),
            ),
            seed=style_seed,
        )
        self.sessions: dict[str, DemoSession] = {}

    def _load_runtimes(self) -> dict[int, HorizonRuntime]:
        from stable_baselines3 import DQN

        runtimes: dict[int, HorizonRuntime] = {}
        for num_days in SUPPORTED_HORIZONS:
            run_dir = self.run_root / f"h{num_days}_seed42"
            cfg_path = run_dir / "config.json"
            model_path = run_dir / "dqn_model.zip"
            if not cfg_path.exists() or not model_path.exists():
                continue
            cfg = AgentConfig.from_json(cfg_path)
            if cfg.num_days != num_days:
                raise ValueError(
                    f"{cfg_path} has num_days={cfg.num_days}, expected {num_days}"
                )
            if cfg.num_meals != self.catalog.num_meals:
                raise ValueError(
                    f"{cfg_path} expects {cfg.num_meals} meals, "
                    f"catalog has {self.catalog.num_meals}"
                )
            if cfg.embedding_dim != self.catalog.embedding_dim:
                raise ValueError(
                    f"{cfg_path} expects embedding_dim={cfg.embedding_dim}, "
                    f"catalog has {self.catalog.embedding_dim}"
                )
            runtimes[num_days] = HorizonRuntime(
                num_days=num_days,
                cfg=cfg,
                model=DQN.load(str(model_path)),
            )
        if not runtimes:
            raise ValueError(f"no horizon runtimes found under {self.run_root}")
        return runtimes

    @property
    def styles(self) -> list[str]:
        return sorted(self.style_lists)

    @property
    def horizons(self) -> list[int]:
        return sorted(self.runtimes)

    def preference_templates(self, limit: int = 128) -> dict[str, Any]:
        selected: list[tuple[int, MealTemplate]] = []
        seen: set[int] = set()
        per_style = max(1, int(np.ceil(limit / max(len(self.styles), 1))))
        meal_lookup = {id(meal): idx for idx, meal in enumerate(self.catalog.meals)}
        for style in self.styles:
            for meal in self.style_lists.get(style, ())[:per_style]:
                meal_idx = meal_lookup.get(id(meal))
                if meal_idx is None or meal_idx in seen:
                    continue
                selected.append((meal_idx, meal))
                seen.add(meal_idx)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break
        return {
            "templates": [
                self._preference_template_payload(meal_idx, meal)
                for meal_idx, meal in selected
            ],
        }

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        num_days, policy_num_days = _parse_horizon(payload.get("num_days"), self.runtimes)
        runtime = self.runtimes[policy_num_days]
        cfg = replace(runtime.cfg, num_days=num_days)
        goals = _parse_goals(payload.get("nutrition_goals", {}))
        preference_meal_indices = _parse_meal_indices(
            payload.get("preference_meal_indices"),
            num_meals=self.catalog.num_meals,
        )
        top_k = _parse_positive_int(payload.get("top_k"), self.default_top_k)
        seed = _parse_positive_int(payload.get("seed"), cfg.seed)

        templates = self._templates_for_indices(preference_meal_indices)
        user = SimulatedUser.from_templates(
            templates,
            daily_cal=goals[0],
            daily_protein=goals[1],
            daily_carbs=goals[2],
            daily_fat=goals[3],
            preference_noise=0.0,
            seed=seed,
        )
        env = MealPlanningEnv.from_config(cfg, self.catalog, user)
        obs, _ = env.reset(seed=seed)

        session_id = uuid.uuid4().hex
        session = DemoSession(
            session_id=session_id,
            runtime=runtime,
            env=env,
            obs=obs,
            preference_meal_indices=tuple(preference_meal_indices),
        )
        self.sessions[session_id] = session

        return {
            "session": self._session_status(session),
            "recommendations": self.recommend(session_id, top_k=top_k)["recommendations"],
        }

    def get_session(self, session_id: str) -> dict[str, Any]:
        return {"session": self._session_status(self._get_session(session_id))}

    def recommend(self, session_id: str, top_k: int | None = None) -> dict[str, Any]:
        session = self._get_session(session_id)
        if session.done:
            return {
                "session": self._session_status(session),
                "recommendations": [],
            }

        k = self.default_top_k if top_k is None else int(top_k)
        if k <= 0:
            raise ValueError("top_k must be positive")

        q_values = self._q_values(session)
        allowed_actions = self._ranked_actions(session, q_values)
        options = [self._option_payload(session, action, q_values[action]) for action in allowed_actions[:k]]
        return {
            "session": self._session_status(session),
            "recommendations": options,
        }

    def select(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._get_session(session_id)
        if session.done:
            raise ValueError("session is already complete")
        action = int(payload["action"])
        if action < 0 or action >= session.env.num_actions:
            raise ValueError(f"action {action} outside [0, {session.env.num_actions})")

        next_obs, reward, terminated, truncated, info = session.env.step(action)
        session.obs = next_obs
        session.done = bool(terminated or truncated)

        nutrition = np.asarray(info["nutrition"], dtype=np.float32)
        session.current_day_nutrition += nutrition
        session.episode_nutrition += nutrition

        meal_payload = self._selected_meal_payload(session, action, info, reward)
        session.selected_meals.append(meal_payload)

        if int(info["slot"]) == session.env.meals_per_day - 1:
            day_deficit = session.env.user.daily_target - session.current_day_nutrition
            session.completed_day_deficits.append(day_deficit.astype(np.float32))
            if not session.done:
                session.current_day_nutrition = np.zeros(4, dtype=np.float32)

        top_k = _parse_positive_int(payload.get("top_k"), self.default_top_k)
        return {
            "session": self._session_status(session),
            "selected": meal_payload,
            "recommendations": self.recommend(session_id, top_k=top_k)["recommendations"],
        }

    def _get_session(self, session_id: str) -> DemoSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown session_id {session_id!r}") from exc

    def _templates_for_indices(self, meal_indices: list[int]) -> list[MealTemplate]:
        if not meal_indices:
            raise ValueError("preference_meal_indices must contain at least one meal")
        return [self.catalog.meals[idx] for idx in meal_indices]

    def _q_values(self, session: DemoSession) -> np.ndarray:
        obs_tensor, _ = session.runtime.model.policy.obs_to_tensor(session.obs)
        with torch.no_grad():
            values = session.runtime.model.policy.q_net(obs_tensor)
        return values.detach().cpu().numpy()[0]

    def _ranked_actions(
        self,
        session: DemoSession,
        q_values: np.ndarray,
    ) -> list[int]:
        ranked = np.argsort(q_values)[::-1]
        actions: list[int] = []
        seen_meals: set[int] = set()
        for raw_action in ranked:
            action = int(raw_action)
            meal_idx, _portion = self.env_decode(action, session.env)
            if meal_idx in seen_meals:
                continue
            seen_meals.add(meal_idx)
            actions.append(action)
        return actions

    def env_decode(self, action: int, env: MealPlanningEnv) -> tuple[int, float]:
        meal_idx = action // env.num_portions
        portion_idx = action % env.num_portions
        return meal_idx, float(env.portion_levels[portion_idx])

    def image_path_for_meal(self, meal_idx: int) -> Path:
        if meal_idx < 0 or meal_idx >= self.catalog.num_meals:
            raise ValueError(f"meal_idx {meal_idx} outside [0, {self.catalog.num_meals})")
        catalog_id = self.catalog.meals[meal_idx].catalog_id or f"meal_{meal_idx}"
        for suffix in (".webp", ".jpg", ".jpeg", ".png"):
            path = self.image_dir / f"{catalog_id}{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(f"no demo image found for meal_idx={meal_idx}")

    def _option_payload(
        self,
        session: DemoSession,
        action: int,
        q_value: float,
    ) -> dict[str, Any]:
        meal_idx, portion = self.env_decode(action, session.env)
        meal = self.catalog.meals[meal_idx]
        nutrition = self.catalog.get_nutrition(meal_idx, portion)
        projected_deficit = session.env._daily_deficit - nutrition
        return {
            "action": action,
            "q_value": float(q_value),
            "meal": _meal_payload(meal_idx, meal),
            "portion": portion,
            "nutrition": _nutrition_payload(nutrition),
            "projected_daily_deficit": _nutrition_payload(projected_deficit),
            "projected_goal_attainment": _goal_attainment(
                projected_deficit,
                session.env.user.daily_target,
            ),
        }

    def _selected_meal_payload(
        self,
        session: DemoSession,
        action: int,
        info: dict[str, Any],
        reward: float,
    ) -> dict[str, Any]:
        meal_idx, portion = self.env_decode(action, session.env)
        meal = self.catalog.meals[meal_idx]
        return {
            "action": action,
            "day": int(info["day"]),
            "slot": int(info["slot"]),
            "meal": _meal_payload(meal_idx, meal),
            "portion": portion,
            "nutrition": _nutrition_payload(info["nutrition"]),
            "reward": float(reward),
            "reward_terms": info.get("reward_terms", {}),
        }

    def _session_status(self, session: DemoSession) -> dict[str, Any]:
        env = session.env
        daily_target = env.user.daily_target
        current_deficit = daily_target - session.current_day_nutrition
        episode_target = daily_target * env.num_days
        episode_deficit = episode_target - session.episode_nutrition
        return {
            "session_id": session.session_id,
            "done": session.done,
            "available_horizons": self.horizons,
            "policy_horizon": int(session.runtime.num_days),
            "step": int(env._step_count),
            "horizon": int(env.horizon),
            "day": min(int(env._get_day()), env.num_days - 1),
            "slot": int(env._get_meal_slot()) if not session.done else env.meals_per_day,
            "num_days": int(env.num_days),
            "meals_per_day": int(env.meals_per_day),
            "nutrition_goal": _nutrition_payload(daily_target),
            "current_day_consumed": _nutrition_payload(session.current_day_nutrition),
            "current_day_deficit": _nutrition_payload(current_deficit),
            "episode_consumed": _nutrition_payload(session.episode_nutrition),
            "episode_deficit": _nutrition_payload(episode_deficit),
            "current_goal_attainment": _goal_attainment(current_deficit, daily_target),
            "completed_day_goal_attainment": [
                _goal_attainment(deficit, daily_target)
                for deficit in session.completed_day_deficits
            ],
            "selected_meals": session.selected_meals,
            "preference_meals": [
                _meal_payload(idx, self.catalog.meals[idx])
                for idx in session.preference_meal_indices
            ],
        }

    def _preference_template_payload(
        self,
        meal_idx: int,
        meal: MealTemplate,
    ) -> dict[str, Any]:
        return {
            **_meal_payload(meal_idx, meal),
            "nutrition": _nutrition_payload(meal.nutrition),
            "meal_type": meal.meal_type,
            "dish_type": meal.dish_type,
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    backend: DemoBackend

    def do_OPTIONS(self) -> None:
        self._send_json({}, status=HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/":
                self._send_json({
                    "name": "VisionMealRL agent demo backend",
                    "ok": True,
                    "available_horizons": self.backend.horizons,
                    "endpoints": [
                        "GET /api/health",
                        "GET /api/preference-templates",
                        "GET /api/horizons",
                        "GET /api/meal-image/{meal_idx}",
                        "POST /api/session",
                        "GET /api/session/{session_id}",
                        "POST /api/session/{session_id}/recommend",
                        "POST /api/session/{session_id}/select",
                    ],
                })
            elif path == "/api/health":
                self._send_json({"ok": True})
            elif path == "/api/preference-templates":
                query = parse_qs(urlparse(self.path).query)
                limit = _parse_positive_int(
                    query.get("limit", [128])[0],
                    default=128,
                )
                self._send_json(self.backend.preference_templates(limit=limit))
            elif path == "/api/horizons":
                self._send_json({"horizons": self.backend.horizons})
            elif path.startswith("/api/meal-image/"):
                meal_idx = int(path.removeprefix("/api/meal-image/").strip("/"))
                self._send_file(self.backend.image_path_for_meal(meal_idx))
            elif path.startswith("/api/session/"):
                session_id = path.removeprefix("/api/session/").strip("/")
                self._send_json(self.backend.get_session(session_id))
            else:
                self._send_error(HTTPStatus.NOT_FOUND, f"unknown endpoint {path}")
        except Exception as exc:  # noqa: BLE001 - return JSON errors for demo UX
            self._send_error_for_exception(exc)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            payload = self._read_json()
            if path == "/api/session":
                self._send_json(self.backend.create_session(payload), status=HTTPStatus.CREATED)
                return

            if path.startswith("/api/session/") and path.endswith("/recommend"):
                session_id = path.removeprefix("/api/session/").removesuffix("/recommend").strip("/")
                top_k = payload.get("top_k")
                self._send_json(self.backend.recommend(session_id, top_k=top_k))
                return

            if path.startswith("/api/session/") and path.endswith("/select"):
                session_id = path.removeprefix("/api/session/").removesuffix("/select").strip("/")
                self._send_json(self.backend.select(session_id, payload))
                return

            self._send_error(HTTPStatus.NOT_FOUND, f"unknown endpoint {path}")
        except Exception as exc:  # noqa: BLE001 - return JSON errors for demo UX
            self._send_error_for_exception(exc)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = b"" if status == HTTPStatus.NO_CONTENT else json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "image/webp"
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "public, max-age=86400")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_error_for_exception(self, exc: Exception) -> None:
        if isinstance(exc, (ValueError, KeyError)):
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        elif isinstance(exc, FileNotFoundError):
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        else:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def _parse_goals(payload: dict[str, Any]) -> np.ndarray:
    if not isinstance(payload, dict):
        raise ValueError("nutrition_goals must be an object")
    defaults = {
        "calories": 2000.0,
        "protein": 80.0,
        "carbs": 250.0,
        "fat": 65.0,
    }
    values = []
    for key in NUTRITION_KEYS:
        value = float(payload.get(key, defaults[key]))
        if value <= 0:
            raise ValueError(f"nutrition goal {key} must be positive")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _parse_meal_indices(raw: Any, num_meals: int) -> list[int]:
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("preference_meal_indices must be a list of meal indices")
    indices: list[int] = []
    for item in raw:
        value = int(item)
        if value < 0 or value >= num_meals:
            raise ValueError(
                f"preference meal index {value} outside [0, {num_meals})"
            )
        if value not in indices:
            indices.append(value)
    return indices


def _parse_positive_int(raw: Any, default: int) -> int:
    if raw is None:
        return int(default)
    value = int(raw)
    if value <= 0:
        raise ValueError("expected a positive integer")
    return value


def _parse_horizon(raw: Any, runtimes: dict[int, HorizonRuntime]) -> tuple[int, int]:
    value = int(raw) if raw is not None else min(runtimes)
    if value <= 0:
        raise ValueError("num_days must be positive")

    available = sorted(runtimes)
    max_horizon = max(available)
    if value > max_horizon:
        raise ValueError(
            f"num_days={value} exceeds the demo limit of {max_horizon} days"
        )

    policy_horizon = next(horizon for horizon in available if horizon >= value)
    return value, policy_horizon


def _nutrition_payload(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {key: float(arr[i]) for i, key in enumerate(NUTRITION_KEYS)}


def _meal_payload(meal_idx: int, meal: MealTemplate) -> dict[str, Any]:
    return {
        "meal_idx": int(meal_idx),
        "catalog_id": meal.catalog_id,
        "name": meal.name,
        "style": meal.style,
        "image_path": meal.image_path,
        "image_url": f"/api/meal-image/{int(meal_idx)}",
    }


def _goal_attainment(deficit: np.ndarray, target: np.ndarray) -> dict[str, float | bool]:
    deficit_sum = float(np.abs(deficit).sum())
    threshold = float(0.10 * np.asarray(target).sum())
    score = float(max(0.0, 1.0 - deficit_sum / (np.asarray(target).sum() + 1e-8)))
    return {
        "within_10_percent": bool(deficit_sum < threshold),
        "l1_deficit": deficit_sum,
        "threshold": threshold,
        "closure_score": score,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the interactive agent demo backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--top-k", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = DemoBackend(
        run_root=args.run_root,
        catalog_dir=args.catalog_dir,
        image_dir=args.image_dir,
        default_top_k=args.top_k,
    )
    DemoRequestHandler.backend = backend
    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    print(f"Agent demo backend listening on http://{args.host}:{args.port}")
    print("Available endpoints: GET /api/preference-templates, POST /api/session")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
