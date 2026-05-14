import { useEffect, useMemo, useState } from "react";
import {
  Check,
  CircleAlert,
  Clock3,
  Loader2,
  PanelRightClose,
  PanelRightOpen,
  RefreshCw,
  Search,
  Utensils,
} from "lucide-react";
import {
  apiAssetUrl,
  createSession,
  getHealth,
  getHorizons,
  getPreferenceTemplates,
  selectMeal,
} from "./api";
import type {
  CreateSessionResponse,
  DemoSession,
  Nutrition,
  PreferenceTemplate,
  Recommendation,
  SelectedMeal,
} from "./types";

const DEFAULT_GOALS: Nutrition = {
  calories: 2000,
  protein: 100,
  carbs: 240,
  fat: 65,
};

const NUTRITION_LABELS: Array<[keyof Nutrition, string, string]> = [
  ["calories", "Calories", "kcal"],
  ["protein", "Protein", "g"],
  ["carbs", "Carbs", "g"],
  ["fat", "Fat", "g"],
];

const SLOT_LABELS = ["Breakfast", "Lunch", "Dinner"];
const MIN_PREFERENCE_MEALS = 5;

const NUTRITION_PRESETS: Array<{
  id: string;
  label: string;
  goals: Nutrition;
}> = [
  {
    id: "weight_loss",
    label: "Weight loss",
    goals: { calories: 1600, protein: 90, carbs: 150, fat: 55 },
  },
  {
    id: "sedentary_female",
    label: "Sedentary",
    goals: { calories: 1800, protein: 65, carbs: 225, fat: 60 },
  },
  {
    id: "high_protein_lifter",
    label: "High protein",
    goals: { calories: 2500, protein: 180, carbs: 220, fat: 70 },
  },
  {
    id: "active_male",
    label: "Active",
    goals: { calories: 2800, protein: 130, carbs: 320, fat: 80 },
  },
  {
    id: "endurance_athlete",
    label: "Endurance",
    goals: { calories: 3000, protein: 100, carbs: 400, fat: 75 },
  },
];

export default function App() {
  const [health, setHealth] = useState<"idle" | "checking" | "ok" | "error">("idle");
  const [horizons, setHorizons] = useState<number[]>([]);
  const [preferenceTemplates, setPreferenceTemplates] = useState<PreferenceTemplate[]>([]);
  const [horizonInput, setHorizonInput] = useState("");
  const [selectedPreferenceMeals, setSelectedPreferenceMeals] = useState<number[]>([]);
  const [templateQuery, setTemplateQuery] = useState("");
  const [goals, setGoals] = useState<Nutrition>(DEFAULT_GOALS);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [sessionState, setSessionState] = useState<CreateSessionResponse | null>(null);
  const [view, setView] = useState<"setup" | "planning">("setup");
  const [nutritionOpen, setNutritionOpen] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!toast) {
      return;
    }
    const timer = window.setTimeout(() => setToast(null), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    let active = true;
    setHealth("checking");
    Promise.all([getHealth(), getHorizons(), getPreferenceTemplates(128)])
      .then(([, horizonList, templates]) => {
        if (!active) {
          return;
        }
        setHealth("ok");
        setHorizons(horizonList);
        setPreferenceTemplates(templates);
        setSelectedPreferenceMeals((current) =>
          current.length
            ? current
            : templates.slice(0, MIN_PREFERENCE_MEALS).map((meal) => meal.meal_idx),
        );
      })
      .catch((err: Error) => {
        if (!active) {
          return;
        }
        setHealth("error");
        setError(err.message);
      });
    return () => {
      active = false;
    };
  }, []);

  const session = sessionState?.session ?? null;
  const recommendations = sessionState?.recommendations ?? [];
  const maxSupportedHorizon = horizons.length ? Math.max(...horizons) : 21;
  const selectedHorizon = parseHorizonInput(horizonInput, maxSupportedHorizon);
  const mappedPolicyHorizon =
    selectedHorizon === null ? null : mapPolicyHorizon(selectedHorizon, horizons);
  const horizonTooLong =
    horizonInput.trim() !== "" && Number(horizonInput) > maxSupportedHorizon;
  const visiblePreferenceTemplates = useMemo(
    () => filterPreferenceTemplates(preferenceTemplates, templateQuery),
    [preferenceTemplates, templateQuery],
  );
  const selectedPreferenceTemplates = useMemo(
    () =>
      selectedPreferenceMeals
        .map((mealIdx) => preferenceTemplates.find((meal) => meal.meal_idx === mealIdx))
        .filter((meal): meal is PreferenceTemplate => Boolean(meal)),
    [preferenceTemplates, selectedPreferenceMeals],
  );
  const setupReady =
    health === "ok" &&
    selectedHorizon !== null &&
    selectedPreferenceMeals.length >= MIN_PREFERENCE_MEALS;

  function updateGoal(key: keyof Nutrition, value: string) {
    setSelectedPreset(null);
    setGoals((current) => ({ ...current, [key]: Number(value) }));
  }

  function applyPreset(preset: (typeof NUTRITION_PRESETS)[number]) {
    setSelectedPreset(preset.id);
    setGoals(preset.goals);
  }

  function togglePreferenceMeal(mealIdx: number) {
    setSelectedPreferenceMeals((current) =>
      current.includes(mealIdx)
        ? current.filter((item) => item !== mealIdx)
        : [...current, mealIdx],
    );
  }

  async function startSession() {
    if (selectedHorizon === null) {
      setError("Choose a planning horizon first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await createSession({
        num_days: selectedHorizon,
        nutrition_goals: goals,
        preference_meal_indices: selectedPreferenceMeals,
        top_k: 4,
      });
      setSessionState(response);
      setNutritionOpen(false);
      setView("planning");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function chooseRecommendation(option: Recommendation) {
    if (!session) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await selectMeal(session.session_id, option.action, 4);
      setSessionState({
        session: response.session,
        recommendations: response.recommendations,
      });
      const slotName = SLOT_LABELS[session.slot] ?? `Meal ${session.slot + 1}`;
      setToast(`${option.meal.name} added to ${slotName}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className={view === "setup" ? "app-shell setup-view" : "app-shell planning-view"}>
      {view === "setup" ? (
        <section className="setup-panel">
          <header className="setup-header">
            <div className="brand-block">
              <img src="/site-icon.png" alt="" aria-hidden="true" />
              <div>
                <h1>VisionMealRL</h1>
                <p>
                  A reinforcement-learning meal planner that balances nutrition
                  goals, preference, and diversity across a custom horizon.
                </p>
              </div>
            </div>
            <div className="status-row" aria-live="polite">
              <span className={`status-dot ${health}`} />
              <span>
                {health === "ok"
                  ? "Backend connected"
                  : health === "checking"
                    ? "Checking backend"
                    : "Backend unavailable"}
              </span>
            </div>
          </header>

          <section className="form-section">
            <SectionTitle index="01" title="Planning horizon" />
            <label className="horizon-field">
              <span>Number of days</span>
              <div>
                <input
                  type="number"
                  min="1"
                  max={maxSupportedHorizon}
                  step="1"
                  inputMode="numeric"
                  value={horizonInput}
                  placeholder="7"
                  onChange={(event) => setHorizonInput(event.target.value)}
                />
                <small>days</small>
              </div>
            </label>
            <p className={horizonTooLong ? "field-note warning-note" : "field-note"}>
              {horizonTooLong
                ? `This demo currently supports plans up to ${maxSupportedHorizon} days.`
                : selectedHorizon && mappedPolicyHorizon
                  ? `${selectedHorizon}-day plan served by the ${mappedPolicyHorizon}-day checkpoint.`
                  : `Enter a whole number from 1 to ${maxSupportedHorizon}.`}
            </p>
          </section>

          <section className={`form-section ${selectedHorizon === null ? "locked" : ""}`}>
            <SectionTitle index="02" title="Daily nutrition goals" />
            <div className="preset-grid" aria-label="Nutrition goal presets">
              {NUTRITION_PRESETS.map((preset) => (
                <button
                  type="button"
                  key={preset.id}
                  disabled={selectedHorizon === null}
                  className={selectedPreset === preset.id ? "preset-option active" : "preset-option"}
                  onClick={() => applyPreset(preset)}
                >
                  <span>{preset.label}</span>
                  <strong>{preset.goals.calories.toLocaleString()} kcal</strong>
                </button>
              ))}
            </div>
            <div className="goal-grid">
              {NUTRITION_LABELS.map(([key, label, unit]) => (
                <label key={key} className="number-field">
                  <span>{label}</span>
                  <div>
                    <input
                      type="number"
                      min="1"
                      value={goals[key]}
                      disabled={selectedHorizon === null}
                      onChange={(event) => updateGoal(key, event.target.value)}
                    />
                    <small>{unit}</small>
                  </div>
                </label>
              ))}
            </div>
            <p className="field-note">
              {selectedHorizon
                ? `These daily targets repeat across the ${selectedHorizon}-day plan.`
                : "Choose a horizon to unlock daily targets."}
            </p>
          </section>

          <section className={`form-section ${selectedHorizon === null ? "locked" : ""}`}>
            <SectionTitle index="03" title="Build your preference plate" />
            <PreferenceTemplateBoard
              templates={visiblePreferenceTemplates}
              selectedTemplates={selectedPreferenceTemplates}
              selectedMealIndices={selectedPreferenceMeals}
              query={templateQuery}
              disabled={selectedHorizon === null}
              onQueryChange={setTemplateQuery}
              onToggleMeal={togglePreferenceMeal}
            />
            <p className="field-note">
              Select at least {MIN_PREFERENCE_MEALS} meals. The backend mean-pools their
              ingredient, cuisine, and name representations to define your preference.
            </p>
          </section>

          <button
            className="primary-action"
            type="button"
            onClick={startSession}
            disabled={loading || !setupReady}
          >
            {loading && <Loader2 size={18} className="spin" />}
            <span>{session ? "Restart plan" : "Start plan"}</span>
          </button>

          {error && (
            <p className="error-box">
              <CircleAlert size={16} />
              <span>{error}</span>
            </p>
          )}
        </section>
      ) : session ? (
        <section className="planning-panel">
          <PlanningSurface
            session={session}
            recommendations={recommendations}
            loading={loading}
            nutritionOpen={nutritionOpen}
            toast={toast}
            onToggleNutrition={() => setNutritionOpen((open) => !open)}
            onChoose={chooseRecommendation}
            onRestart={() => {
              setView("setup");
              setSessionState(null);
              setToast(null);
            }}
          />
        </section>
      ) : (
        <section className="planning-panel">
          <EmptyState />
        </section>
      )}
    </main>
  );
}

function SectionTitle({ index, title }: { index: string; title: string }) {
  return (
    <div className="section-title">
      <span className="section-index">{index}</span>
      <h2>{title}</h2>
    </div>
  );
}

function PreferenceTemplateBoard({
  templates,
  selectedTemplates,
  selectedMealIndices,
  query,
  disabled,
  onQueryChange,
  onToggleMeal,
}: {
  templates: PreferenceTemplate[];
  selectedTemplates: PreferenceTemplate[];
  selectedMealIndices: number[];
  query: string;
  disabled: boolean;
  onQueryChange: (value: string) => void;
  onToggleMeal: (mealIdx: number) => void;
}) {
  return (
    <div className="preference-builder">
      <div className="preference-toolbar">
        <div className="preference-count">
          <strong>{selectedMealIndices.length}</strong>
          <span>selected</span>
        </div>
        <label className="template-search" htmlFor="preference-template-search">
          <Search size={15} />
          <input
            id="preference-template-search"
            type="search"
            value={query}
            disabled={disabled}
            placeholder="Search meal or cuisine"
            onChange={(event) => onQueryChange(event.target.value)}
          />
        </label>
      </div>

      <div className="selected-template-strip" aria-label="Selected preference meals">
        {selectedTemplates.length ? (
          selectedTemplates.map((meal) => (
            <button
              type="button"
              key={meal.meal_idx}
              disabled={disabled}
              onClick={() => onToggleMeal(meal.meal_idx)}
            >
              <img src={apiAssetUrl(meal.image_url)} alt="" loading="lazy" />
              <span>{meal.name}</span>
            </button>
          ))
        ) : (
          <p>Choose meals below to seed preference.</p>
        )}
      </div>

      <div className="template-grid">
        {templates.map((meal) => {
          const selected = selectedMealIndices.includes(meal.meal_idx);
          return (
            <button
              type="button"
              key={meal.meal_idx}
              disabled={disabled}
              className={selected ? "template-option active" : "template-option"}
              onClick={() => onToggleMeal(meal.meal_idx)}
            >
              <img src={apiAssetUrl(meal.image_url)} alt={meal.name} loading="lazy" />
              <span className="template-check">
                {selected ? <Check size={15} /> : null}
              </span>
              <span className="template-copy">
                <strong>{meal.name}</strong>
                <span>
                  {meal.style || "catalog"} / {formatAmount(meal.nutrition.calories)} kcal
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function PlanningSurface({
  session,
  recommendations,
  loading,
  nutritionOpen,
  toast,
  onToggleNutrition,
  onChoose,
  onRestart,
}: {
  session: DemoSession;
  recommendations: Recommendation[];
  loading: boolean;
  nutritionOpen: boolean;
  toast: string | null;
  onToggleNutrition: () => void;
  onChoose: (option: Recommendation) => void;
  onRestart: () => void;
}) {
  return (
    <div className="planning-stage">
      {toast && (
        <div className="toast" role="status" aria-live="polite">
          <Check size={17} />
          <span>{toast}</span>
        </div>
      )}

      <header className="planning-topbar">
        <h1>{session.done ? "Plan complete" : "Meal selection"}</h1>
        <div className="topbar-actions">
          <MealStepIndicator session={session} />
          <button className="secondary-action" type="button" onClick={onToggleNutrition}>
            {nutritionOpen ? <PanelRightClose size={17} /> : <PanelRightOpen size={17} />}
            <span>{nutritionOpen ? "Hide deficit" : "Show deficit"}</span>
          </button>
          <button className="ghost-action" type="button" onClick={onRestart}>
            <RefreshCw size={16} />
            <span>New setup</span>
          </button>
        </div>
      </header>

      <div className={nutritionOpen ? "planning-grid nutrition-open" : "planning-grid"}>
        <section className="recommendation-panel">
          {session.done ? (
            <CompletionSummary session={session} onRestart={onRestart} />
          ) : (
            <>
              <div className="recommendation-heading">
                <h2>{recommendations.length ? "Recommended meals" : "No remaining meals"}</h2>
                <p>Hover or focus a row to inspect projected deficit after choosing.</p>
              </div>

              <div className="recommendation-grid">
                {recommendations.map((option) => (
                  <MealCard
                    key={option.action}
                    option={option}
                    disabled={loading}
                    onChoose={() => onChoose(option)}
                  />
                ))}
              </div>
            </>
          )}
        </section>

        <aside className={nutritionOpen ? "nutrition-drawer open" : "nutrition-drawer"}>
          <header className="dashboard-header">
            <h2>Nutrition status</h2>
            <button className="icon-action" type="button" onClick={onToggleNutrition} aria-label="Close nutrition drawer">
              <PanelRightClose size={18} />
            </button>
          </header>

          <section className="drawer-block">
            <h3>Daily status</h3>
            <NutritionBoard session={session} />
          </section>
          <MealHistory meals={session.selected_meals} session={session} />
        </aside>
      </div>
    </div>
  );
}

function MealStepIndicator({ session }: { session: DemoSession }) {
  const currentMeal = session.done
    ? session.meals_per_day
    : Math.min(session.slot + 1, session.meals_per_day);
  const policyHorizon =
    Number.isFinite(session.policy_horizon) && session.policy_horizon > 0
      ? session.policy_horizon
      : null;
  return (
    <div className="step-indicator" aria-label="Meal planning progress">
      <Clock3 size={17} />
      <div>
        <strong>
          Day {Math.min(session.day + 1, session.num_days)} / {session.num_days}
        </strong>
        <span>
          Meal {currentMeal} / {session.meals_per_day}
          {policyHorizon !== null && policyHorizon !== session.num_days
            ? ` · ${policyHorizon}d model`
            : ""}
        </span>
      </div>
    </div>
  );
}

function NutritionBoard({ session }: { session: DemoSession }) {
  return (
    <section className="nutrition-board">
      {NUTRITION_LABELS.map(([key, label, unit]) => {
        const consumed = session.current_day_consumed[key];
        const target = session.nutrition_goal[key];
        const deficit = session.current_day_deficit[key];
        const pct = Math.min(100, Math.max(0, (consumed / target) * 100));
        return (
          <article key={key} className="nutrition-meter">
            <div className="meter-head">
              <span>{label}</span>
              <strong>
                {formatAmount(consumed)} / {formatAmount(target)} {unit}
              </strong>
            </div>
            <div className="meter-track">
              <span style={{ width: `${pct}%` }} />
            </div>
            <small className={deficit < 0 ? "over" : "remaining"}>
              {formatDeficit(deficit, unit)}
            </small>
          </article>
        );
      })}
    </section>
  );
}

function MealCard({
  option,
  disabled,
  onChoose,
}: {
  option: Recommendation;
  disabled: boolean;
  onChoose: () => void;
}) {
  return (
    <article className="meal-card" tabIndex={0}>
      <div className="meal-image-wrap">
        <img
          src={apiAssetUrl(option.meal.image_url)}
          alt={option.meal.name}
          loading="lazy"
        />
        <span className="meal-style">{option.meal.style || "catalog"}</span>
      </div>

      <div className="meal-body">
        <h3>{option.meal.name}</h3>
        <dl className="meal-nutrition">
          <NutriFact label="Portion" value={`${option.portion.toFixed(2)}x`} />
          <NutriFact label="Calories" value={formatAmount(option.nutrition.calories)} />
          <NutriFact label="Protein" value={`${formatAmount(option.nutrition.protein)}g`} />
          <NutriFact label="Carbs" value={`${formatAmount(option.nutrition.carbs)}g`} />
          <NutriFact label="Fat" value={`${formatAmount(option.nutrition.fat)}g`} />
        </dl>
      </div>

      <ProjectedDeficit option={option} />

      <button type="button" onClick={onChoose} disabled={disabled}>
        <Utensils size={16} />
        <span>Choose meal</span>
      </button>
    </article>
  );
}

function NutriFact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function ProjectedDeficit({ option }: { option: Recommendation }) {
  return (
    <aside className="projected-deficit" aria-label="Projected deficit after choosing">
      <p>Projected deficit</p>
      <div>
        {NUTRITION_LABELS.map(([key, label, unit]) => {
          const value = option.projected_daily_deficit[key];
          return (
            <span key={key} className={deficitClass(value)}>
              <b>{label}</b>
              {formatSigned(value, unit)}
            </span>
          );
        })}
      </div>
    </aside>
  );
}

function MealHistory({
  meals,
  session,
}: {
  meals: SelectedMeal[];
  session: DemoSession;
}) {
  const [query, setQuery] = useState("");
  const slots = Array.from({ length: session.meals_per_day }, (_, index) => index);
  const normalizedQuery = query.trim().toLowerCase();
  const visibleMeals = meals
    .filter((item) => {
      if (!normalizedQuery) {
        return true;
      }
      const slotName = SLOT_LABELS[item.slot] ?? `Meal ${item.slot + 1}`;
      return [
        item.meal.name,
        item.meal.style,
        `day ${item.day + 1}`,
        slotName,
      ].some((value) => value.toLowerCase().includes(normalizedQuery));
    })
    .slice()
    .reverse();

  return (
    <section className="history-panel">
      <section className="drawer-block">
        <h3>Today</h3>
        <div className="meal-timeline">
          {slots.map((slot) => {
            const meal = meals.filter((item) => item.day === session.day).find((item) => item.slot === slot);
            return (
              <div key={slot} className={meal ? "timeline-item filled" : "timeline-item"}>
                <span>{SLOT_LABELS[slot] ?? `Meal ${slot + 1}`}</span>
                <strong>{meal?.meal.name ?? "Pending"}</strong>
              </div>
            );
          })}
        </div>
      </section>

      <section className="drawer-block history-lookup">
        <label htmlFor="meal-history-search">Meal history</label>
        <div className="history-search">
          <Search size={15} />
          <input
            id="meal-history-search"
            type="search"
            placeholder="Search meal, style, or day"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <div className="history-results">
          {visibleMeals.length ? (
            visibleMeals.map((item) => (
              <article key={`${item.day}-${item.slot}-${item.action}`} className="history-result">
                <div>
                  <strong>{item.meal.name}</strong>
                  <span>
                    Day {item.day + 1}, {SLOT_LABELS[item.slot] ?? `Meal ${item.slot + 1}`}
                  </span>
                </div>
                <span>{item.meal.style || "catalog"}</span>
              </article>
            ))
          ) : (
            <p className="history-empty">
              {meals.length ? "No matching meals." : "No meals selected yet."}
            </p>
          )}
        </div>
      </section>
    </section>
  );
}

function CompletionSummary({
  session,
  onRestart,
}: {
  session: DemoSession;
  onRestart: () => void;
}) {
  const finalDeficit = session.episode_deficit;
  const completedDays = session.completed_day_goal_attainment;
  const daysMet = completedDays.filter((item) => item.within_10_percent).length;
  const dayGroups = Array.from({ length: session.num_days }, (_, day) => ({
    day,
    meals: session.selected_meals.filter((item) => item.day === day),
    attainment: completedDays[day],
  }));

  return (
    <section className="completion-panel">
      <div className="completion-heading">
        <div>
          <h2>Plan complete</h2>
          <p>
            {daysMet} of {session.num_days} days finished within the 10% nutrition threshold.
          </p>
        </div>
        <button className="primary-action" type="button" onClick={onRestart}>
          Start new plan
        </button>
      </div>

      <div className="completion-metrics">
        <SummaryMetric label="Meals selected" value={`${session.selected_meals.length}`} />
        <SummaryMetric label="Days within target" value={`${daysMet}/${session.num_days}`} />
        <SummaryMetric
          label="Final closure"
          value={`${Math.round((1 - normalizedDeficit(finalDeficit, session.nutrition_goal, session.num_days)) * 100)}%`}
        />
      </div>

      <div className="completion-deficit">
        {NUTRITION_LABELS.map(([key, label, unit]) => (
          <span key={key} className={deficitClass(finalDeficit[key])}>
            <b>{label}</b>
            {formatSigned(finalDeficit[key], unit)}
          </span>
        ))}
      </div>

      <div className="completion-days">
        {dayGroups.map(({ day, meals, attainment }) => (
          <article key={day} className="completion-day">
            <header>
              <strong>Day {day + 1}</strong>
              <span>{attainment?.within_10_percent ? "Within target" : "Outside target"}</span>
            </header>
            <div>
              {meals.map((item) => (
                <p key={`${item.day}-${item.slot}-${item.action}`}>
                  <span>{SLOT_LABELS[item.slot] ?? `Meal ${item.slot + 1}`}</span>
                  {item.meal.name}
                </p>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function SummaryMetric({ label, value }: { label: string; value: string }) {
  return (
    <article className="summary-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <h2>Start with a horizon.</h2>
      <p>
        The demo will open a fresh planning session and return four ranked meal
        options with live nutrition-deficit projections.
      </p>
    </div>
  );
}

function parseHorizonInput(value: string, maxSupportedHorizon: number): number | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > maxSupportedHorizon) {
    return null;
  }
  return parsed;
}

function mapPolicyHorizon(numDays: number, horizons: number[]): number | null {
  const available = horizons.slice().sort((a, b) => a - b);
  return available.find((horizon) => horizon >= numDays) ?? null;
}

function deficitClass(value: number): string {
  if (Math.abs(value) < 20) {
    return "near";
  }
  return value < 0 ? "negative" : "positive";
}

function filterPreferenceTemplates(
  templates: PreferenceTemplate[],
  query: string,
): PreferenceTemplate[] {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return templates;
  }
  return templates.filter((meal) =>
    [meal.name, meal.style, meal.meal_type, meal.dish_type]
      .filter(Boolean)
      .some((value) => value.toLowerCase().includes(normalized)),
  );
}

function formatAmount(value: number): string {
  return Math.round(value).toLocaleString();
}

function formatSigned(value: number, unit: string): string {
  const rounded = Math.round(value);
  if (rounded === 0) {
    return `0 ${unit}`;
  }
  return `${rounded > 0 ? "+" : ""}${rounded.toLocaleString()} ${unit}`;
}

function formatDeficit(value: number, unit: string): string {
  if (Math.round(value) < 0) {
    return `${formatAmount(Math.abs(value))} ${unit} overshoot`;
  }
  return `${formatAmount(value)} ${unit} remaining`;
}

function normalizedDeficit(deficit: Nutrition, target: Nutrition, numDays: number): number {
  const deficitSum = NUTRITION_LABELS.reduce(
    (total, [key]) => total + Math.abs(deficit[key]),
    0,
  );
  const targetSum = NUTRITION_LABELS.reduce(
    (total, [key]) => total + Math.abs(target[key] * numDays),
    0,
  );
  return Math.min(1, deficitSum / Math.max(targetSum, 1e-8));
}
