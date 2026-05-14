export type Nutrition = {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
};

export type GoalAttainment = {
  within_10_percent: boolean;
  l1_deficit: number;
  threshold: number;
  closure_score: number;
};

export type Meal = {
  meal_idx: number;
  catalog_id: string;
  name: string;
  style: string;
  image_path: string;
  image_url: string;
};

export type PreferenceTemplate = Meal & {
  nutrition: Nutrition;
  meal_type: string;
  dish_type: string;
};

export type Recommendation = {
  action: number;
  q_value: number;
  meal: Meal;
  portion: number;
  nutrition: Nutrition;
  projected_daily_deficit: Nutrition;
  projected_goal_attainment: GoalAttainment;
};

export type SelectedMeal = {
  action: number;
  day: number;
  slot: number;
  meal: Meal;
  portion: number;
  nutrition: Nutrition;
  reward: number;
  reward_terms: Record<string, number>;
};

export type DemoSession = {
  session_id: string;
  done: boolean;
  step: number;
  horizon: number;
  policy_horizon: number;
  day: number;
  slot: number;
  num_days: number;
  meals_per_day: number;
  preference_meals: Meal[];
  nutrition_goal: Nutrition;
  current_day_consumed: Nutrition;
  current_day_deficit: Nutrition;
  episode_consumed: Nutrition;
  episode_deficit: Nutrition;
  current_goal_attainment: GoalAttainment;
  completed_day_goal_attainment: GoalAttainment[];
  selected_meals: SelectedMeal[];
};

export type CreateSessionPayload = {
  num_days?: number;
  nutrition_goals: Nutrition;
  preference_meal_indices?: number[];
  top_k?: number;
};

export type CreateSessionResponse = {
  session: DemoSession;
  recommendations: Recommendation[];
};

export type SelectMealResponse = {
  session: DemoSession;
  selected: SelectedMeal;
  recommendations: Recommendation[];
};
