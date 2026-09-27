"""
Recipe Finder — "What can I cook with what's in my fridge?"

Two sources:
  1. TheMealDB (free web API, no key needed) — real recipes from the web
  2. A small built-in recipe list — works offline and covers Balkan/Greek
     dishes the web API doesn't have (e.g. Greek fried zucchini with yogurt)

Every recipe is ranked by how much of it you can make with your ingredients:
  match % = recipe ingredients you HAVE / recipe ingredients that matter
Pantry staples (salt, pepper, oil, water...) are assumed to be at home.
"""

import logging
import requests
import streamlit as st

logger = logging.getLogger(__name__)

MEALDB_URL = "https://www.themealdb.com/api/json/v1/1"
CACHE_TTL = 86400  # 24 hours

# Assumed to be in every kitchen — never counted as "missing"
PANTRY_STAPLES = {
    'salt', 'pepper', 'black pepper', 'water', 'oil', 'olive oil',
    'vegetable oil', 'sunflower oil', 'sugar', 'garlic',
}

# Different words for the same ingredient → one canonical name.
# Includes a few Bulgarian names so you can type in either language.
SYNONYMS = {
    'courgette': 'zucchini', 'courgettes': 'zucchini', 'zucchinis': 'zucchini',
    'тиквички': 'zucchini', 'тиквичка': 'zucchini',
    'yoghurt': 'yogurt', 'greek yogurt': 'yogurt', 'кисело мляко': 'yogurt',
    'wheat': 'flour', 'wheat flour': 'flour', 'plain flour': 'flour',
    'all-purpose flour': 'flour', 'self-raising flour': 'flour', 'брашно': 'flour',
    'eggs': 'egg', 'яйца': 'egg', 'яйце': 'egg',
    'tomatoes': 'tomato', 'домати': 'tomato', 'домат': 'tomato',
    'potatoes': 'potato', 'картофи': 'potato',
    'onions': 'onion', 'лук': 'onion',
    'garlic clove': 'garlic', 'garlic cloves': 'garlic', 'чесън': 'garlic',
    'cucumbers': 'cucumber', 'краставица': 'cucumber', 'краставици': 'cucumber',
    'peppers': 'pepper (vegetable)', 'bell pepper': 'pepper (vegetable)',
    'red pepper': 'pepper (vegetable)', 'green pepper': 'pepper (vegetable)',
    'чушки': 'pepper (vegetable)',
    'feta cheese': 'feta', 'white cheese': 'feta', 'сирене': 'feta',
    'кашкавал': 'cheese', 'cheddar cheese': 'cheese',
    'dill': 'dill', 'копър': 'dill',
    'walnuts': 'walnut', 'орехи': 'walnut',
    'mushrooms': 'mushroom', 'гъби': 'mushroom',
    'rice': 'rice', 'ориз': 'rice',
    'milk': 'milk', 'прясно мляко': 'milk',
    'chicken breast': 'chicken', 'chicken breasts': 'chicken', 'пиле': 'chicken',
    'minced beef': 'minced meat', 'ground beef': 'minced meat', 'кайма': 'minced meat',
    'aubergine': 'eggplant', 'aubergines': 'eggplant', 'патладжан': 'eggplant',
    'spring onions': 'green onion', 'scallions': 'green onion',
    'lemon juice': 'lemon', 'лимон': 'lemon',
    'butter': 'butter', 'масло': 'butter',
    'морков': 'carrot', 'моркови': 'carrot', 'целина': 'celery',
    'зеле': 'cabbage', 'карфиол': 'cauliflower', 'броколи': 'broccoli',
    'спанак': 'spinach', 'маруля': 'lettuce', 'салата': 'lettuce',
    'тиква': 'pumpkin', 'цвекло': 'beetroot', 'beet': 'beetroot', 'beets': 'beetroot',
    'репички': 'radish', 'зелен боб': 'green beans', 'грах': 'peas',
    'царевица': 'corn', 'sweetcorn': 'corn', 'маслини': 'olive', 'праз': 'leek',
    'пресен лук': 'green onion', 'люта чушка': 'chili pepper', 'chilli': 'chili pepper',
    'сладък картоф': 'sweet potato', 'ябълка': 'apple', 'ябълки': 'apple',
    'круша': 'pear', 'банан': 'banana', 'портокал': 'orange', 'ягоди': 'strawberry',
    'сметана': 'cream', 'double cream': 'cream', 'heavy cream': 'cream',
    'заквасена сметана': 'sour cream', 'извара': 'cottage cheese',
    'моцарела': 'mozzarella', 'пармезан': 'parmesan',
    'свинско': 'pork', 'телешко': 'beef', 'агнешко': 'lamb', 'пуйка': 'turkey',
    'бекон': 'bacon', 'шунка': 'ham', 'наденица': 'sausage', 'риба': 'fish',
    'сьомга': 'salmon', 'риба тон': 'tuna', 'скариди': 'shrimp', 'prawns': 'shrimp',
    'макарони': 'pasta', 'спагети': 'spaghetti', 'хляб': 'bread',
    'галета': 'breadcrumbs', 'овесени ядки': 'oats', 'rolled oats': 'oats',
    'булгур': 'bulgur', 'кускус': 'couscous', 'кори': 'phyllo dough',
    'filo pastry': 'phyllo dough', 'filo': 'phyllo dough', 'phyllo': 'phyllo dough',
    'царевично брашно': 'cornmeal', 'боб': 'beans', 'нахут': 'chickpeas',
    'леща': 'lentils', 'бадеми': 'almond', 'фъстъци': 'peanut', 'лешници': 'hazelnut',
    'сусам': 'sesame', 'магданоз': 'parsley', 'босилек': 'basil', 'мента': 'mint',
    'джоджен': 'mint', 'риган': 'oregano', 'мащерка': 'thyme', 'чубрица': 'savory',
    'розмарин': 'rosemary', 'кориандър': 'cilantro', 'coriander': 'cilantro',
    'дафинов лист': 'bay leaf', 'bay leaves': 'bay leaf', 'червен пипер': 'paprika',
    'кимион': 'cumin', 'канела': 'cinnamon', 'джинджифил': 'ginger',
    'доматено пюре': 'tomato paste', 'tomato puree': 'tomato paste',
    'chopped tomatoes': 'canned tomatoes', 'оцет': 'vinegar', 'горчица': 'mustard',
    'майонеза': 'mayonnaise', 'mayo': 'mayonnaise', 'мед': 'honey',
    'шоколад': 'chocolate', 'какао': 'cocoa', 'бакпулвер': 'baking powder',
    'сода': 'baking soda', 'мая': 'yeast', 'бульон': 'stock', 'broth': 'stock',
    'chicken stock': 'stock', 'vegetable stock': 'stock', 'вино': 'wine',
}

# Offline recipes — ingredients use canonical names from SYNONYMS
LOCAL_RECIPES = [
    {
        'name': 'Greek Fried Zucchini with Yogurt-Garlic Sauce',
        'ingredients': ['zucchini', 'flour', 'yogurt', 'garlic', 'dill', 'oil', 'salt'],
        'optional': ['dill'],
        'steps': [
            'Slice the zucchini into 5 mm rounds, salt them and leave 15 min; pat dry.',
            'Toss the slices in flour and shake off the excess.',
            'Fry in hot oil 2–3 min per side until golden; drain on paper.',
            'Mix yogurt with crushed garlic, chopped dill and a pinch of salt.',
            'Serve the zucchini warm with the yogurt sauce.',
        ],
        'area': 'Greek / Balkan',
    },
    {
        'name': 'Kolokithokeftedes (Greek Zucchini Fritters)',
        'ingredients': ['zucchini', 'feta', 'egg', 'flour', 'dill', 'green onion', 'oil', 'salt'],
        'optional': ['dill', 'green onion'],
        'steps': [
            'Grate the zucchini, salt it, leave 10 min, then squeeze out the water.',
            'Mix with crumbled feta, egg, chopped dill and onion.',
            'Add flour until the mix holds its shape.',
            'Fry spoonfuls in oil 3 min per side. Serve with yogurt.',
        ],
        'area': 'Greek',
    },
    {
        'name': 'Zucchini Moussaka',
        'ingredients': ['zucchini', 'minced meat', 'onion', 'tomato', 'egg', 'yogurt', 'flour'],
        'steps': [
            'Slice and lightly fry the zucchini.',
            'Brown the minced meat with onion; add chopped tomato and simmer 10 min.',
            'Layer zucchini and meat in a baking dish.',
            'Whisk yogurt, eggs and 2 tbsp flour; pour on top.',
            'Bake at 200 °C for 30–35 min until golden.',
        ],
        'area': 'Balkan',
    },
    {
        'name': 'Tarator (Cold Cucumber-Yogurt Soup)',
        'ingredients': ['yogurt', 'cucumber', 'garlic', 'dill', 'walnut', 'water', 'oil', 'salt'],
        'optional': ['walnut'],
        'steps': [
            'Dice or grate the cucumber.',
            'Thin the yogurt with cold water to soup consistency.',
            'Add cucumber, crushed garlic, dill and chopped walnuts.',
            'Season, add a drizzle of oil and chill before serving.',
        ],
        'area': 'Bulgarian',
    },
    {
        'name': 'Tzatziki',
        'ingredients': ['yogurt', 'cucumber', 'garlic', 'dill', 'olive oil', 'lemon', 'salt'],
        'steps': [
            'Grate the cucumber and squeeze out the water.',
            'Mix with strained yogurt, crushed garlic, dill and a little lemon.',
            'Finish with olive oil. Rest 30 min in the fridge.',
        ],
        'area': 'Greek',
    },
    {
        'name': 'Shopska Salad',
        'ingredients': ['tomato', 'cucumber', 'pepper (vegetable)', 'onion', 'feta', 'oil', 'salt'],
        'steps': [
            'Chop tomatoes, cucumber, peppers and onion.',
            'Season with salt and oil (and a little vinegar if you like).',
            'Grate plenty of white cheese on top.',
        ],
        'area': 'Bulgarian',
    },
    {
        'name': 'Mish-Mash (Peppers, Tomatoes, Eggs & Cheese)',
        'ingredients': ['pepper (vegetable)', 'tomato', 'egg', 'feta', 'onion', 'oil', 'salt'],
        'steps': [
            'Soften chopped onion and peppers in oil.',
            'Add chopped tomatoes and cook 5 min.',
            'Stir in crumbled cheese, then the beaten eggs; stir until just set.',
        ],
        'area': 'Bulgarian',
    },
    {
        'name': 'Yogurt Pancakes (Mekitsi-style)',
        'ingredients': ['yogurt', 'flour', 'egg', 'oil', 'sugar', 'salt'],
        'steps': [
            'Mix yogurt, egg and a pinch of salt; stir in flour to a soft dough.',
            'Leave 20 min, then stretch small pieces by hand.',
            'Fry in hot oil until puffed and golden. Serve with jam or cheese.',
        ],
        'area': 'Bulgarian',
    },
    {
        'name': 'Stuffed Peppers with Rice',
        'ingredients': ['pepper (vegetable)', 'rice', 'onion', 'tomato', 'minced meat', 'oil', 'salt'],
        'steps': [
            'Fry onion and minced meat, add rice and chopped tomato, cook 5 min.',
            'Fill the hollowed peppers with the mixture.',
            'Arrange in a dish, add water to half height, bake at 180 °C for ~50 min.',
        ],
        'area': 'Balkan',
    },
    {
        'name': 'Mushrooms with Garlic and Butter',
        'ingredients': ['mushroom', 'butter', 'garlic', 'dill', 'salt'],
        'steps': [
            'Slice mushrooms and fry in butter until golden.',
            'Add crushed garlic for the last minute; finish with dill.',
        ],
        'area': 'Balkan',
    },
]


# Ingredients offered in the drop-down, grouped so the list is easy to extend.
# Use the canonical names (the right-hand side of SYNONYMS).
COMMON_INGREDIENTS = {
    'Vegetables': [
        'zucchini', 'tomato', 'cucumber', 'pepper (vegetable)', 'chili pepper',
        'onion', 'green onion', 'leek', 'garlic', 'potato', 'sweet potato',
        'carrot', 'celery', 'cabbage', 'cauliflower', 'broccoli', 'spinach',
        'lettuce', 'eggplant', 'mushroom', 'pumpkin', 'beetroot', 'radish',
        'green beans', 'peas', 'corn', 'asparagus', 'kale', 'olive',
    ],
    'Fruit': [
        'lemon', 'lime', 'orange', 'apple', 'pear', 'banana', 'strawberry',
        'blueberry', 'grape', 'peach', 'plum', 'cherry', 'avocado', 'raisin',
    ],
    'Dairy & eggs': [
        'egg', 'milk', 'yogurt', 'butter', 'cream', 'sour cream', 'feta',
        'cheese', 'mozzarella', 'parmesan', 'cream cheese', 'cottage cheese',
        'ricotta',
    ],
    'Meat & fish': [
        'chicken', 'minced meat', 'pork', 'beef', 'lamb', 'turkey', 'bacon',
        'ham', 'sausage', 'fish', 'salmon', 'tuna', 'shrimp',
    ],
    'Grains, pasta & bread': [
        'flour', 'rice', 'pasta', 'spaghetti', 'noodles', 'bread',
        'breadcrumbs', 'oats', 'bulgur', 'couscous', 'quinoa', 'phyllo dough',
        'tortilla', 'cornmeal',
    ],
    'Beans & legumes': [
        'beans', 'white beans', 'chickpeas', 'lentils', 'tofu',
    ],
    'Nuts & seeds': [
        'walnut', 'almond', 'peanut', 'hazelnut', 'sesame', 'sunflower seeds',
    ],
    'Herbs & spices': [
        'dill', 'parsley', 'basil', 'mint', 'oregano', 'thyme', 'rosemary',
        'cilantro', 'bay leaf', 'paprika', 'cumin', 'cinnamon', 'ginger',
        'savory', 'chili powder', 'nutmeg', 'curry powder',
    ],
    'Pantry': [
        'tomato paste', 'canned tomatoes', 'vinegar', 'soy sauce', 'mustard',
        'mayonnaise', 'ketchup', 'honey', 'jam', 'chocolate', 'cocoa',
        'baking powder', 'baking soda', 'yeast', 'stock', 'coconut milk',
        'wine',
    ],
}


# ──────────────────────────────────────────────────────────────
# Ingredient helpers
# ──────────────────────────────────────────────────────────────

def normalize(ingredient: str) -> str:
    """'Courgettes ' → 'zucchini', 'Кисело мляко' → 'yogurt'"""
    name = ingredient.lower().strip()
    return SYNONYMS.get(name, name)


def parse_fridge(text: str, selected: list) -> set:
    """Combine multiselect choices + free text ('zucchini, yogurt, wheat')."""
    items = list(selected or [])
    if text:
        items += [part for part in text.replace(';', ',').replace('\n', ',').split(',')]
    return {normalize(i) for i in items if i and i.strip()}


def _singular(word: str) -> str:
    """Very small plural stripper: carrots→carrot, tomatoes→tomato, peas→pea."""
    if len(word) > 4 and word.endswith('oes'):
        return word[:-2]
    if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def ingredient_matches(recipe_ing: str, fridge: set) -> bool:
    """'plain flour' matches 'flour'; 'greek yogurt' matches 'yogurt'."""
    ing = normalize(recipe_ing)
    if ing in fridge:
        return True
    ing_sing = ' '.join(_singular(w) for w in ing.split())
    words = set(ing_sing.replace('(', ' ').replace(')', ' ').split())
    for f in fridge:
        f_sing = ' '.join(_singular(w) for w in f.split())
        if ' ' in f_sing:
            if f_sing in ing_sing:   # multi-word item, e.g. 'minced meat'
                return True
        elif f_sing in words:        # 'flour' in 'plain flour', 'carrot' in 'carrots'
            return True
    return False


def score_recipe(recipe_ingredients: list, fridge: set, optional=()) -> dict:
    """How many of the recipe's ingredients do we have?"""
    optional = {normalize(o) for o in optional}
    have, missing = [], []
    for ing in recipe_ingredients:
        norm = normalize(ing)
        if ingredient_matches(ing, fridge):
            have.append(ing)
        elif norm in PANTRY_STAPLES or norm in optional:
            continue  # assumed at home / nice-to-have — not counted as missing
        else:
            missing.append(ing)
    counted = len(have) + len(missing)
    return {
        'have': have,
        'missing': missing,
        'match': len(have) / counted if counted else 0.0,
    }


# ──────────────────────────────────────────────────────────────
# Web recipes (TheMealDB)
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _mealdb_get(endpoint: str, params: tuple):
    try:
        r = requests.get(f"{MEALDB_URL}/{endpoint}", params=dict(params), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"TheMealDB request failed ({endpoint} {params}): {e}")
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _mealdb_ingredient_names() -> list:
    """All ingredient names TheMealDB knows (e.g. 'Greek Yogurt', 'Courgettes')."""
    data = _mealdb_get('list.php', (('i', 'list'),))
    if not data or not data.get('meals'):
        return []
    return [m['strIngredient'] for m in data['meals'] if m.get('strIngredient')]


def _mealdb_names_for(fridge_item: str) -> list:
    """Map one fridge item to TheMealDB's names: 'yogurt' → ['Greek Yogurt', 'Yogurt']."""
    names = []
    for db_name in _mealdb_ingredient_names():
        if normalize(db_name) == fridge_item or ingredient_matches(db_name, {fridge_item}):
            names.append(db_name)
    return names[:4]  # keep the number of API calls small


def _parse_meal(meal: dict) -> dict:
    ingredients = []
    for i in range(1, 21):
        ing = (meal.get(f'strIngredient{i}') or '').strip()
        measure = (meal.get(f'strMeasure{i}') or '').strip()
        if ing:
            ingredients.append((ing, measure))
    return {
        'name': meal.get('strMeal', 'Recipe'),
        'ingredients': [i for i, _ in ingredients],
        'measures': ingredients,
        'instructions': meal.get('strInstructions', ''),
        'area': meal.get('strArea') or '',
        'image': meal.get('strMealThumb'),
        'url': meal.get('strSource') or meal.get('strYoutube') or '',
        'source': 'TheMealDB (web)',
    }


def search_web_recipes(fridge: set, max_results: int = 8) -> list:
    """Find web recipes that use as many fridge ingredients as possible."""
    hits = {}  # meal id → number of fridge items it uses
    for item in fridge:
        found_for_item = set()
        for db_name in _mealdb_names_for(item):
            data = _mealdb_get('filter.php', (('i', db_name),))
            for meal in (data or {}).get('meals') or []:
                found_for_item.add(meal['idMeal'])
        for meal_id in found_for_item:
            hits[meal_id] = hits.get(meal_id, 0) + 1

    # Look up full details only for the most promising meals
    best_ids = sorted(hits, key=hits.get, reverse=True)[:max_results * 2]
    recipes = []
    for meal_id in best_ids:
        data = _mealdb_get('lookup.php', (('i', meal_id),))
        meals = (data or {}).get('meals') or []
        if meals:
            recipes.append(_parse_meal(meals[0]))
    return recipes


def search_web_recipes_by_dish(dish_name: str, max_results: int = 3) -> list:
    """Web recipes for a recognised dish name, e.g. 'pizza'."""
    data = _mealdb_get('search.php', (('s', dish_name),))
    meals = (data or {}).get('meals') or []
    return [_parse_meal(m) for m in meals[:max_results]]


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def suggest_recipes(fridge: set, use_web: bool = True, max_missing: int = 3) -> list:
    """Rank local + web recipes by how well they fit the fridge."""
    candidates = [dict(r, source='Built-in') for r in LOCAL_RECIPES]
    if use_web:
        candidates += search_web_recipes(fridge)

    results = []
    for recipe in candidates:
        s = score_recipe(recipe['ingredients'], fridge, recipe.get('optional', ()))
        if not s['have'] or len(s['missing']) > max_missing:
            continue
        results.append({**recipe, **s})

    # Recipes that use MORE of your ingredients first, then best match %,
    # then fewer missing ingredients
    results.sort(key=lambda r: (len(r['have']), r['match'], -len(r['missing'])),
                 reverse=True)
    return results


def render_fridge_page(analyzer):
    """Streamlit UI for the 'What's in my fridge?' mode."""
    st.header("🥕 What's in my fridge?")
    st.markdown("Tell me what you have and I'll suggest recipes — "
                "from the web and from a built-in Balkan/Greek collection.")

    common = [item for group in COMMON_INGREDIENTS.values() for item in group]

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.multiselect("Pick ingredients", common,
                                  default=st.session_state.get('fridge_selected', []))
        extra = st.text_input("…or type them (comma-separated, English or Bulgarian)",
                              placeholder="e.g. zucchini, yogurt, wheat")
    with col2:
        max_missing = st.slider("Max missing ingredients", 0, 6, 2,
                                help="Salt, pepper, oil, water and sugar are assumed to be at home")
        use_web = st.checkbox("🌐 Search recipes on the web", value=True)

    fridge = parse_fridge(extra, selected)
    if not fridge:
        st.info("👆 Add at least one ingredient.")
        return

    st.caption("Using: " + ", ".join(sorted(fridge)))

    if not st.button("🍳 Find recipes", type="primary"):
        return

    with st.spinner("Looking for recipes..."):
        recipes = suggest_recipes(fridge, use_web=use_web, max_missing=max_missing)

    if not recipes:
        st.warning("No recipes found. Try raising 'Max missing ingredients' or adding more items.")
        return

    st.success(f"Found {len(recipes)} recipes")

    for rank, r in enumerate(recipes[:12]):
        icon = "✅" if not r['missing'] else "🟡"
        header = f"{icon} {r['name']} — {r['match']:.0%} match"
        if r['missing']:
            header += f" (missing {len(r['missing'])})"
        with st.expander(header, expanded=(rank == 0)):
            c1, c2 = st.columns([1, 2])
            with c1:
                if r.get('image'):
                    st.image(r['image'], use_container_width=True)
                st.caption(f"{r.get('area', '')} · {r['source']}")

                # Reuse the app's existing health scoring + allergen detection.
                # Health score only for the top 3 — each ingredient may call
                # the USDA API, and DEMO_KEY has a low hourly limit.
                scores = ([analyzer.get_health_score(i) for i in r['ingredients']]
                          if rank < 3 else [])
                if scores:
                    avg = sum(scores) / len(scores)
                    _, emoji, _ = analyzer.get_health_category(round(avg))
                    st.markdown(f"Health: {emoji} **{avg:.1f}/10**")
                allergens = analyzer.detect_allergens(r['ingredients'])
                if allergens:
                    st.markdown("⚠️ Allergens: " + ", ".join(
                        analyzer.allergen_info.get(a, {}).get('emoji', '⚠️') + ' ' + a
                        for a in allergens))
            with c2:
                st.markdown("**You have:** " + ", ".join(r['have']))
                if r['missing']:
                    st.markdown("**You need:** " + ", ".join(r['missing']))
                if r.get('measures'):
                    st.markdown("**Ingredients:**\n" + "\n".join(
                        f"- {m} {i}".strip() for i, m in r['measures']))
                if r.get('steps'):
                    st.markdown("**Steps:**\n" + "\n".join(
                        f"{n}. {s}" for n, s in enumerate(r['steps'], 1)))
                elif r.get('instructions'):
                    st.markdown("**Instructions:**")
                    st.write(r['instructions'])
                if r.get('url'):
                    st.markdown(f"[🔗 Original recipe]({r['url']})")
