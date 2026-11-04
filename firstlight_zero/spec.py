"""Fixed 3.0 X-Bow mirror: deck, forms, entity vocabulary, board constants and the policy tensor schema."""
from pathlib import Path

XBOW_DECK_ID = '4b838780adc1b30a4e0c8b3fe75a683748cf3dbc0338d89f37eab3ba0933be0e'
POOL = str(Path(__file__).resolve().parent / 'data' / 'xbow-mirror-deck.json')
BATTLE_LEVEL = 11

# Card index order used by the model (not the engine's hand order).
CARDS = (
    27000008,  # X-Bow
    26000000,  # Knight (hero)
    26000001,  # Archers (evo)
    27000006,  # Tesla (evo)
    26000010,  # Skeletons
    26000084,  # Electro Spirit
    28000000,  # Fireball
    28000011,  # The Log
)
CARD_INDEX = {card: i for i, card in enumerate(CARDS)}
NO_CARD = len(CARDS)          # padding / unknown card index
EVO_CARDS = (26000001, 27000006)

# Board entities. Base card ids, runtime form ids, then a catch-all.
ENTITY_IDS = (
    26000000, 203000000, 13000000,   # Knight, Hero Knight, Evo Knight
    26000001, 13000001,              # Archers, Evo Archers
    27000006, 13000102,              # Tesla, Evo Tesla
    27000008,                        # X-Bow
    26000010,                        # Skeletons
    26000084,                        # Electro Spirit
    28000000, 28000011,              # Fireball, Log
)
ENTITY_INDEX = {card: i + 1 for i, card in enumerate(ENTITY_IDS)}   # 0 = padding
ENTITY_OTHER = len(ENTITY_IDS) + 1
ENTITY_VOCAB = ENTITY_OTHER + 1
ENTITY_KINDS = ('troop', 'building', 'projectile')                  # anything else -> other

MAX_ENTITIES = 64
ENTITY_FEATURES = 14
SCALAR_FEATURES = 26
BASE_OPTION_FEATURES = 4
OPTIONS = 5                   # four hand slots + Hero Knight ability
ABILITY_OPTION = 4
OPP_QUEUE = 4

WIDTH, HEIGHT = 18, 32        # tiles
CELLS = WIDTH * HEIGHT
UNITS_X, UNITS_Y = 18000.0, 32000.0
DECISION_TICKS = 5
DELAY_BINS_MS = (0, 50, 100, 150, 200)
BASE_LATENCY_TICKS = 1
FORCE_OPEN_TICK = 300         # 15 s after the battle clock starts at tick 0
GAMEPLAY_END_TICK = 6000

KNIGHT, ARCHERS, TESLA = 26000000, 26000001, 27000006
CARD_NAMES = ('XBow', 'Knight', 'Archers', 'Tesla', 'Skeletons', 'ESpirit', 'Fireball', 'Log')   # CARDS order = option_card index

# Board units: the entity-archetype keys of this mirror, plus towers and a catch-all.
UNIT_KEYS = (
    'form:KnightHero', 'form:Archer', 'form:Archer_EV1', 'form:Skeleton', 'form:ElectroSpirit',
    'form:Xbow', 'form:Tesla', 'form:Tesla_EV1',
    'projectile:10000000',    # Fireball
    'projectile:10000031',    # Log
    'projectile:10000017',    # X-Bow bolt
    'projectile:10000002',    # Archer arrow
    'projectile:3459002866',  # Evo Archer arrow
    'projectile:10000003', 'projectile:10000001',   # Tesla / Evo Tesla zap
    'projectile:10000084',    # Electro Spirit jump
    'projectile:10000032', 'projectile:2780738542',  # Hero Knight / Evo Tesla extras
)
UNIT_INDEX = {key: i + 1 for i, key in enumerate(UNIT_KEYS)}      # 0 = padding
UNIT_PRINCESS = len(UNIT_KEYS) + 1
UNIT_KING = UNIT_PRINCESS + 1
UNIT_OTHER = UNIT_KING + 1
UNIT_VOCAB = UNIT_OTHER + 1
# Coarse kind per unit type, for the explicit count planes: 0 character, 1 building, 2 projectile, 3 tower.
UNIT_KIND = [0] + [1 if k.startswith('form:') and ('Xbow' in k or 'Tesla' in k) else 2 if k.startswith('projectile') else 0
                   for k in UNIT_KEYS] + [3, 3, 0]
MAX_UNITS = 40                 # towers included

# Per-unit runtime features (unit type is a separate embedding).
UNIT_FEATURE_NAMES = ('mine', 'x', 'y', 'hp_ratio', 'hp_over_5000', 'age_over_60s', 'has_target', 'vx', 'vy',
                      'attack_cooldown_remaining', 'attack_windup', 'attack_cooldown_phase',
                      'deploying', 'deploy_remaining', 'projectile_in_flight', 'projectile_target_x', 'projectile_target_y')
# (no projectile damage: the engine never exposes it, and for a fixed deck/level it is a per-type constant.)
UNIT_FEATURES = len(UNIT_FEATURE_NAMES)

SCALAR_NAMES = ('remaining_over_300s', 'elixir_multiplier_over_3', 'overtime', 'tiebreak',
                'my_elixir_over_10', 'reserved_elixir_over_10', 'opp_elixir_estimate_over_10', 'opp_elixir_upper_over_10',
                'my_crowns_over_3', 'opp_crowns_over_3',
                'my_left_hp', 'my_right_hp', 'my_king_hp', 'opp_left_hp', 'opp_right_hp', 'opp_king_hp',
                'my_king_activated', 'opp_king_activated',
                'my_evo_archers_ready', 'my_evo_archers_cycles_over_4', 'my_evo_tesla_ready', 'my_evo_tesla_cycles_over_4', 'my_hero_ability_ready',
                'opp_evo_archers_ready', 'opp_evo_archers_cycles_over_4', 'opp_evo_tesla_ready', 'opp_evo_tesla_cycles_over_4', 'opp_hero_on_board_ability_ready',
                'i_have_played',
                # Hero Knight ability: 1.2 s cast, then 4 s taunt + 5 s shield. The engine exposes neither, but the public tracker
                # records the activation tick for both sides, so the remaining window is deterministic.
                'my_hero_ability_remaining_over_5s', 'opp_hero_ability_remaining_over_5s')
SCALARS = len(SCALAR_NAMES)

CARD_FEATURE_NAMES = ('in_hand', 'is_next', 'pending', 'pending_x', 'pending_y')        # my 8 cards, CARDS order
CARD_FEATURES = len(CARD_FEATURE_NAMES)
OPP_CARD_FEATURE_NAMES = ('available', 'cycling', 'cycle_distance_over_4')             # opponent's 8 cards, public tracker
OPP_CARD_FEATURES = len(OPP_CARD_FEATURE_NAMES)
OPTION_FEATURE_NAMES = ('cost_over_10', 'legal_cell_fraction', 'earliest_bin_over_4')
OPTION_FEATURES = len(OPTION_FEATURE_NAMES)
