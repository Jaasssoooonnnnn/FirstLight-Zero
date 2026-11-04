# Competitive facts

These JSON files are reviewed **data** for the game release named in `manifest.json`.
Runtime code loads these facts instead of parsing arbitrary APK releases, mode
assets, inheritance trees, action graphs, or CSV/TOML tables.

On another host, `CR_WORKSPACE_ROOT` points to the matching resource directory.
Updating the game requires a new data export and parity review.

| File | Meaning |
| --- | --- |
| `card_specs.json` | Original card/ability contracts and source provenance |
| `card_logic.json` | Resolved static resource facts and provenance |
| `effects.json` | Buff identities and exact native semantics |
| `projectiles.json` | Projectile identities and native homing configuration |
| `form_evidence.json` | 62 competitive form and ability bindings |
| `model_catalogs.json` | Vocabulary IDs, feature rows, and model input tables |

Only the 122 cards admitted by normal-mode readiness are legal policy cards.
The original 152-row source catalog and its associated evidence identities are
retained as compatibility facts for existing checkpoints. They provide no activity-mode
parser, executor, or alternative training route.

The mechanic operation rows in `model_catalogs.json` are model input features
from the previous compiler, including categorical trigger/condition labels.
They are never evaluated as Python, a rule language, or game logic. The native
engine still executes game mechanics; runtime feature packing and level scaling
remain ordinary reviewed Python functions.
