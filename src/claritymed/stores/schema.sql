-- Per-user PHI database schema for ClarityMed.
--
-- Applied to each user's SQLite file at:
--     ~/.claritymed/data/users/<user_id>/profile.db
--
-- This file mirrors the SQLModel definitions in
-- src/claritymed/stores/profile.py and exists so the database can be
-- recreated from scratch without booting the Python ORM. Keep them in
-- sync — CLAUDE.md "Schema File Sync" requires updating this file in
-- the same commit as any table/column change.
--
-- Comments live ABOVE each CREATE statement, never inside the parens —
-- SQLite preserves inline comments verbatim in sqlite_master.sql, which
-- would break the "schema.sql round-trips against SQLModel" check from
-- docs/plans/2026-06-06-001-foundation/04-per-user-storage-isolation.md.
--
-- Regenerate (after model edits):
--     uv run python -c "
--     import tempfile, sqlite3
--     from sqlmodel import SQLModel, create_engine
--     from claritymed.stores import profile  # noqa: F401
--     with tempfile.NamedTemporaryFile(suffix='.db') as tf:
--         SQLModel.metadata.create_all(create_engine(f'sqlite:///{tf.name}'))
--         con = sqlite3.connect(tf.name)
--         for row in con.execute(
--             \"SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name\"
--         ):
--             print(row[0] + ';')
--     "


-- ============================================================================
-- Table: allergy
--   One row per known allergy for a single patient. Pydantic contract:
--   claritymed.core.schemas.patient.Allergy.
--
-- Columns:
--   id           Surrogate primary key. SQLite auto-increments INTEGER PRIMARY KEY.
--   user_id      Owner of this row. Defense-in-depth: even if path isolation
--                fails, every read filters WHERE user_id = ?. Indexed.
--   substance    Free-text substance name (e.g. "penicillin"). NOT NULL.
--   severity     Reaction severity. Enum at the app layer:
--                mild | moderate | severe | anaphylactic.
--   source       Provenance of the entry. Enum: self_report | clinical_record.
--   create_time  Row creation timestamp (UTC, set by the ORM at INSERT).
--   update_time  Last mutation timestamp (UTC, refreshed by the ORM on write).
-- ============================================================================
CREATE TABLE allergy (
	id INTEGER NOT NULL,
	user_id VARCHAR NOT NULL,
	substance VARCHAR NOT NULL,
	severity VARCHAR NOT NULL,
	source VARCHAR NOT NULL,
	create_time DATETIME NOT NULL,
	update_time DATETIME NOT NULL,
	PRIMARY KEY (id)
);

-- Every query filters by user_id, so this is the hot-path index.
CREATE INDEX ix_allergy_user_id ON allergy (user_id);


-- ============================================================================
-- Table: condition
--   One row per diagnosed medical condition. Pydantic contract:
--   claritymed.core.schemas.patient.Condition.
--
-- Columns:
--   id           Surrogate primary key.
--   user_id      Owner of this row. Indexed.
--   display      Human-readable condition name (e.g. "Type 2 diabetes"). NOT NULL.
--   code         Optional coded identifier (ICD-10 / SNOMED). Nullable until
--                the ingest pipeline can resolve a code.
--   onset_date   Optional date the condition began. Stored as DATETIME in UTC.
--   create_time  Row creation timestamp (UTC).
--   update_time  Last mutation timestamp (UTC).
-- ============================================================================
CREATE TABLE condition (
	id INTEGER NOT NULL,
	user_id VARCHAR NOT NULL,
	display VARCHAR NOT NULL,
	code VARCHAR,
	onset_date DATETIME,
	create_time DATETIME NOT NULL,
	update_time DATETIME NOT NULL,
	PRIMARY KEY (id)
);

CREATE INDEX ix_condition_user_id ON condition (user_id);


-- ============================================================================
-- Table: medication
--   One row per active medication. Pydantic contract:
--   claritymed.core.schemas.patient.Medication.
--
-- Columns:
--   id           Surrogate primary key.
--   user_id      Owner of this row. Indexed.
--   display      Human-readable drug name (e.g. "metformin 500 mg"). NOT NULL.
--   code         Optional coded identifier (RxNorm / ATC). Nullable.
--   dose         Optional dose string (e.g. "500 mg"). Free text; no parsing.
--   frequency    Optional cadence (e.g. "twice daily"). Free text; no parsing.
--   create_time  Row creation timestamp (UTC).
--   update_time  Last mutation timestamp (UTC).
-- ============================================================================
CREATE TABLE medication (
	id INTEGER NOT NULL,
	user_id VARCHAR NOT NULL,
	display VARCHAR NOT NULL,
	code VARCHAR,
	dose VARCHAR,
	frequency VARCHAR,
	create_time DATETIME NOT NULL,
	update_time DATETIME NOT NULL,
	PRIMARY KEY (id)
);

CREATE INDEX ix_medication_user_id ON medication (user_id);


-- ============================================================================
-- Table: profile
--   Singleton biometric basics for the user (one row per database). Pydantic
--   contract: claritymed.core.schemas.patient.Profile.
--
--   We store birth_date rather than age because age drifts every birthday;
--   a stored birth_date is stable for life and age is derived in code.
--
-- Columns:
--   id           Surrogate primary key.
--   user_id      Owner of this row. UNIQUE — at most one profile per DB.
--                Indexed (the unique index is the user_id lookup index).
--   sex          Optional. Enum at the app layer: female | male | intersex | unknown.
--   weight_kg    Optional. Float, app-layer bounds (0, 500].
--   height_cm    Optional. Float, app-layer bounds (0, 300].
--   birth_date   Optional. DATE; app layer rejects future dates.
--   create_time  Row creation timestamp (UTC).
--   update_time  Last mutation timestamp (UTC, refreshed on every upsert).
-- ============================================================================
CREATE TABLE profile (
	id INTEGER NOT NULL,
	user_id VARCHAR NOT NULL,
	sex VARCHAR,
	weight_kg FLOAT,
	height_cm FLOAT,
	birth_date DATE,
	create_time DATETIME NOT NULL,
	update_time DATETIME NOT NULL,
	PRIMARY KEY (id)
);

-- UNIQUE index doubles as the user_id lookup index — enforces the singleton.
CREATE UNIQUE INDEX ix_profile_user_id ON profile (user_id);
