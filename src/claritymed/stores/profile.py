"""Per-user PHI store (SQLModel + SQLite).

Each user gets their own SQLite file under ``data/users/<id>/profile.db``.
The schema mirrors the Pydantic contracts in ``core/schemas/patient.py``;
every table also carries a ``user_id`` column as defense-in-depth — if path
isolation ever fails, the ``WHERE user_id = ?`` filter still catches a cross-
user read.

The CLAUDE.md "No JOINs / No FKs / surrogate id + create_time + update_time"
rules apply per the project standard. We deliberately deviate on one point:
``user_id`` is ``str``, not ``BigInteger``, because ClarityMed identifies
users by human-readable handles.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import event
from sqlmodel import Field, Session, SQLModel, create_engine, select

from claritymed.context import user_id_ctx
from claritymed.core.schemas import Allergy, Condition, Medication, Profile
from claritymed.errors import UserIdMismatch
from claritymed.stores.paths import user_db_path, user_root, validate_user_id

_ENGINES: dict[str, object] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProfileRow(SQLModel, table=True):
    """Singleton biometric basics for one user.

    ``user_id`` is UNIQUE so there is at most one row per database. Upsert
    semantics live in ``ProfileStore.upsert_profile``.
    """

    __tablename__ = "profile"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(unique=True, index=True)
    sex: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    birth_date: Optional[date] = None
    create_time: datetime = Field(default_factory=_now)
    update_time: datetime = Field(default_factory=_now)


class AllergyRow(SQLModel, table=True):
    __tablename__ = "allergy"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    substance: str
    severity: str
    source: str
    create_time: datetime = Field(default_factory=_now)
    update_time: datetime = Field(default_factory=_now)


class ConditionRow(SQLModel, table=True):
    __tablename__ = "condition"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    display: str
    code: Optional[str] = None
    onset_date: Optional[datetime] = None
    create_time: datetime = Field(default_factory=_now)
    update_time: datetime = Field(default_factory=_now)


class MedicationRow(SQLModel, table=True):
    __tablename__ = "medication"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    display: str
    code: Optional[str] = None
    dose: Optional[str] = None
    frequency: Optional[str] = None
    create_time: datetime = Field(default_factory=_now)
    update_time: datetime = Field(default_factory=_now)


def _engine_for(user_id: str):
    cached = _ENGINES.get(user_id)
    if cached is not None:
        return cached
    user_root(user_id).mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{user_db_path(user_id)}", echo=False)
    SQLModel.metadata.create_all(engine)
    _ENGINES[user_id] = engine
    return engine


@event.listens_for(SQLModel.metadata, "before_create")
def _enable_foreign_keys(target, connection, **kw):  # noqa: ARG001
    # We have no FKs by policy, but enabling the pragma keeps any future
    # ORM-default FKs honest in this single-user database.
    connection.exec_driver_sql("PRAGMA foreign_keys=ON")


class ProfileStore:
    """CRUD against one user's profile.db. All methods refuse cross-user writes."""

    def __init__(self, user_id: str) -> None:
        self.user_id = validate_user_id(user_id)
        self.engine = _engine_for(self.user_id)

    @classmethod
    def for_current_user(cls) -> "ProfileStore":
        uid = user_id_ctx.get()
        if not uid:
            from claritymed.context import MissingContextError

            raise MissingContextError("user_id_ctx is not set")
        return cls(uid)

    # --- Profile (biometric basics) ----------------------------------

    def get_profile(self) -> Profile | None:
        """Return the user's biometric row, or ``None`` if not set yet."""
        with Session(self.engine) as session:
            stmt = select(ProfileRow).where(ProfileRow.user_id == self.user_id)
            row = session.exec(stmt).first()
        if row is None:
            return None
        return Profile(
            sex=row.sex,  # type: ignore[arg-type]
            weight_kg=row.weight_kg,
            height_cm=row.height_cm,
            birth_date=row.birth_date,
        )

    def upsert_profile(self, profile: Profile, *, owner_user_id: str) -> Profile:
        """Insert or update this user's biometric row. Returns the saved Profile."""
        if owner_user_id != self.user_id:
            raise UserIdMismatch(
                f"upsert_profile called for {owner_user_id!r} on store {self.user_id!r}"
            )
        with Session(self.engine) as session:
            stmt = select(ProfileRow).where(ProfileRow.user_id == self.user_id)
            row = session.exec(stmt).first()
            if row is None:
                row = ProfileRow(user_id=self.user_id)
            row.sex = profile.sex
            row.weight_kg = profile.weight_kg
            row.height_cm = profile.height_cm
            row.birth_date = profile.birth_date
            row.update_time = _now()
            session.add(row)
            session.commit()
        return profile

    # --- Allergy -----------------------------------------------------

    def list_allergies(self) -> list[Allergy]:
        with Session(self.engine) as session:
            stmt = select(AllergyRow).where(AllergyRow.user_id == self.user_id)
            rows = session.exec(stmt).all()
        return [
            Allergy(substance=r.substance, severity=r.severity, source=r.source)  # type: ignore[arg-type]
            for r in rows
        ]

    def add_allergy(self, allergy: Allergy, *, owner_user_id: str) -> Allergy:
        if owner_user_id != self.user_id:
            raise UserIdMismatch(
                f"add_allergy called for {owner_user_id!r} on store {self.user_id!r}"
            )
        row = AllergyRow(
            user_id=self.user_id,
            substance=allergy.substance,
            severity=allergy.severity,
            source=allergy.source,
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()
        return allergy

    # --- Condition (Unit 6: save_condition tool) ---------------------

    def list_conditions(self) -> list[Condition]:
        with Session(self.engine) as session:
            stmt = select(ConditionRow).where(ConditionRow.user_id == self.user_id)
            rows = session.exec(stmt).all()
        return [
            Condition(
                display=r.display,
                code=r.code,
                onset_date=r.onset_date.date() if r.onset_date else None,
            )
            for r in rows
        ]

    def add_condition(self, condition: Condition, *, owner_user_id: str) -> Condition:
        """Append one condition row. Used by ``save_condition`` (Unit 6)."""
        if owner_user_id != self.user_id:
            raise UserIdMismatch(
                f"add_condition called for {owner_user_id!r} on store {self.user_id!r}"
            )
        from datetime import datetime as _dt

        onset = condition.onset_date
        onset_dt = _dt(onset.year, onset.month, onset.day) if onset else None
        row = ConditionRow(
            user_id=self.user_id,
            display=condition.display,
            code=condition.code,
            onset_date=onset_dt,
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()
        return condition

    # --- Medication (Unit 6: save_medication tool) -------------------

    def list_medications(self) -> list[Medication]:
        with Session(self.engine) as session:
            stmt = select(MedicationRow).where(MedicationRow.user_id == self.user_id)
            rows = session.exec(stmt).all()
        return [
            Medication(
                display=r.display,
                code=r.code,
                dose=r.dose,
                frequency=r.frequency,
            )
            for r in rows
        ]

    def add_medication(
        self, medication: Medication, *, owner_user_id: str
    ) -> Medication:
        """Append one medication row. Used by ``save_medication`` (Unit 6)."""
        if owner_user_id != self.user_id:
            raise UserIdMismatch(
                f"add_medication called for {owner_user_id!r} on store {self.user_id!r}"
            )
        row = MedicationRow(
            user_id=self.user_id,
            display=medication.display,
            code=medication.code,
            dose=medication.dose,
            frequency=medication.frequency,
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()
        return medication

    # --- Profile field updates (Unit 6: update_profile_field tool) ---

    def update_profile_field(self, field: str, value, *, owner_user_id: str) -> Profile:
        """Set one ``Profile`` field; returns the updated profile.

        The Patient schema's validators do the coercion (``birth_date``
        from ISO string, ``weight_kg`` from int/str). Storage validates
        only the field name — Pydantic enforces value bounds.
        """
        if owner_user_id != self.user_id:
            raise UserIdMismatch(
                f"update_profile_field called for {owner_user_id!r} on store {self.user_id!r}"
            )
        if field not in Profile.model_fields:
            raise ValueError(f"unknown profile field: {field!r}")
        current = self.get_profile() or Profile()
        # ``model_copy`` validates the change against the field's bounds.
        updated = current.model_copy(update={field: value})
        return self.upsert_profile(updated, owner_user_id=owner_user_id)
