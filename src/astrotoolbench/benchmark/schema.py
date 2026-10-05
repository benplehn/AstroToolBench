"""Versioned, strict task records; scientific invalidity is a valid task outcome."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from .catalog import RECIPES

Unit = Literal["km", "km/s", "s", "km^3/s^2", "rad", "1", "m", "m/s", "min"]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Positive = Annotated[Number, Field(gt=0)]
Nonnegative = Annotated[Number, Field(ge=0)]
Name = Annotated[str, Field(strict=True, min_length=1, pattern=r"^[a-z][a-z0-9_]*$")]
Text = Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]
ScientificTool = Literal[
    "propagate_kepler", "delta_v", "hohmann_transfer", "in_cylindrical_shadow",
    "eclipse_windows", "closest_approach",
]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class Quantity(Record):
    # None deliberately represents missing scientific information in diagnostics.
    # Shape and sign remain tool checks: invalid-state tasks must be representable.
    value: Number | list[Number] | None
    unit: Unit


class NumericContract(Record):
    kind: Literal["numeric"]
    unit: Unit
    absolute_tolerance: Nonnegative | None = None
    relative_tolerance: Nonnegative | None = None

    @model_validator(mode="after")
    def usable_tolerance(self):
        if not (self.absolute_tolerance or self.relative_tolerance):
            raise ValueError("A numeric result needs a positive absolute or relative tolerance.")
        return self


class NumericResult(NumericContract):
    value: Number | Annotated[list[Number], Field(min_length=1)]

    @model_validator(mode="after")
    def zero_requires_absolute_tolerance(self):
        values = self.value if isinstance(self.value, list) else [self.value]
        if not self.absolute_tolerance and any(value == 0 for value in values):
            raise ValueError("Zero reference components require a positive absolute tolerance.")
        return self


class BooleanContract(Record):
    kind: Literal["boolean"]
    unit: Literal["1"]


class BooleanResult(BooleanContract):
    value: StrictBool


Result = Annotated[NumericResult | BooleanResult, Field(discriminator="kind")]


class Success(Record):
    kind: Literal["success"]
    outputs: Annotated[dict[Name, Result], Field(min_length=1)]


OutputContract = Annotated[NumericContract | BooleanContract, Field(discriminator="kind")]


class SuccessContract(Record):
    kind: Literal["success"]
    outputs: Annotated[dict[Name, OutputContract], Field(min_length=1)]


ErrorCode = Literal[
    "invalid_input", "inconsistent_units", "missing_input", "physical_impossibility",
    "unsupported_model", "unachievable_precision",
]


class Failure(Record):
    kind: Literal["error"]
    code: ErrorCode
    reason: Text


class Context(Record):
    frame: Literal["ECI"]
    epoch_s: Number
    body: Literal["Earth"]
    body_radius_km: Positive
    dynamics: Literal["two_body", "J2"]
    shadow: Literal["cylindrical_fixed_sun"]
    requested_time_accuracy_s: Positive | None


class Evidence(Record):
    method: Literal["cartesian_rk4", "analytic", "independent_input_check"]
    rationale: Text
    # Independent verification uses tighter, absolute per-output error budgets.
    absolute_tolerances: dict[Name, Positive]


class TaskMetadata(Record):
    schema_version: Literal["0.1"]
    id: Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_-]+$")]
    domain: Literal["astrodynamics"]
    category: Literal["propagation", "maneuvers", "eclipse", "proximity", "multistep", "diagnostic"]
    difficulty: Literal["simple", "multistep", "diagnostic", "trap"]
    split: Literal["train", "validation", "test"]
    family: Name
    prompt: Text
    context: Context
    inputs: Annotated[dict[Name, Quantity], Field(min_length=1)]
    recipe: Literal[
        "propagation", "delta_v", "circularization", "hohmann", "compare_transfers",
        "shadow", "eclipse", "proximity", "separation", "transfer_propagation",
        "transfer_eclipse", "transfer_arrival_shadow",
    ]
    outputs: list[Name]
    required_tools: Annotated[list[ScientificTool], Field(min_length=1)]
    verification: Evidence
    notes: Text

    @model_validator(mode="after")
    def consistent_recipe(self):
        recipe = RECIPES[self.recipe]
        if set(self.inputs) != set(recipe.inputs):
            raise ValueError("Recipe input names must match the catalog; use null for missing information.")
        if self.required_tools != list(recipe.tools):
            raise ValueError("required_tools must match the scientific recipe in dependency order.")
        if len(self.outputs) != len(set(self.outputs)) or not set(self.outputs) <= recipe.outputs.keys():
            raise ValueError("Output names must be unique and supported by the recipe.")
        return self

    def check_answer_contract(self, contract):
        recipe = RECIPES[self.recipe]
        if isinstance(contract, (Success, SuccessContract)):
            if set(self.outputs) != set(contract.outputs):
                raise ValueError("Selected outputs and expected outputs must match.")
            numeric_keys = set()
            for name, result in contract.outputs.items():
                if result.unit != recipe.outputs[name]:
                    raise ValueError(f"Incorrect output unit for {name}.")
                if (name == "in_shadow") != isinstance(result, BooleanContract):
                    raise ValueError(f"Incorrect result type for {name}.")
                if isinstance(result, NumericContract):
                    numeric_keys.add(name)
            if set(self.verification.absolute_tolerances) != numeric_keys:
                raise ValueError("Every numeric output needs an independent verification error budget.")
            if self.verification.method == "independent_input_check":
                raise ValueError("A successful numerical task needs analytic or numerical verification.")
            supported_methods = {"analytic"}
            if self.recipe == "propagation":
                supported_methods.add("cartesian_rk4")
            elif self.recipe == "transfer_propagation":
                supported_methods = {"cartesian_rk4"}
            if self.verification.method not in supported_methods:
                raise ValueError("The declared verification method is not implemented for this recipe.")
            for name in numeric_keys:
                result = contract.outputs[name]
                if isinstance(result, NumericResult):
                    values = result.value if isinstance(result.value, list) else [result.value]
                    allowed_error = min(max(result.absolute_tolerance or 0.0,
                                            (result.relative_tolerance or 0.0) * abs(value)) for value in values)
                else:
                    # Relative-only budgets can only be compared after computing
                    # the reference value; the final record repeats this check.
                    allowed_error = result.absolute_tolerance if not result.relative_tolerance else None
                if allowed_error is not None and self.verification.absolute_tolerances[name] > allowed_error:
                    raise ValueError("Independent verification must be at least as strict as scoring.")
        else:
            if self.outputs or self.verification.absolute_tolerances:
                raise ValueError("Error outcomes have no numeric outputs or numeric tolerances.")
            if self.verification.method != "independent_input_check":
                raise ValueError("An error outcome needs independent input verification.")
        return self


class BenchmarkTask(TaskMetadata):
    expected: Annotated[Success | Failure, Field(discriminator="kind")]

    @model_validator(mode="after")
    def consistent_contract(self):
        return self.check_answer_contract(self.expected)


class TaskSpecification(TaskMetadata):
    """An authoring record containing contracts, never numerical/Boolean answers."""
    answer_contract: Annotated[SuccessContract | Failure, Field(discriminator="kind")]

    @model_validator(mode="after")
    def consistent_contract(self):
        return self.check_answer_contract(self.answer_contract)
