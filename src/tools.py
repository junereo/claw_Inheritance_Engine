from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .models import PortingBacklog, PortingModule
from .permissions import ToolPermissionContext


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SceneType = Literal["narration", "image", "dialogue", "quote", "emphasis", "nameInput"]


class StoryMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: NonEmptyStr
    subtitle: NonEmptyStr | None = None
    genre: NonEmptyStr | None = None
    synopsis: NonEmptyStr | None = None
    summary: str | None = None


class StoryScene(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: SceneType
    imageDescription: str | None = None
    imageUrl: str | None = None
    text: str | None = None
    speaker: str | None = None
    title: str | None = None
    placeholder: str | None = None
    variableName: str | None = None


class StoryChoicePsychologyMapping(BaseModel):
    boundary_acceptance: int
    action_observation: int
    control_compliance: int
    connection_isolation: int


class StoryChoiceOption(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: NonEmptyStr
    subtext: NonEmptyStr
    reaction: NonEmptyStr
    psychologyMapping: StoryChoicePsychologyMapping


class StoryPhaseChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: NonEmptyStr
    imageDescription: str | None = None
    imageUrl: str | None = None
    choices: list[StoryChoiceOption] = Field(default_factory=list)


class StoryPhase(BaseModel):
    model_config = ConfigDict(extra="allow")

    phaseNumber: int
    scenes: list[StoryScene] = Field(default_factory=list)
    choice: StoryPhaseChoice


class StoryPoster(BaseModel):
    model_config = ConfigDict(extra="allow")

    titleKo: str | None = None
    titleEn: str | None = None
    synopsis: str | None = None
    credit: str | None = None
    imageDescription: str | None = None
    imageUrl: str | None = None
    footerText: str | None = None


class StoryEndingItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    endingId: str | None = None
    conditionHint: str | None = None
    typeBadge: str | None = None
    posterTagline: str | None = None
    lines: list[str] = Field(default_factory=list)


class StoryEndingSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    poster: StoryPoster
    endings: list[StoryEndingItem] = Field(default_factory=list)
    buttons: list[str] = Field(default_factory=list)
    brandText: str | None = None


class StoryJSON(BaseModel):
    model_config = ConfigDict(extra="allow")

    meta: StoryMeta
    phases: list[StoryPhase] = Field(default_factory=list)
    ending: StoryEndingSection


IssueSeverity = Literal["error", "warning"]
CutType = Literal[
    "establishing",
    "environment_focus",
    "dialogue",
    "interaction",
    "reaction",
    "insert",
    "transition",
    "cliffhanger",
]
ShotType = Literal[
    "establishing_wide",
    "wide",
    "medium",
    "medium_close_up",
    "close_up",
    "insert",
    "first_person_pov",
    "over_shoulder",
]
TemporalRelation = Literal["continuous", "short_pause", "time_jump", "flashback", "parallel"]
SpatialRelation = Literal[
    "same_location_same_axis",
    "same_location_focus_shift",
    "same_location_camera_shift",
    "same_location_new_frame",
    "new_location",
]
LayerDeltaType = Literal["add", "remove", "update", "preserve"]
CharacterAssetRole = Literal["protagonist", "named", "group", "extra"]
CharacterRole = Literal["primary", "secondary", "background", "silhouette"]
PropRole = Literal["primary", "secondary", "ambient", "environment_anchor"]
CutObjective = Literal[
    "discover_space",
    "preserve_space_continuity",
    "find_better_composition",
    "tighten_focus",
    "preserve_identity",
    "introduce_character",
    "introduce_prop",
    "fix_costume",
    "fix_prop",
    "fix_interaction",
    "fix_harmony",
    "final_polish",
]


class StoryReviewCheckItem(BaseModel):
    code: NonEmptyStr
    label: NonEmptyStr
    passed: bool
    details: str | None = None


class StoryReviewHandoffNotes(BaseModel):
    location_candidates: list[NonEmptyStr] = Field(default_factory=list)
    recurring_character_candidates: list[NonEmptyStr] = Field(default_factory=list)
    ambiguity_flags: list[NonEmptyStr] = Field(default_factory=list)
    manual_review_points: list[NonEmptyStr] = Field(default_factory=list)


class StoryReviewReport(BaseModel):
    passed: bool
    story_quality_checks: list[StoryReviewCheckItem] = Field(default_factory=list)
    relational_readiness_checks: list[StoryReviewCheckItem] = Field(default_factory=list)
    handoff_notes: StoryReviewHandoffNotes = Field(default_factory=StoryReviewHandoffNotes)
    warnings: list[str] = Field(default_factory=list)


class AnchorObject(BaseModel):
    id: NonEmptyStr
    description: NonEmptyStr
    firstAppearanceCutId: NonEmptyStr | None = None


class LocationMaster(BaseModel):
    id: NonEmptyStr
    label: NonEmptyStr
    baseStructure: NonEmptyStr
    anchors: list[AnchorObject] = Field(default_factory=list)
    defaultPaletteId: NonEmptyStr | None = None
    defaultLightingId: NonEmptyStr | None = None
    notes: str | None = None


class CharacterDNA(BaseModel):
    id: NonEmptyStr
    role: CharacterAssetRole
    silhouetteDescription: NonEmptyStr
    signatureProps: list[NonEmptyStr] = Field(default_factory=list)
    firstAppearanceCutId: NonEmptyStr | None = None
    notes: str | None = None


class PaletteDefinition(BaseModel):
    id: NonEmptyStr
    warmLight: str | None = None
    base: str | None = None
    shadow: str | None = None
    accent: str | None = None
    description: str | None = None


class LightingPreset(BaseModel):
    id: NonEmptyStr
    description: NonEmptyStr
    mood: str | None = None


class GlobalStyleBlock(BaseModel):
    styleBlock: NonEmptyStr
    globalRules: list[NonEmptyStr] = Field(default_factory=list)


class SharedAssets(BaseModel):
    schemaVersion: Literal["1.0"]
    storyId: NonEmptyStr
    storyTitle: NonEmptyStr
    locations: list[LocationMaster] = Field(default_factory=list)
    characters: list[CharacterDNA] = Field(default_factory=list)
    palettes: list[PaletteDefinition] = Field(default_factory=list)
    lightingPresets: list[LightingPreset] = Field(default_factory=list)
    globalStyle: GlobalStyleBlock | None = None


class PaletteSignature(BaseModel):
    warmLight: str | None = None
    base: str | None = None
    shadow: str | None = None
    accent: str | None = None
    paletteId: NonEmptyStr | None = None


class ContinuityLock(BaseModel):
    keepLocation: bool
    keepAnchors: list[NonEmptyStr] = Field(default_factory=list)
    keepCharacters: list[NonEmptyStr] = Field(default_factory=list)
    keepProps: list[NonEmptyStr] = Field(default_factory=list)
    keepPalette: bool
    keepLighting: bool
    keepMood: bool


class FrameRelation(BaseModel):
    inheritsFromCutId: NonEmptyStr | None = None
    inheritsFromFrameId: NonEmptyStr | None = None
    temporalRelation: TemporalRelation
    spatialRelation: SpatialRelation
    cameraShift: str | None = None
    focusShift: str | None = None
    shotType: ShotType
    focusTarget: str | None = None


class CharacterDelta(BaseModel):
    deltaType: LayerDeltaType
    characterId: NonEmptyStr
    role: CharacterRole
    description: NonEmptyStr
    requiredAction: str | None = None
    requiredExpression: str | None = None
    gazeTarget: str | None = None


class PropDelta(BaseModel):
    deltaType: LayerDeltaType
    propId: NonEmptyStr
    role: PropRole
    description: NonEmptyStr
    placement: str | None = None
    interactionOwnerCharacterId: NonEmptyStr | None = None


class ActionDelta(BaseModel):
    deltaType: LayerDeltaType
    actionId: NonEmptyStr
    description: NonEmptyStr
    participants: list[NonEmptyStr] = Field(default_factory=list)


class EditingNote(BaseModel):
    objective: CutObjective
    instruction: NonEmptyStr


class SourceEvidence(BaseModel):
    sourceTextIds: list[NonEmptyStr] = Field(default_factory=list)
    sourceExcerpt: str | None = None


class RelationalCutReviewHints(BaseModel):
    ambiguityFlags: list[NonEmptyStr] = Field(default_factory=list)
    manualReviewRequired: bool = False
    suggestedChecks: list[NonEmptyStr] = Field(default_factory=list)


class RelationalCut(BaseModel):
    schemaVersion: Literal["1.1-local"]
    storyId: NonEmptyStr
    sceneId: NonEmptyStr
    cutId: NonEmptyStr
    cutType: CutType
    locationId: NonEmptyStr
    summary: NonEmptyStr
    notes: str | None = None
    continuityLock: ContinuityLock
    frameRelation: FrameRelation
    paletteSignature: PaletteSignature | None = None
    characterDeltas: list[CharacterDelta] = Field(default_factory=list)
    propDeltas: list[PropDelta] = Field(default_factory=list)
    actionDeltas: list[ActionDelta] = Field(default_factory=list)
    editingNotes: list[EditingNote] = Field(default_factory=list)
    sourceEvidence: SourceEvidence | None = None
    reviewHints: RelationalCutReviewHints | None = None


class RelationalCutsFile(BaseModel):
    schemaVersion: Literal["1.1-local"]
    storyId: NonEmptyStr
    sharedAssetsRef: NonEmptyStr
    cuts: list[RelationalCut] = Field(default_factory=list)


class InheritanceValidationIssue(BaseModel):
    severity: IssueSeverity
    code: NonEmptyStr
    message: NonEmptyStr
    cut_id: str | None = None
    field_path: str | None = None


class InheritanceValidationResult(BaseModel):
    valid: bool
    errors: list[InheritanceValidationIssue] = Field(default_factory=list)
    warnings: list[InheritanceValidationIssue] = Field(default_factory=list)


class ContinuityAppliedTrace(BaseModel):
    kept_location: bool = False
    kept_anchor_ids: list[str] = Field(default_factory=list)
    kept_character_ids: list[str] = Field(default_factory=list)
    kept_palette: str | None = None
    kept_lighting: bool = False
    kept_mood: bool = False
    inherited_from_cut_id: str | None = None
    compression_mode: str = "first-appearance"


class CompiledCutPrompt(BaseModel):
    cut_id: str
    location_id: str
    inherits_from_cut_id: str | None = None
    prompt: str
    word_count: int
    continuity_applied: ContinuityAppliedTrace
    warnings: list[str] = Field(default_factory=list)


class InheritanceCompilePreview(BaseModel):
    story_id: str
    target_model: str
    validation: InheritanceValidationResult
    compiled_cuts: list[CompiledCutPrompt] = Field(default_factory=list)


class ImagePromptArtifactMeta(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    story_title: str = Field(alias="storyTitle")
    story_id: str = Field(alias="storyId")
    genre: str | None = None
    target_model: str = Field(alias="targetModel")
    compiler_version: str = Field(alias="compilerVersion")
    source_shared_assets: str = Field(alias="sourceSharedAssets")
    source_relational_cuts: str = Field(alias="sourceRelationalCuts")
    total_images: int = Field(alias="totalImages")
    notes: str | None = None


class ImagePromptArtifactItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cut_id: str = Field(alias="cutId")
    scene_id: str = Field(alias="sceneId")
    location_id: str = Field(alias="locationId")
    inherits_from_cut_id: str | None = Field(default=None, alias="inheritsFromCutId")
    aspect_ratio: str = Field(alias="aspectRatio")
    word_count: int = Field(alias="wordCount")
    prompt: str
    compilation_trace: dict[str, Any] = Field(alias="compilationTrace")


class ImagePromptArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    meta: ImagePromptArtifactMeta
    prompts: list[ImagePromptArtifactItem] = Field(default_factory=list)
    consistency_report: dict[str, Any] = Field(alias="consistencyReport")


class CompilerInputs(BaseModel):
    shared_assets: SharedAssets
    relational_cuts: RelationalCutsFile


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def _sanitize_text(value: str | None) -> str:
    return " ".join((value or "").replace("\n", " ").split()).strip()


def _count_image_targets(story: StoryJSON) -> int:
    count = 0
    for phase in story.phases:
        count += sum(1 for scene in phase.scenes if scene.type == "image" and _has_text(scene.imageDescription))
        if _has_text(phase.choice.imageDescription):
            count += 1
    if _has_text(story.ending.poster.imageDescription):
        count += 1
    return count


def _slugify(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return lowered or "unnamed"


def _extract_story(story_or_text: StoryJSON | str | dict[str, Any]) -> StoryJSON:
    if isinstance(story_or_text, StoryJSON):
        return story_or_text
    if isinstance(story_or_text, dict):
        return StoryJSON.model_validate(story_or_text)
    normalized = story_or_text.strip()
    if normalized.startswith("{"):
        return StoryJSON.model_validate(json.loads(normalized))
    raise ValueError("Story reviewer requires StoryJSON payload or JSON string.")


def _collect_dialogue_speakers(story: StoryJSON) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()
    for phase in story.phases:
        for scene in phase.scenes:
            if scene.type == "dialogue" and _has_text(scene.speaker):
                speaker = scene.speaker.strip()
                if speaker != "???" and speaker not in seen:
                    seen.add(speaker)
                    speakers.append(speaker)
    return speakers


def _collect_location_candidates(story: StoryJSON) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for phase in story.phases:
        for scene in phase.scenes:
            if scene.type != "image" or not _has_text(scene.imageDescription):
                continue
            snippet = " ".join(scene.imageDescription.strip().split()[:6])
            if snippet and snippet not in seen:
                seen.add(snippet)
                candidates.append(snippet)
            if len(candidates) >= 5:
                return candidates
        if _has_text(phase.choice.imageDescription):
            snippet = " ".join(phase.choice.imageDescription.strip().split()[:6])
            if snippet and snippet not in seen:
                seen.add(snippet)
                candidates.append(snippet)
            if len(candidates) >= 5:
                return candidates
    return candidates


def _all_image_descriptions(story: StoryJSON) -> list[str]:
    descriptions: list[str] = []
    for phase in story.phases:
        for scene in phase.scenes:
            if scene.type == "image" and _has_text(scene.imageDescription):
                descriptions.append(scene.imageDescription.strip())
        if _has_text(phase.choice.imageDescription):
            descriptions.append(phase.choice.imageDescription.strip())
    if _has_text(story.ending.poster.imageDescription):
        descriptions.append(story.ending.poster.imageDescription.strip())
    return descriptions


WORD_BUDGET = 80
COMPILER_VERSION = "v2-relational"

SHOT_TYPE_LABELS: dict[str, str] = {
    "establishing_wide": "establishing wide shot",
    "wide": "wide shot",
    "medium": "medium shot",
    "medium_close_up": "medium close-up shot",
    "close_up": "close-up shot",
    "insert": "insert shot",
    "first_person_pov": "first person POV shot",
    "over_shoulder": "over-the-shoulder shot",
}


@dataclass(slots=True)
class IndexedSharedAssets:
    story_id: str
    locations: dict[str, LocationMaster]
    characters: dict[str, CharacterDNA]
    palettes: dict[str, PaletteDefinition]
    lighting_presets: dict[str, LightingPreset]
    anchors: dict[str, AnchorObject]
    anchor_locations: dict[str, str]


@dataclass(slots=True)
class ResolvedCutContinuity:
    cut: RelationalCut
    inherited_from_cut_id: str | None
    location: LocationMaster
    trace: ContinuityAppliedTrace
    location_clause: str | None
    anchor_clauses: list[str]
    character_clauses: list[str]
    prop_clauses: list[str]
    action_clauses: list[str]
    palette_clause: str | None
    lighting_clause: str | None
    mood_clause: str | None
    warnings: list[str]


def _word_count(value: str) -> int:
    return len(re.findall(r"\S+", value))


def _sameish_description(value: str, *, limit: int) -> str:
    cleaned = _sanitize_text(value)
    lowered = cleaned.lower()
    if lowered.startswith("same ") or lowered.startswith("the same "):
        return cleaned
    words = cleaned.split()
    if words and words[0].lower() in {"a", "an", "the"}:
        words = words[1:]
    core = " ".join(words[:limit]).strip()
    return f"same {core}".strip()


def _shorten_clause(value: str, *, limit: int) -> str:
    cleaned = _sanitize_text(value)
    words = cleaned.split()
    if len(words) <= limit:
        return cleaned
    return " ".join(words[:limit]).strip()


def _join_clauses(clauses: list[str]) -> str:
    normalized = [clause.strip(" .") for clause in clauses if _sanitize_text(clause)]
    return ". ".join(normalized).strip()


def _target_model_label(target_model: str) -> str:
    return {
        "flux-klein-v2": "Flux 2 Klein",
    }.get(target_model, target_model)


def _aspect_ratio_for_cut(cut: RelationalCut) -> str:
    scene_id = cut.sceneId.lower()
    if scene_id.endswith("-hook") or scene_id.endswith("-world"):
        return "9:16"
    if scene_id.endswith("-choice"):
        return "16:9"
    if scene_id.endswith("-ending"):
        return "2:3"
    return "3:4"


def _delta_identifier_groups(cut: RelationalCut) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {"add": [], "remove": [], "update": [], "preserve": []}
    for delta in cut.characterDeltas:
        grouped[delta.deltaType].append(delta.characterId)
    for delta in cut.propDeltas:
        grouped[delta.deltaType].append(delta.propId)
    for delta in cut.actionDeltas:
        grouped[delta.deltaType].append(delta.actionId)
    return {key: value for key, value in grouped.items() if value}


def _trace_to_camel_dict(trace: ContinuityAppliedTrace) -> dict[str, Any]:
    return {
        "keepLocation": trace.kept_location,
        "keepAnchors": trace.kept_anchor_ids,
        "keepCharacters": trace.kept_character_ids,
        "keepPalette": trace.kept_palette,
        "keepLighting": trace.kept_lighting,
        "keepMood": trace.kept_mood,
        "inheritsFrom": trace.inherited_from_cut_id,
    }


def _issue(
    severity: IssueSeverity,
    code: str,
    message: str,
    *,
    cut_id: str | None = None,
    field_path: str | None = None,
) -> InheritanceValidationIssue:
    return InheritanceValidationIssue(
        severity=severity,
        code=code,
        message=message,
        cut_id=cut_id,
        field_path=field_path,
    )


def _copy_cut(cut: RelationalCut, *, inherits_from_cut_id: str | None) -> RelationalCut:
    payload = cut.model_dump()
    payload["frameRelation"]["inheritsFromCutId"] = inherits_from_cut_id
    return RelationalCut.model_validate(payload)


def _find_duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def index_shared_assets(shared_assets: SharedAssets) -> IndexedSharedAssets:
    locations = {item.id: item for item in shared_assets.locations}
    characters = {item.id: item for item in shared_assets.characters}
    palettes = {item.id: item for item in shared_assets.palettes}
    lighting_presets = {item.id: item for item in shared_assets.lightingPresets}
    anchors: dict[str, AnchorObject] = {}
    anchor_locations: dict[str, str] = {}
    for location in shared_assets.locations:
        for anchor in location.anchors:
            anchors[anchor.id] = anchor
            anchor_locations[anchor.id] = location.id
    return IndexedSharedAssets(
        story_id=shared_assets.storyId,
        locations=locations,
        characters=characters,
        palettes=palettes,
        lighting_presets=lighting_presets,
        anchors=anchors,
        anchor_locations=anchor_locations,
    )


def infer_inherits_from_cut_id(cuts: list[RelationalCut]) -> list[RelationalCut]:
    normalized: list[RelationalCut] = []
    last_cut_by_location: dict[str, str] = {}
    for cut in cuts:
        inherits_from_cut_id = cut.frameRelation.inheritsFromCutId
        if inherits_from_cut_id is None:
            inherits_from_cut_id = last_cut_by_location.get(cut.locationId)
        normalized_cut = _copy_cut(cut, inherits_from_cut_id=inherits_from_cut_id)
        normalized.append(normalized_cut)
        last_cut_by_location[cut.locationId] = cut.cutId
    return normalized


def validate_inheritance_inputs(
    shared_assets: SharedAssets,
    relational_cuts: RelationalCutsFile,
) -> InheritanceValidationResult:
    errors: list[InheritanceValidationIssue] = []
    warnings: list[InheritanceValidationIssue] = []
    indexed = index_shared_assets(shared_assets)

    if shared_assets.storyId != relational_cuts.storyId:
        errors.append(
            _issue(
                "error",
                "story_id.mismatch",
                f"Shared assets storyId '{shared_assets.storyId}' does not match relational cuts storyId '{relational_cuts.storyId}'.",
                field_path="storyId",
            )
        )

    for duplicate_id in sorted(_find_duplicates([item.id for item in shared_assets.locations])):
        errors.append(_issue("error", "location_id.duplicate", f"Duplicate location id '{duplicate_id}'.", field_path="locations"))
    for duplicate_id in sorted(_find_duplicates([item.id for item in shared_assets.characters])):
        errors.append(_issue("error", "character_id.duplicate", f"Duplicate character id '{duplicate_id}'.", field_path="characters"))
    for duplicate_id in sorted(_find_duplicates([item.id for item in shared_assets.palettes])):
        errors.append(_issue("error", "palette_id.duplicate", f"Duplicate palette id '{duplicate_id}'.", field_path="palettes"))
    for duplicate_id in sorted(_find_duplicates([item.id for item in shared_assets.lightingPresets])):
        errors.append(_issue("error", "lighting_id.duplicate", f"Duplicate lighting preset id '{duplicate_id}'.", field_path="lightingPresets"))

    anchor_ids = [anchor.id for location in shared_assets.locations for anchor in location.anchors]
    for duplicate_id in sorted(_find_duplicates(anchor_ids)):
        errors.append(_issue("error", "anchor_id.duplicate", f"Duplicate anchor id '{duplicate_id}'.", field_path="locations[].anchors"))

    cut_ids = [cut.cutId for cut in relational_cuts.cuts]
    for duplicate_id in sorted(_find_duplicates(cut_ids)):
        errors.append(_issue("error", "cut_id.duplicate", f"Duplicate cut id '{duplicate_id}'.", field_path="cuts"))

    prop_ids = {prop_delta.propId for cut in relational_cuts.cuts for prop_delta in cut.propDeltas}
    character_ids = set(indexed.characters.keys())
    known_cut_ids = set(cut_ids)

    for location in shared_assets.locations:
        if location.defaultPaletteId and location.defaultPaletteId not in indexed.palettes:
            errors.append(_issue("error", "location.default_palette.missing", f"Location '{location.id}' references unknown palette '{location.defaultPaletteId}'."))
        if location.defaultLightingId and location.defaultLightingId not in indexed.lighting_presets:
            errors.append(_issue("error", "location.default_lighting.missing", f"Location '{location.id}' references unknown lighting preset '{location.defaultLightingId}'."))
        for anchor in location.anchors:
            if anchor.firstAppearanceCutId is None:
                warnings.append(_issue("warning", "anchor.first_appearance.missing", f"Anchor '{anchor.id}' is missing firstAppearanceCutId."))
            elif anchor.firstAppearanceCutId not in known_cut_ids:
                errors.append(_issue("error", "anchor.first_appearance.unknown_cut", f"Anchor '{anchor.id}' references unknown firstAppearanceCutId '{anchor.firstAppearanceCutId}'."))

    for character in shared_assets.characters:
        if character.firstAppearanceCutId is None:
            warnings.append(_issue("warning", "character.first_appearance.missing", f"Character '{character.id}' is missing firstAppearanceCutId."))
        elif character.firstAppearanceCutId not in known_cut_ids:
            errors.append(_issue("error", "character.first_appearance.unknown_cut", f"Character '{character.id}' references unknown firstAppearanceCutId '{character.firstAppearanceCutId}'."))

    for cut in relational_cuts.cuts:
        if cut.storyId != relational_cuts.storyId:
            errors.append(_issue("error", "cut.story_id.mismatch", f"Cut '{cut.cutId}' has storyId '{cut.storyId}' but file storyId is '{relational_cuts.storyId}'.", cut_id=cut.cutId))
        if cut.locationId not in indexed.locations:
            errors.append(_issue("error", "cut.location.unknown", f"Cut '{cut.cutId}' references unknown location '{cut.locationId}'.", cut_id=cut.cutId))
        if cut.frameRelation.inheritsFromCutId and cut.frameRelation.inheritsFromCutId not in known_cut_ids:
            errors.append(_issue("error", "cut.inherits_from.unknown", f"Cut '{cut.cutId}' references unknown inheritsFromCutId '{cut.frameRelation.inheritsFromCutId}'.", cut_id=cut.cutId))
        if cut.paletteSignature and cut.paletteSignature.paletteId and cut.paletteSignature.paletteId not in indexed.palettes:
            errors.append(_issue("error", "cut.palette.unknown", f"Cut '{cut.cutId}' references unknown palette '{cut.paletteSignature.paletteId}'.", cut_id=cut.cutId))

        for anchor_id in cut.continuityLock.keepAnchors:
            if anchor_id not in indexed.anchors:
                errors.append(_issue("error", "cut.keep_anchor.unknown", f"Cut '{cut.cutId}' keeps unknown anchor '{anchor_id}'.", cut_id=cut.cutId))
        for character_id in cut.continuityLock.keepCharacters:
            if character_id not in character_ids:
                errors.append(_issue("error", "cut.keep_character.unknown", f"Cut '{cut.cutId}' keeps unknown character '{character_id}'.", cut_id=cut.cutId))
        for prop_id in cut.continuityLock.keepProps:
            if prop_id not in prop_ids:
                errors.append(_issue("error", "cut.keep_prop.unknown", f"Cut '{cut.cutId}' keeps unknown story-local prop '{prop_id}'.", cut_id=cut.cutId))

        for character_delta in cut.characterDeltas:
            if character_delta.characterId not in character_ids:
                errors.append(_issue("error", "character_delta.character.unknown", f"Cut '{cut.cutId}' references unknown character '{character_delta.characterId}'.", cut_id=cut.cutId))
        for prop_delta in cut.propDeltas:
            if prop_delta.interactionOwnerCharacterId and prop_delta.interactionOwnerCharacterId not in character_ids:
                errors.append(_issue("error", "prop_delta.owner_character.unknown", f"Cut '{cut.cutId}' references unknown interaction owner '{prop_delta.interactionOwnerCharacterId}'.", cut_id=cut.cutId))
        for action_delta in cut.actionDeltas:
            for participant in action_delta.participants:
                if participant not in character_ids and participant not in prop_ids:
                    errors.append(_issue("error", "action_delta.participant.unknown", f"Cut '{cut.cutId}' action '{action_delta.actionId}' references unknown participant '{participant}'.", cut_id=cut.cutId))
        if cut.reviewHints:
            for ambiguity in cut.reviewHints.ambiguityFlags:
                warnings.append(_issue("warning", "cut.ambiguity_flag", ambiguity, cut_id=cut.cutId))

    return InheritanceValidationResult(valid=not errors, errors=errors, warnings=warnings)


def _build_first_appearance_maps(
    shared_assets: SharedAssets,
    cuts: list[RelationalCut],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    anchor_first: dict[str, str] = {}
    for location in shared_assets.locations:
        for anchor in location.anchors:
            if anchor.firstAppearanceCutId:
                anchor_first[anchor.id] = anchor.firstAppearanceCutId
    character_first: dict[str, str] = {
        character.id: character.firstAppearanceCutId
        for character in shared_assets.characters
        if character.firstAppearanceCutId is not None
    }
    prop_first: dict[str, str] = {}
    for cut in cuts:
        for anchor_id in cut.continuityLock.keepAnchors:
            anchor_first.setdefault(anchor_id, cut.cutId)
        for character_id in cut.continuityLock.keepCharacters:
            character_first.setdefault(character_id, cut.cutId)
        for prop_delta in cut.propDeltas:
            prop_first.setdefault(prop_delta.propId, cut.cutId)
            anchor_first.setdefault(prop_delta.propId, cut.cutId)
        for character_delta in cut.characterDeltas:
            character_first.setdefault(character_delta.characterId, cut.cutId)
    return anchor_first, character_first, prop_first


def _find_prop_delta(cut: RelationalCut, prop_id: str) -> PropDelta | None:
    for prop_delta in cut.propDeltas:
        if prop_delta.propId == prop_id:
            return prop_delta
    return None


def _find_character_delta(cut: RelationalCut, character_id: str) -> CharacterDelta | None:
    for character_delta in cut.characterDeltas:
        if character_delta.characterId == character_id:
            return character_delta
    return None


def _find_latest_prop_description(cuts: list[RelationalCut], *, cut_index: int, prop_id: str) -> str | None:
    current_cut = cuts[cut_index]
    current_delta = _find_prop_delta(current_cut, prop_id)
    if current_delta and current_delta.deltaType != "remove":
        return current_delta.description
    for previous_cut in reversed(cuts[:cut_index]):
        previous_delta = _find_prop_delta(previous_cut, prop_id)
        if previous_delta and previous_delta.deltaType != "remove":
            return previous_delta.description
    return None


def _describe_palette(palette: PaletteDefinition | None, *, compact: bool) -> str | None:
    if palette is None:
        return None
    if compact:
        hex_values = [value for value in [palette.base, palette.shadow, palette.warmLight, palette.accent] if value]
        if hex_values:
            return f"palette {' '.join(hex_values[:4])}"
        return palette.id
    return _sanitize_text(palette.description) or _describe_palette(palette, compact=True)


def _describe_lighting(lighting: LightingPreset | None, *, compact: bool) -> str | None:
    if lighting is None:
        return None
    if compact:
        return _shorten_clause(lighting.description, limit=10)
    return _sanitize_text(lighting.description)


def _resolve_normalized_cuts(relational_cuts: RelationalCutsFile) -> RelationalCutsFile:
    normalized = infer_inherits_from_cut_id(relational_cuts.cuts)
    payload = relational_cuts.model_dump()
    payload["cuts"] = [cut.model_dump() for cut in normalized]
    return RelationalCutsFile.model_validate(payload)


def resolve_cut_continuity(
    shared_assets: SharedAssets,
    cuts: list[RelationalCut] | RelationalCutsFile,
    cut_id: str,
) -> ResolvedCutContinuity:
    normalized_cuts = infer_inherits_from_cut_id(cuts.cuts if isinstance(cuts, RelationalCutsFile) else cuts)
    indexed = index_shared_assets(shared_assets)
    cut_index = next((index for index, item in enumerate(normalized_cuts) if item.cutId == cut_id), None)
    if cut_index is None:
        raise ValueError(f"Unknown cut id '{cut_id}'.")

    cut = normalized_cuts[cut_index]
    location = indexed.locations[cut.locationId]
    anchor_first, character_first, prop_first = _build_first_appearance_maps(shared_assets, normalized_cuts)
    inherited_from_cut_id = cut.frameRelation.inheritsFromCutId
    is_reappearance = inherited_from_cut_id is not None or bool(cut.continuityLock.keepAnchors or cut.continuityLock.keepCharacters)
    compression_mode = "reappearance" if is_reappearance else "first-appearance"

    location_clause: str | None = None
    if inherited_from_cut_id is None or cut.continuityLock.keepLocation:
        location_clause = location.baseStructure if compression_mode == "first-appearance" else _sameish_description(location.baseStructure, limit=8)

    anchor_clauses: list[str] = []
    for anchor_id in cut.continuityLock.keepAnchors:
        anchor = indexed.anchors[anchor_id]
        current_prop_delta = _find_prop_delta(cut, anchor_id)
        first_cut_id = anchor_first.get(anchor_id)
        if current_prop_delta and current_prop_delta.deltaType in {"preserve", "update", "add"}:
            description = current_prop_delta.description
        elif first_cut_id == cut.cutId:
            description = anchor.description
        else:
            description = _sameish_description(anchor.description, limit=6)
        anchor_clauses.append(description)

    character_clauses: list[str] = []
    for character_id in cut.continuityLock.keepCharacters:
        character = indexed.characters[character_id]
        current_character_delta = _find_character_delta(cut, character_id)
        first_cut_id = character_first.get(character_id)
        if current_character_delta and current_character_delta.deltaType in {"preserve", "update", "add"}:
            description = current_character_delta.description
        elif first_cut_id == cut.cutId:
            description = character.silhouetteDescription
        else:
            description = _sameish_description(character.silhouetteDescription, limit=10)
        character_clauses.append(description)

    prop_clauses: list[str] = []
    for prop_id in cut.continuityLock.keepProps:
        description = _find_latest_prop_description(normalized_cuts, cut_index=cut_index, prop_id=prop_id)
        if description:
            if prop_first.get(prop_id) == cut.cutId:
                prop_clauses.append(description)
            else:
                prop_clauses.append(_sameish_description(description, limit=6))

    kept_anchor_ids = set(cut.continuityLock.keepAnchors)
    kept_character_ids = set(cut.continuityLock.keepCharacters)
    kept_prop_ids = set(cut.continuityLock.keepProps)

    for prop_delta in cut.propDeltas:
        if prop_delta.deltaType == "remove":
            continue
        if prop_delta.propId in kept_anchor_ids or prop_delta.propId in kept_prop_ids:
            continue
        if prop_first.get(prop_delta.propId) == cut.cutId or prop_delta.deltaType in {"add", "update"}:
            prop_clauses.append(prop_delta.description)
        else:
            prop_clauses.append(_sameish_description(prop_delta.description, limit=6))

    for character_delta in cut.characterDeltas:
        if character_delta.deltaType == "remove":
            continue
        if character_delta.characterId in kept_character_ids:
            continue
        if character_first.get(character_delta.characterId) == cut.cutId or character_delta.deltaType in {"add", "update"}:
            character_clauses.append(character_delta.description)
        else:
            character_clauses.append(_sameish_description(character_delta.description, limit=10))

    action_clauses: list[str] = [delta.description for delta in cut.actionDeltas if delta.deltaType != "remove"]

    palette_id = cut.paletteSignature.paletteId if cut.paletteSignature and cut.paletteSignature.paletteId else location.defaultPaletteId
    palette = indexed.palettes.get(palette_id) if palette_id else None
    palette_clause = None
    if palette and (inherited_from_cut_id is None or cut.continuityLock.keepPalette):
        palette_clause = _describe_palette(palette, compact=False)

    lighting = indexed.lighting_presets.get(location.defaultLightingId) if location.defaultLightingId else None
    lighting_clause = None
    if lighting and (inherited_from_cut_id is None or cut.continuityLock.keepLighting):
        lighting_clause = _describe_lighting(lighting, compact=False)

    mood_clause = None
    if lighting and lighting.mood and (inherited_from_cut_id is None or cut.continuityLock.keepMood):
        mood_clause = _sanitize_text(lighting.mood)

    trace = ContinuityAppliedTrace(
        kept_location=cut.continuityLock.keepLocation,
        kept_anchor_ids=list(cut.continuityLock.keepAnchors),
        kept_character_ids=list(cut.continuityLock.keepCharacters),
        kept_palette=palette_id if cut.continuityLock.keepPalette or inherited_from_cut_id is None else None,
        kept_lighting=cut.continuityLock.keepLighting or inherited_from_cut_id is None,
        kept_mood=cut.continuityLock.keepMood or inherited_from_cut_id is None,
        inherited_from_cut_id=inherited_from_cut_id,
        compression_mode=compression_mode,
    )

    return ResolvedCutContinuity(
        cut=cut,
        inherited_from_cut_id=inherited_from_cut_id,
        location=location,
        trace=trace,
        location_clause=location_clause,
        anchor_clauses=anchor_clauses,
        character_clauses=character_clauses,
        prop_clauses=prop_clauses,
        action_clauses=action_clauses,
        palette_clause=palette_clause,
        lighting_clause=lighting_clause,
        mood_clause=mood_clause,
        warnings=[],
    )


def _build_prompt_clauses(
    shared_assets: SharedAssets,
    resolved: ResolvedCutContinuity,
    *,
    compact: bool,
) -> list[str]:
    global_style = _sanitize_text(shared_assets.globalStyle.styleBlock) if shared_assets.globalStyle else ""
    shot_label = SHOT_TYPE_LABELS.get(resolved.cut.frameRelation.shotType, resolved.cut.frameRelation.shotType.replace("_", " "))
    shot_bits = [shot_label]
    if resolved.cut.frameRelation.cameraShift:
        shot_bits.append(_shorten_clause(resolved.cut.frameRelation.cameraShift, limit=4) if compact else _sanitize_text(resolved.cut.frameRelation.cameraShift))
    if resolved.cut.frameRelation.focusTarget:
        focus_target = _shorten_clause(resolved.cut.frameRelation.focusTarget, limit=5) if compact else _sanitize_text(resolved.cut.frameRelation.focusTarget)
        shot_bits.append(f"focus on {focus_target}")
    shot_clause = ", ".join(bit for bit in shot_bits if bit)

    clauses: list[str] = []
    if global_style:
        clauses.append(_shorten_clause(global_style, limit=5) if compact else global_style)
    if shot_clause:
        clauses.append(shot_clause)
    if resolved.location_clause:
        clauses.append(_shorten_clause(resolved.location_clause, limit=8) if compact else resolved.location_clause)
    for item in resolved.anchor_clauses:
        clauses.append(_shorten_clause(item, limit=5) if compact else item)
    for item in resolved.character_clauses:
        clauses.append(_shorten_clause(item, limit=8) if compact else item)
    for item in resolved.prop_clauses:
        clauses.append(_shorten_clause(item, limit=5) if compact else item)
    for item in resolved.action_clauses:
        clauses.append(_shorten_clause(item, limit=6) if compact else item)
    if resolved.palette_clause:
        indexed = index_shared_assets(shared_assets)
        clauses.append(_describe_palette(indexed.palettes.get(resolved.trace.kept_palette or ""), compact=compact) or resolved.palette_clause)
    if resolved.lighting_clause:
        lighting_already_visible = compact and any(token in " ".join(resolved.anchor_clauses + resolved.prop_clauses).lower() for token in ("lamp", "light"))
        if not lighting_already_visible:
            clauses.append(_shorten_clause(resolved.lighting_clause, limit=6) if compact else resolved.lighting_clause)
    if resolved.mood_clause:
        clauses.append(_shorten_clause(resolved.mood_clause, limit=2) if compact else resolved.mood_clause)
    if shared_assets.globalStyle:
        rules = [_sanitize_text(rule) for rule in shared_assets.globalStyle.globalRules]
        text_rule = next((rule for rule in rules if "text" in rule.lower() or "letter" in rule.lower()), "")
        if text_rule:
            clauses.append("no visible text")
    return clauses


def compile_cut_preview(
    shared_assets: SharedAssets,
    relational_cuts: RelationalCutsFile,
    cut_id: str,
    *,
    target_model: str = "flux-klein-v2",
) -> CompiledCutPrompt:
    cuts_file = _resolve_normalized_cuts(relational_cuts)
    validation = validate_inheritance_inputs(shared_assets, cuts_file)
    if not validation.valid:
        raise ValueError("; ".join(issue.message for issue in validation.errors))

    resolved = resolve_cut_continuity(shared_assets, cuts_file, cut_id)
    warnings = [issue.message for issue in validation.warnings if issue.cut_id in {None, cut_id}]
    prompt = _join_clauses(_build_prompt_clauses(shared_assets, resolved, compact=False))
    word_count = _word_count(prompt)
    if word_count > WORD_BUDGET:
        warnings.append(f"Prompt exceeded {WORD_BUDGET} words at {word_count}; compact compression reapplied for preview output.")
        prompt = _join_clauses(_build_prompt_clauses(shared_assets, resolved, compact=True))
        word_count = _word_count(prompt)
        resolved.trace.compression_mode = "budget-reduced"

    return CompiledCutPrompt(
        cut_id=resolved.cut.cutId,
        location_id=resolved.cut.locationId,
        inherits_from_cut_id=resolved.inherited_from_cut_id,
        prompt=prompt,
        word_count=word_count,
        continuity_applied=resolved.trace,
        warnings=warnings,
    )


def compile_story_preview(
    shared_assets: SharedAssets,
    relational_cuts: RelationalCutsFile,
    *,
    target_model: str = "flux-klein-v2",
) -> InheritanceCompilePreview:
    cuts_file = _resolve_normalized_cuts(relational_cuts)
    validation = validate_inheritance_inputs(shared_assets, cuts_file)
    if not validation.valid:
        raise ValueError("; ".join(issue.message for issue in validation.errors))
    compiled_cuts = [
        compile_cut_preview(shared_assets, cuts_file, cut.cutId, target_model=target_model)
        for cut in cuts_file.cuts
    ]
    return InheritanceCompilePreview(
        story_id=shared_assets.storyId,
        target_model=target_model,
        validation=validation,
        compiled_cuts=compiled_cuts,
    )


def _build_consistency_report(
    shared_assets: SharedAssets,
    relational_cuts: RelationalCutsFile,
    compiled_preview: InheritanceCompilePreview,
) -> dict[str, Any]:
    same_location_reuse: list[dict[str, Any]] = []
    cuts_by_location: dict[str, list[RelationalCut]] = {}
    for cut in relational_cuts.cuts:
        cuts_by_location.setdefault(cut.locationId, []).append(cut)

    for location_id, cuts in cuts_by_location.items():
        inherited_cuts = [cut for cut in cuts if cut.frameRelation.inheritsFromCutId]
        if not inherited_cuts:
            continue
        first_cut = cuts[0]
        repeat_cut = inherited_cuts[0]
        compiled_cut = next((item for item in compiled_preview.compiled_cuts if item.cut_id == repeat_cut.cutId), None)
        same_location_reuse.append(
            {
                "locationId": location_id,
                "cuts": [cut.cutId for cut in cuts],
                "firstAppearance": first_cut.cutId,
                "reappearance": repeat_cut.cutId,
                "compressionApplied": compiled_cut.continuity_applied.compression_mode != "first-appearance"
                if compiled_cut
                else True,
                "sharedAnchors": repeat_cut.continuityLock.keepAnchors,
                "sharedCharacters": repeat_cut.continuityLock.keepCharacters,
            }
        )

    palette_consistency: dict[str, Any] | None = None
    if shared_assets.palettes:
        first_palette = shared_assets.palettes[0]
        palette_consistency = {
            "paletteId": first_palette.id,
            "usedIn": f"all {len(compiled_preview.compiled_cuts)} cuts",
            "hexCodes": {
                "warmLight": first_palette.warmLight,
                "base": first_palette.base,
                "shadow": first_palette.shadow,
                "accent": first_palette.accent,
            },
        }

    return {
        "sameLocationReuse": same_location_reuse,
        "crossLocationBuildingConsistency": None,
        "paletteConsistency": palette_consistency,
    }


def assemble_image_prompt_artifact(
    shared_assets: SharedAssets,
    relational_cuts: RelationalCutsFile,
    compiled_preview: InheritanceCompilePreview,
    *,
    story_title: str | None = None,
    genre: str | None = None,
    target_model: str = "flux-klein-v2",
) -> ImagePromptArtifact:
    normalized_cuts = _resolve_normalized_cuts(relational_cuts)
    cut_by_id = {cut.cutId: cut for cut in normalized_cuts.cuts}

    artifact_items: list[ImagePromptArtifactItem] = []
    for compiled_cut in compiled_preview.compiled_cuts:
        source_cut = cut_by_id[compiled_cut.cut_id]
        item_trace: dict[str, Any] = {
            "continuityApplied": _trace_to_camel_dict(compiled_cut.continuity_applied),
            "deltasApplied": _delta_identifier_groups(source_cut),
            "compressionMode": compiled_cut.continuity_applied.compression_mode,
        }
        if compiled_cut.warnings:
            item_trace["warnings"] = compiled_cut.warnings

        artifact_items.append(
            ImagePromptArtifactItem(
                cut_id=compiled_cut.cut_id,
                scene_id=source_cut.sceneId,
                location_id=compiled_cut.location_id,
                inherits_from_cut_id=compiled_cut.inherits_from_cut_id,
                aspect_ratio=_aspect_ratio_for_cut(source_cut),
                word_count=compiled_cut.word_count,
                prompt=compiled_cut.prompt,
                compilation_trace=item_trace,
            )
        )

    return ImagePromptArtifact(
        meta=ImagePromptArtifactMeta(
            story_title=story_title or shared_assets.storyTitle,
            story_id=shared_assets.storyId,
            genre=genre,
            target_model=_target_model_label(target_model),
            compiler_version=COMPILER_VERSION,
            source_shared_assets=f"./{shared_assets.storyId}.shared-assets.json",
            source_relational_cuts=f"./{shared_assets.storyId}.relational-cuts.json",
            total_images=len(artifact_items),
            notes="Deterministic inheritance compile assembled into image-prompts-flux-v2 artifact shape.",
        ),
        prompts=artifact_items,
        consistency_report=_build_consistency_report(shared_assets, normalized_cuts, compiled_preview),
    )


def validate_story_json(story: StoryJSON) -> dict[str, Any]:
    """
    Validates the generator 3-phase story contract used by the Inheritance Engine.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if len(story.phases) != 3:
        errors.append(f"Story must have exactly 3 phases, found {len(story.phases)}.")

    expected_phase_numbers = [1, 2, 3]
    actual_phase_numbers = [phase.phaseNumber for phase in story.phases]
    if actual_phase_numbers != expected_phase_numbers:
        errors.append(f"Phase numbers must be exactly {expected_phase_numbers}, found {actual_phase_numbers}.")

    for phase in story.phases:
        if not (5 <= len(phase.scenes) <= 30):
            errors.append(
                f"Phase {phase.phaseNumber} must contain 5-30 scenes, found {len(phase.scenes)}."
            )

        if len(phase.choice.choices) != 2:
            errors.append(
                f"Phase {phase.phaseNumber} choice must contain exactly 2 options, found {len(phase.choice.choices)}."
            )

        image_scene_count = 0
        for scene_index, scene in enumerate(phase.scenes, start=1):
            if scene.type == "image":
                if not _has_text(scene.imageDescription):
                    errors.append(
                        f"Phase {phase.phaseNumber} scene {scene_index} is image but imageDescription is missing."
                    )
                else:
                    image_scene_count += 1
            elif scene.type in {"narration", "quote", "emphasis"} and not _has_text(scene.text):
                errors.append(
                    f"Phase {phase.phaseNumber} scene {scene_index} of type {scene.type} must include text."
                )
            elif scene.type == "dialogue" and (not _has_text(scene.text) or not _has_text(scene.speaker)):
                errors.append(
                    f"Phase {phase.phaseNumber} scene {scene_index} dialogue must include speaker and text."
                )
            elif scene.type == "nameInput" and (
                not _has_text(scene.placeholder) or not _has_text(scene.variableName)
            ):
                errors.append(
                    f"Phase {phase.phaseNumber} scene {scene_index} nameInput must include placeholder and variableName."
                )

        if image_scene_count < 2:
            warnings.append(
                f"Phase {phase.phaseNumber} has only {image_scene_count} image scenes; relational cut extraction may be weak."
            )

        if not _has_text(phase.choice.question):
            errors.append(f"Phase {phase.phaseNumber} choice question is missing.")

    if len(story.ending.endings) < 3 or len(story.ending.endings) > 4:
        errors.append(
            f"Ending section must contain 3-4 endings, found {len(story.ending.endings)}."
        )

    if not story.ending.buttons:
        warnings.append("Ending buttons are empty.")

    if not _has_text(story.ending.brandText):
        warnings.append("Ending brandText is missing.")

    if not _has_text(story.ending.poster.imageDescription):
        errors.append("Ending poster imageDescription is missing.")

    for ending_index, ending in enumerate(story.ending.endings, start=1):
        if len(ending.lines) != 4:
            errors.append(
                f"Ending item {ending_index} must contain exactly 4 lines, found {len(ending.lines)}."
            )

    image_targets = _count_image_targets(story)
    if image_targets < 8:
        warnings.append(
            f"Detected only {image_targets} image-bearing targets across scenes, choices, and ending poster."
        )

    if not _has_text(story.meta.title):
        errors.append("Story meta.title is required.")
    if not _has_text(story.meta.genre):
        warnings.append("Story meta.genre is missing.")
    if not _has_text(story.meta.synopsis):
        warnings.append("Story meta.synopsis is missing.")

    if errors:
        return {"valid": False, "error": " | ".join(errors), "warnings": warnings, "image_targets": image_targets}

    return {
        "valid": True,
        "message": "Story JSON validation passed.",
        "warnings": warnings,
        "phase_count": len(story.phases),
        "image_targets": image_targets,
    }


def run_story_reviewer(story_or_text: StoryJSON | str | dict[str, Any]) -> StoryReviewReport:
    """
    Structured review gate aligned to the StoryReviewReport contract.
    """
    try:
        story = _extract_story(story_or_text)
    except Exception as exc:
        return StoryReviewReport(
            passed=False,
            story_quality_checks=[
                StoryReviewCheckItem(
                    code="story.parse_failure",
                    label="Story payload parse",
                    passed=False,
                    details=str(exc),
                )
            ],
            relational_readiness_checks=[],
            handoff_notes=StoryReviewHandoffNotes(
                ambiguity_flags=["Story payload could not be parsed into StoryJson."],
                manual_review_points=["Inspect upstream Story Creator output before Cut Architect."],
            ),
            warnings=["Reviewer received malformed story payload."],
        )

    validation = validate_story_json(story)
    image_targets = int(validation.get("image_targets", 0))
    warnings = list(validation.get("warnings", []))
    speakers = _collect_dialogue_speakers(story)
    location_candidates = _collect_location_candidates(story)
    image_descriptions = _all_image_descriptions(story)

    phase_structure_passed = len(story.phases) == 3 and [phase.phaseNumber for phase in story.phases] == [1, 2, 3]
    choice_coverage_passed = all(len(phase.choice.choices) == 2 for phase in story.phases)
    image_density_passed = image_targets >= 8
    anchorable_visual_passed = all(
        len(description.split()) >= 5 for description in image_descriptions
    ) and bool(image_descriptions)

    story_quality_checks = [
        StoryReviewCheckItem(
            code="story.phase_count",
            label="Three-phase structure",
            passed=phase_structure_passed,
            details=f"Detected {len(story.phases)} phases.",
        ),
        StoryReviewCheckItem(
            code="story.choice_coverage",
            label="Choice coverage",
            passed=choice_coverage_passed,
            details="Each phase should include exactly two choice options.",
        ),
    ]

    relational_checks = [
        StoryReviewCheckItem(
            code="relational.image_density",
            label="Image scene density",
            passed=image_density_passed,
            details=f"Detected {image_targets} image-bearing targets across phases, choices, and ending poster.",
        ),
        StoryReviewCheckItem(
            code="relational.location_anchorability",
            label="Anchorable visual scenes",
            passed=anchorable_visual_passed,
            details="Image descriptions should be concrete enough for location, character, lighting, and palette extraction.",
        ),
    ]

    ambiguity_flags: list[str] = []
    if not image_density_passed:
        ambiguity_flags.append("Image-bearing targets are too sparse for stable cut extraction.")
    if not anchorable_visual_passed:
        ambiguity_flags.append("Some image descriptions are too short or abstract for reliable relational authoring.")
    if not speakers:
        ambiguity_flags.append("No named dialogue speakers detected; recurring character inference may be weak.")

    manual_review_points: list[str] = []
    if validation.get("valid") is False:
        manual_review_points.append(validation.get("error", "Story validation failed."))

    passed = all(item.passed for item in story_quality_checks + relational_checks) and validation.get("valid", False)
    if not passed and not warnings:
        warnings.append("Reviewer failed structural or relational readiness checks.")

    return StoryReviewReport(
        passed=passed,
        story_quality_checks=story_quality_checks,
        relational_readiness_checks=relational_checks,
        handoff_notes=StoryReviewHandoffNotes(
            location_candidates=location_candidates,
            recurring_character_candidates=speakers,
            ambiguity_flags=ambiguity_flags,
            manual_review_points=manual_review_points,
        ),
        warnings=warnings,
    )


def run_deterministic_compiler(
    shared_assets: SharedAssets,
    relational_cuts: RelationalCutsFile,
    *,
    target_model: str = "flux-klein-v2",
) -> InheritanceCompilePreview:
    """
    Compile relational cuts into deterministic preview prompts with continuity resolution.
    """
    return compile_story_preview(
        shared_assets,
        relational_cuts,
        target_model=target_model,
    )


@dataclass(frozen=True)
class ToolExecution:
    name: str
    source_hint: str
    payload: str
    handled: bool
    message: str


PORTED_TOOLS = (
    PortingModule(
        name="validate_story_json",
        responsibility="Validates generator 3-phase StoryJson structure for Inheritance Engine",
        source_hint="Inheritance Engine",
        status="active",
    ),
    PortingModule(
        name="run_story_reviewer",
        responsibility="Performs quality review of story text",
        source_hint="Inheritance Engine",
        status="active",
    ),
    PortingModule(
        name="run_deterministic_compiler",
        responsibility="Compiles inherited prompts from assets and cuts",
        source_hint="Inheritance Engine",
        status="active",
    ),
)


def build_tool_backlog() -> PortingBacklog:
    return PortingBacklog(title="Webtoon Tool surface", modules=list(PORTED_TOOLS))


def tool_names() -> list[str]:
    return [module.name for module in PORTED_TOOLS]


def get_tool(name: str) -> PortingModule | None:
    needle = name.lower()
    for module in PORTED_TOOLS:
        if module.name.lower() == needle:
            return module
    return None


def get_tools(
    simple_mode: bool = False,
    include_mcp: bool = True,
    permission_context: ToolPermissionContext | None = None,
) -> tuple[PortingModule, ...]:
    return PORTED_TOOLS


def execute_tool(name: str, payload: str = "") -> ToolExecution:
    module = get_tool(name)
    if module is None:
        return ToolExecution(name=name, source_hint="", payload=payload, handled=False, message=f"Unknown tool: {name}")

    return ToolExecution(
        name=module.name,
        source_hint=module.source_hint,
        payload=payload,
        handled=True,
        message=f"Executed {module.name} with payload.",
    )


def render_tool_index(limit: int = 20, query: str | None = None) -> str:
    lines = [f"Webtoon Tool entries: {len(PORTED_TOOLS)}", ""]
    lines.extend(f"- {module.name} — {module.responsibility}" for module in PORTED_TOOLS)
    return "\n".join(lines)
