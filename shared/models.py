from pydantic import BaseModel
from typing import Literal

class Stroke(BaseModel):
    rally_id: int
    player: Literal["A", "B"] 
    ball_round: int
    frame_num: int
    shot_type: str
    hit_x: float|None
    hit_y: float|None
    landing_x: float|None
    landing_y: float|None
    player_location_x: float|None
    player_location_y: float|None
    opponent_location_x: float|None
    opponent_location_y: float|None
    getpoint_player: Literal["A", "B"] | None = None   # who won the point after this stroke, if any

class Rally(BaseModel):
    match_id: str
    set_num: int
    rally_id: int
    strokes: list[Stroke]
    rally_winner: Literal["A", "B"] | None = None   # pulled from the last stroke's getpoint_player

class FrameGroup(BaseModel):
    match_id: str
    set_num: int
    rally_id: int
    frames: dict[int, str]   # frame_num -> file path

class Job(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "complete", "failed"] = "queued"
    # "shuttleset": an annotated match (match_id). "video": any footage -- a
    # YouTube link or an uploaded file (video_key) -- analyzed via vision/.
    source: Literal["shuttleset", "video"] = "shuttleset"
    match_id: str | None = None
    youtube_url: str | None = None
    video_key: str | None = None    # S3 key of an uploaded video (uploads/...)
    result_key: str | None = None   # S3 key, only set once complete
    error: str | None = None        # set when status == "failed"
    created_at: str | None = None   # ISO-8601 UTC
    updated_at: str | None = None

class AnalysisResult(BaseModel):
    match_id: str
    set_num: int
    rally_id: int
    depth: Literal["deep", "surface", "skip"]
    tactical_pattern: str | None = None
    suggestion_type: Literal["positioning", "shot_selection", "pattern_exploitation"] | None = None
    suggestion_text: str | None = None

class RallyReport(BaseModel):
    """One rally's analysis plus what the UI needs to sync it to the video
    without loading ShuttleSet itself."""
    analysis: AnalysisResult
    start_time_sec: float
    rally_winner: Literal["A", "B"] | None = None
    n_strokes: int
    score_a: int   # running score after this rally
    score_b: int

class RallyError(BaseModel):
    set_num: int
    rally_id: int
    error: str

class JobResult(BaseModel):
    """The JSON the worker writes to S3 and the UI reads back."""
    job_id: str | None = None   # None for local (no-AWS) runs
    match_id: str
    youtube_url: str | None = None
    fps: float
    rallies: list[RallyReport]
    errors: list[RallyError] = []

SHOT_TYPE_TRANSLATIONS: dict[str, str] = {
    "放小球": "net shot",
    "擋小球": "return net",
    "殺球": "smash",
    "點扣": "wrist smash",
    "挑球": "lob",
    "防守回挑": "defensive return lob",
    "長球": "clear",
    "平球": "drive",
    "小平球": "driven flight",
    "後場抽平球": "back-court drive",
    "切球": "drop",
    "過渡切球": "passive drop",
    "推球": "push",
    "撲球": "rush",
    "防守回抽": "defensive return drive",
    "勾球": "cross-court net shot",
    "發短球": "short service",
    "發長球": "long service",
}
