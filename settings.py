# settings.py

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import os

MODE_TRAIN = "train"
MODE_ONLINE = "online"

@dataclass
class Connection:
    ip: str = "127.0.0.1"
    unity_port: int = 50000
    matlab_port: int = 9999
    unity_exe: str = "./unity/EndlessRunnerMI.exe"
    matlab_script: str = "./matlab/run_exo_tcp.m"
    launch_unity: bool = True
    launch_matlab: bool = False  # <- opcional: no se usa en tu main actual

    def resolve_unity(self) -> str:
        return self._resolve(self.unity_exe)

    def resolve_matlab(self) -> str:
        return self._resolve(self.matlab_script)

    def _resolve(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.abspath(os.path.join(os.path.dirname(__file__), p))

@dataclass
class Thresholds:
    mi_start_hi: float = 0.68
    mi_start_lo: float = 0.56
    att_start_lo: float = 0.38
    mi_stop: float = 0.38
    index_window_s: float = 5.0
    cooldown_s: float = 5.0
    consecutive_start_needed: int = 1
    decision_rate_hz: float = 5.0
    require_att_on_high: bool = False
    send_binary_to_unity: bool = True
    send_binary_to_matlab: bool = True

@dataclass
class Timings:
    # 3 s prep (yellow) -> 2 s cue -> 5 s feedback (EEG) -> 5 s decision window -> 5 s rest
    updateRate_ms: int = 200
    window_ms: int = 200
    o_trial_cue: int = 3000
    o_trial_start_feedback: int = 5000
    o_trial_final_feedback: int = 10000
    o_trial_end: int = 10000
    decision_window_ms: int = 5000
    t_trial_rest: int = 5000
    t_prerun_ms: int = 1000
    t_postrun_ms: int = 1000
    n_trials: int = 10

@dataclass
class Run:
    mode: str = MODE_ONLINE
    # Deben coincidir con el entrenamiento (orden y nombres)
    target_channels: List[str] = field(default_factory=lambda: ["F3", "C3", "P3", "Fz", "Cz", "F4", "C4", "P4"])
    # Training e inferencia a 5 s
    epoch_len_s: float = 5.0
    epoch_step_s: float = 0.2
    # Modelos
    mi_model_path: Optional[str] = "./model_riem_tangent_lda_full.joblib"
    att_model_path: Optional[str] = "./model_att_frontal_lda_full.joblib"
    att_scaler_path: Optional[str] = "./scaler_att_frontal_full.joblib"
    use_att_model: bool = True
    # Preprocesado
    apply_car: bool = True
    apply_laplacian: bool = False

    def resolve(self, p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        return p if os.path.isabs(p) else os.path.abspath(os.path.join(os.path.dirname(__file__), p))

    def validate(self) -> None:
        if self.mode == MODE_ONLINE and not self.mi_model_path:
            raise ValueError("ONLINE mode requires mi_model_path.")
        if self.mi_model_path and not os.path.isfile(self.resolve(self.mi_model_path)):
            raise FileNotFoundError(f"MI model not found: {self.resolve(self.mi_model_path)}")
        if self.use_att_model:
            if not self.att_model_path:
                raise ValueError("use_att_model=True requires att_model_path.")
            if not self.att_scaler_path:
                raise ValueError("use_att_model=True requires att_scaler_path.")
            if not os.path.isfile(self.resolve(self.att_model_path)):
                raise FileNotFoundError(f"ATT model not found: {self.resolve(self.att_model_path)}")
            if not os.path.isfile(self.resolve(self.att_scaler_path)):
                raise FileNotFoundError(f"ATT scaler not found: {self.resolve(self.att_scaler_path)}")

@dataclass
class Protocol:
    proto_header_len_bytes: int = 2
    proto_header_byteorder: str = "big"
    json_header_defaults: Dict[str, str] = field(default_factory=lambda: {
        "byteorder": "little",
        "content-type": "text/json",
        "content-encoding": "utf-8",
    })
    events: Dict[str, str] = field(default_factory=lambda: {
        "waiting": "waiting",
        "set_parameters": "setParameters",
        "classification_result": "classification_result",
        "stop": "stop",
        "ready": "ready",
        "play": "play",
        "pause": "pause",
        "resume": "resume",
    })

@dataclass
class Settings:
    game_id: str = "EndlessRunnerMI"
    connection: Connection = field(default_factory=Connection)
    thresholds: Thresholds = field(default_factory=Thresholds)
    timings: Timings = field(default_factory=Timings)
    run: Run = field(default_factory=Run)
    protocol: Protocol = field(default_factory=Protocol)

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "timings": asdict(self.timings),
            "thresholds": asdict(self.thresholds),
            "channels": self.run.target_channels,
            "epoch_len_s": self.run.epoch_len_s,
            "epoch_step_s": self.run.epoch_step_s,
            "policy_decision_rate_hz": self.thresholds.decision_rate_hz,
        }

    def validate(self) -> None:
        self.run.validate()
        assert self.timings.o_trial_start_feedback >= self.timings.o_trial_cue
        assert self.timings.o_trial_final_feedback >= self.timings.o_trial_start_feedback
        assert self.thresholds.mi_stop <= self.thresholds.mi_start_lo <= self.thresholds.mi_start_hi

if __name__ == "__main__":
    s = Settings()
    try:
        s.validate()
        print("Settings OK")
        print(s.to_dict())
        print("Unity:", s.connection.resolve_unity())
        print("MATLAB:", s.connection.resolve_matlab())
    except Exception as e:
        print("Settings error:", e)
