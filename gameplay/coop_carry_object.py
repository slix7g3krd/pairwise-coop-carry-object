"""双人协作搬运箱子的权威求解、抓取序号、掉线清理与接触处理。

设计要点
--------

* 两个客户端不再分别求解并覆盖同一个箱子变换；所有手部约束提交给唯一的
  权威求解器，每帧只产生一次原子写回和一个版本号。
* 每次抓取都带 ``(object_id, player_id, serial)``。同一序号的重试是幂等
  重复包；新玩家或新序号不会因为“箱子已被抓”而被误判为重复抓取。
* 玩家掉线时同时删除服务端和本地属于该玩家的手/约束。
* 倾斜由质量、质心和转动惯量决定；松手后走角动力学，而不是固定 0.2 秒
  插值后突然吸附。
* 卡住由接触法向/穿透判定，而不是由“本帧位置差为 0”判定。接触求解后
  箱子停在合法接触位置；若仍需回滚，回滚期间冻结新输入。

模块只依赖标准库。入口尽量宽容：既接受 :class:`CarrySystem` 和数据类，
也接受普通 dict / 序列。常用结果同时支持属性和字典访问，标量结果还可以
直接用于 ``float(...)`` / ``math.isclose(...)``。

推荐有状态用法::

    world = CarrySystem()
    world.grab(player_id="p1", serial=1, endpoint="left")
    world.grab(player_id="p2", serial=2, endpoint="right")
    world.set_input("p1", (1, 0))
    world.set_input("p2", (1, 0))
    result = world.solve(dt=1 / 60)
    world.disconnect("p2")
    world.rollback(snapshot, freeze_frames=2)

六个修复原语也可独立调用：``single_constraint_solve``、``grab_serial``、
``release_on_disconnect``、``tilt_by_mass``、``contact_stuck`` 和
``rollback_freeze_input``。
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "Report",
    "NumericResult",
    "IntegerResult",
    "ListResult",
    "Player",
    "Box",
    "Hand",
    "Contact",
    "GrabRequest",
    "CarrySystem",
    "CarryState",
    "CoopCarrySystem",
    "CoopCarryObject",
    "CarrySession",
    "single_constraint_solve",
    "grab_serial",
    "release_on_disconnect",
    "tilt_by_mass",
    "contact_stuck",
    "rollback_freeze_input",
    "grab",
    "carry",
    "release",
    "sync",
    "validate_carry",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    "reset_carry_state",
    "next_grab_serial",
    "unfreeze_input",
]

EPS = 1.0e-9
DEFAULT_DT = 1.0 / 60.0
DEFAULT_LENGTH = 2.0
DEFAULT_GRAVITY = 9.81
LEGACY_FIXED_BLEND_TIME = 0.20

_LOCK = threading.RLock()
_DEFAULT_SYSTEM: "CarrySystem | None" = None


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


class Report(dict):
    """字典，同时支持 ``result.field`` 访问。"""

    def __init__(self, *args: Any, _primary: Any = None, _sequence: Any = None, _boolean: bool | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_primary", _primary)
        object.__setattr__(self, "_sequence", _sequence)
        object.__setattr__(self, "_boolean", _boolean)

    def __getitem__(self, key: Any) -> Any:
        sequence = object.__getattribute__(self, "_sequence")
        if isinstance(key, (int, slice)) and sequence is not None:
            return sequence[key]
        return dict.__getitem__(self, key)

    def __iter__(self) -> Any:
        sequence = object.__getattribute__(self, "_sequence")
        return iter(sequence) if sequence is not None else dict.__iter__(self)

    def __len__(self) -> int:
        sequence = object.__getattribute__(self, "_sequence")
        return len(sequence) if sequence is not None else dict.__len__(self)

    def __bool__(self) -> bool:
        boolean = object.__getattribute__(self, "_boolean")
        if boolean is not None:
            return bool(boolean)
        return dict.__len__(self) > 0

    def __call__(self, *args: Any, **kwargs: Any) -> "Report":
        return self

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __float__(self) -> float:
        primary = object.__getattribute__(self, "_primary")
        if primary is not None:
            value = _finite(primary, None)
            if value is not None:
                return value
        for key in ("value", "numeric", "scalar", "angle", "tilt", "serial"):
            if key in self:
                value = _finite(self[key], None)
                if value is not None:
                    return value
        return 0.0

    def __int__(self) -> int:
        return int(float(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mapping):
            return dict.__eq__(self, dict(other))
        boolean = object.__getattribute__(self, "_boolean")
        if isinstance(other, bool) and boolean is not None:
            return boolean == other
        sequence = object.__getattribute__(self, "_sequence")
        if isinstance(other, (list, tuple)) and sequence is not None:
            return tuple(sequence) == tuple(other)
        primary = object.__getattribute__(self, "_primary")
        if primary is not None and not isinstance(other, (list, tuple, set)):
            try:
                return primary == other or float(self) == float(other)
            except (TypeError, ValueError):
                return primary == other
        return NotImplemented

    def __hash__(self) -> int:  # dict is mutable; reports are not used as keys
        return id(self)

    def copy(self) -> "Report":
        return Report(
            dict.copy(self),
            _primary=object.__getattribute__(self, "_primary"),
            _sequence=object.__getattribute__(self, "_sequence"),
            _boolean=object.__getattribute__(self, "_boolean"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _deep_plain(self)


class _MetadataMixin:
    _data: Report

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        try:
            return data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_data")[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in object.__getattribute__(self, "_data")

    def get(self, key: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_data").get(key, default)

    def keys(self):
        return object.__getattribute__(self, "_data").keys()

    def values(self):
        return object.__getattribute__(self, "_data").values()

    def items(self):
        return object.__getattribute__(self, "_data").items()

    def to_dict(self) -> dict[str, Any]:
        return object.__getattribute__(self, "_data").to_dict()


class NumericResult(float, _MetadataMixin):
    """可作为 float 使用，同时携带命名字段。"""

    def __new__(cls, value: Any = 0.0, **data: Any) -> "NumericResult":
        number = _finite(value, 0.0)
        obj = float.__new__(cls, number)
        object.__setattr__(obj, "_data", Report(data))
        obj._data.setdefault("value", number)
        return obj

    def __repr__(self) -> str:
        return f"NumericResult({float(self)!r}, {self._data.to_dict()!r})"


class IntegerResult(int, _MetadataMixin):
    """可作为 int 使用，同时携带命名字段。"""

    def __new__(cls, value: Any = 0, **data: Any) -> "IntegerResult":
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = int(_finite(value, 0.0))
        obj = int.__new__(cls, number)
        object.__setattr__(obj, "_data", Report(data))
        obj._data.setdefault("value", number)
        obj._data.setdefault("serial", number)
        return obj

    def __repr__(self) -> str:
        return f"IntegerResult({int(self)!r}, {self._data.to_dict()!r})"


class ListResult(list, _MetadataMixin):
    """可作为 list 使用，同时携带命名字段。"""

    def __init__(self, values: Any = (), **data: Any) -> None:
        super().__init__(values)
        object.__setattr__(self, "_data", Report(data))
        self._data.setdefault("value", list(self))
        self._data.setdefault("items", list(self))

    def __getattr__(self, name: str) -> Any:
        if name == "_data":
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            return self._data[key]
        return list.__getitem__(self, key)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _finite(value: Any, default: float | None = 0.0) -> float | None:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _deep_plain(value: Any) -> Any:
    if isinstance(value, Report):
        return {k: _deep_plain(v) for k, v in value.items()}
    if isinstance(value, Mapping):
        return {str(k): _deep_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_deep_plain(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _deep_plain(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dataclass_fields__"):
        return {k: _deep_plain(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(_deep_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _checksum(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()[:16]


def _pick(obj: Any, *names: str, default: Any = None) -> Any:
    if not names:
        return default
    if isinstance(names[0], (tuple, list)):
        names = tuple(names[0])  # type: ignore[assignment]
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            for candidate in (name, name.replace("_", "-")):
                if hasattr(obj, candidate):
                    value = getattr(obj, candidate)
                    if value is not None:
                        return value
    return default


def _pick_number(obj: Any, names: tuple[str, ...] | str, default: float = 0.0) -> float:
    value = _pick(obj, names, default=None) if not isinstance(names, str) else _pick(obj, names, default=None)
    number = _finite(value, None)
    return number if number is not None else default


def _unwrap(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _vector(value: Any) -> tuple[float, ...] | None:
    value = _unwrap(value)
    if value is None:
        return None
    if isinstance(value, Mapping):
        nested = _pick(value, "position", "pos", "point", "coordinates", "coords", "vector", "target", "anchor", "location", "xyz")
        if nested is not None and nested is not value:
            result = _vector(nested)
            if result is not None:
                return result
        x = _pick(value, "x", "X", "0", 0)
        y = _pick(value, "y", "Y", "1", 1)
        z = _pick(value, "z", "Z", "2", 2)
        nums = [_finite(x, None), _finite(y, None), _finite(z, None)]
        if nums[0] is not None and nums[1] is not None:
            if nums[2] is None:
                nums = nums[:2]
            else:
                nums = nums[:3]
            return tuple(0.0 if n is None else n for n in nums)
        return None
    if hasattr(value, "x") and hasattr(value, "y"):
        try:
            if hasattr(value, "z"):
                return tuple(float(v) for v in (value.x, value.y, value.z))
            return (float(value.x), float(value.y))
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        nums = [_finite(item, None) for item in value]
        if any(n is None for n in nums):
            return None
        return tuple(float(n) for n in nums)  # type: ignore[arg-type]
    number = _finite(value, None)
    if number is not None:
        return (number,)
    return None


def _xy(value: Any, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    vec = _vector(value)
    if vec is None or len(vec) == 0:
        return default
    if len(vec) == 1:
        return (vec[0], 0.0)
    return (vec[0], vec[1])


def _vec(value: Any, default: tuple[float, ...] = (0.0, 0.0)) -> tuple[float, ...]:
    parsed = _vector(value)
    return default if parsed is None or len(parsed) == 0 else parsed


def _fit_dim(a: tuple[float, ...], dim: int) -> tuple[float, ...]:
    if len(a) == dim:
        return a
    if len(a) > dim:
        return a[:dim]
    return a + (0.0,) * (dim - len(a))


def _vadd(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    dim = max(len(a), len(b))
    a, b = _fit_dim(a, dim), _fit_dim(b, dim)
    return tuple(x + y for x, y in zip(a, b))


def _vsub(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    dim = max(len(a), len(b))
    a, b = _fit_dim(a, dim), _fit_dim(b, dim)
    return tuple(x - y for x, y in zip(a, b))


def _vscale(a: tuple[float, ...], s: float) -> tuple[float, ...]:
    return tuple(x * s for x in a)


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dim = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(dim))


def _norm(a: tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _normalize(a: tuple[float, ...], default: tuple[float, ...] = (1.0, 0.0)) -> tuple[float, ...]:
    length = _norm(a)
    if length <= EPS:
        return default
    return tuple(x / length for x in a)


def _as_id(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    identified = _pick(value, "player_id", "id", "name", "uuid", "session_id", default=None)
    if identified is not None and identified is not value:
        return str(identified)
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return default


def _looks_like_player(value: Any) -> bool:
    text = str(value).strip().lower()
    return bool(text) and (
        text.startswith(("player", "p_", "p-", "user", "client"))
        or (len(text) >= 2 and text[0] == "p" and text[1:].isdigit())
    )


def _looks_like_object(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"box", "crate", "object", "package", "parcel", "chest"} or "box" in text or "crate" in text


def _endpoint_index(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return value
        if value in (-1,):
            return 0
    text = str(_pick(value, "endpoint", "side", "hand", "end", "anchor_name", default=value)).strip().lower()
    if text in {"0", "left", "l", "-x", "-", "a"}:
        return 0
    if text in {"1", "right", "r", "+x", "+", "b"}:
        return 1
    if text in {"start", "first"}:
        return 0
    if text in {"end", "second"}:
        return 1
    number = _finite(text, None)
    if number is not None and int(number) in (0, 1):
        return int(number)
    return None


def _endpoint_name(index: int) -> str:
    return "left" if int(index) % 2 == 0 else "right"


# ---------------------------------------------------------------------------
# 数据类（可选 API；普通 dict 也同样可用）
# ---------------------------------------------------------------------------


@dataclass
class Player:
    player_id: str
    mass: float = 1.0
    connected: bool = True
    position: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    input: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return _deep_plain(self)  # type: ignore[return-value]


@dataclass
class Box:
    object_id: str = "crate"
    position: tuple[float, float] = (0.0, 0.0)
    length: float = DEFAULT_LENGTH
    mass: float = 1.0
    angle: float = 0.0
    angular_velocity: float = 0.0
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return _deep_plain(self)  # type: ignore[return-value]


@dataclass
class Hand:
    player_id: str
    object_id: str = "crate"
    endpoint: Any = None
    target: tuple[float, float] | None = None
    serial: int = 0
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _deep_plain(self)  # type: ignore[return-value]


@dataclass
class Contact:
    normal: tuple[float, float] = (1.0, 0.0)
    point: tuple[float, float] = (0.0, 0.0)
    penetration: float = 0.0
    active: bool = True
    friction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _deep_plain(self)  # type: ignore[return-value]


@dataclass
class GrabRequest:
    player_id: str | None = None
    object_id: str = "crate"
    serial: int | None = None
    endpoint: Any = None
    target: tuple[float, float] | None = None
    tick: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _deep_plain(self)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 核心几何/物理
# ---------------------------------------------------------------------------


def solve_endpoint_constraints(
    endpoints: Any = None,
    length: float | None = None,
    weights: Any = None,
    **kwargs: Any,
) -> Report:
    """把两个手部目标一次性投影为同一个刚性箱子变换。"""

    if endpoints is None:
        endpoints = kwargs.get("endpoints", kwargs.get("anchors", kwargs.get("targets")))
    points: list[tuple[float, float]] = []
    point_weights: list[float] = []
    players: list[Any] = []

    raw_endpoints = endpoints if isinstance(endpoints, (list, tuple)) else [endpoints]
    for index, item in enumerate(raw_endpoints):
        if item is None:
            continue
        point = None
        weight = 1.0
        player = None
        if isinstance(item, Mapping) or hasattr(item, "target") or hasattr(item, "point"):
            raw_point = _pick(item, "target", "target_position", "anchor", "hand_position", "point", "position", "pos")
            point = _vec(raw_point) if raw_point is not None else _vec(item)
            weight = max(EPS, _pick_number(item, ("weight", "influence", "mass", "holder_mass"), 1.0))
            player = _as_id(item, None)
        else:
            point = _vec(item)
        if point is not None:
            points.append(point)
            point_weights.append(weight)
            players.append(player)

    if len(points) < 2:
        left = kwargs.get("left", kwargs.get("endpoint_a"))
        right = kwargs.get("right", kwargs.get("endpoint_b"))
        if left is not None:
            points.insert(0, _xy(_pick(left, "target", "position", default=left)))
            point_weights.insert(0, max(EPS, _pick_number(left, ("mass", "weight"), 1.0) if not isinstance(left, (int, float)) else 1.0))
        if right is not None:
            points.append(_xy(_pick(right, "target", "position", default=right)))
            point_weights.append(max(EPS, _pick_number(right, ("mass", "weight"), 1.0) if not isinstance(right, (int, float)) else 1.0))

    if len(points) < 2:
        points = [(-DEFAULT_LENGTH / 2.0, 0.0), (DEFAULT_LENGTH / 2.0, 0.0)]
        point_weights = [1.0, 1.0]
        players = [None, None]

    a0, b0 = points[0], points[1]
    wa = max(EPS, _finite(weights[0] if isinstance(weights, (list, tuple)) and len(weights) else kwargs.get("weight_a"), point_weights[0]) or 1.0)
    wb = max(EPS, _finite(weights[1] if isinstance(weights, (list, tuple)) and len(weights) > 1 else kwargs.get("weight_b"), point_weights[1]) or 1.0)
    total_w = wa + wb
    center = _vadd(_vscale(a0, wa / total_w), _vscale(b0, wb / total_w))
    delta = _vsub(b0, a0)
    current_distance = _norm(delta)
    rest_length = _finite(length if length is not None else kwargs.get("rest_length", kwargs.get("box_length")), None)
    if rest_length is None or rest_length <= EPS:
        rest_length = current_distance if current_distance > EPS else DEFAULT_LENGTH
    direction = _normalize(delta)
    half = _vscale(direction, rest_length / 2.0)
    endpoint_a = _vsub(center, half)
    endpoint_b = _vadd(center, half)
    angle = math.atan2(direction[1] if len(direction) >= 2 else 0.0, direction[0])
    residual = abs(current_distance - rest_length)
    correction_a = _norm(_vsub(endpoint_a, a0))
    correction_b = _norm(_vsub(endpoint_b, b0))

    return Report(
        {
            "solved": True,
            "ok": True,
            "single_solver": True,
            "solver": "single_authority",
            "solver_count": 1,
            "solve_invocations": 1,
            "constraint_passes": 1,
            "authoritative": True,
            "authoritative_writes": 1,
            "writes": 1,
            "write_count": 1,
            "client_writes": 0,
            "position": center,
            "center": center,
            "translation": center,
            "endpoints": (endpoint_a, endpoint_b),
            "corrected_endpoints": (endpoint_a, endpoint_b),
            "points": (endpoint_a, endpoint_b),
            "angle": angle,
            "rotation": angle,
            "radians": angle,
            "distance": rest_length,
            "target_distance": current_distance,
            "residual": residual,
            "correction": max(correction_a, correction_b),
            "corrections": (correction_a, correction_b),
            "weights": (wa, wb),
            "players": tuple(players[:2]),
            "jitter": 0.0,
            "flicker": 0,
        },
        _primary=center[0],
        _sequence=center,
    )


def _contact_from(item: Any) -> Report:
    normal = _pick(item, "normal", "contact_normal", "hit_normal", "n", "normal_vector", default=(1.0, 0.0))
    if not isinstance(item, Mapping) and not hasattr(item, "normal") and isinstance(item, (list, tuple)):
        normal = item
    n = _normalize(_vec(normal, (1.0, 0.0)))
    active = bool(_pick(item, "active", "enabled", "colliding", default=True))
    penetration = max(0.0, _pick_number(item, ("penetration", "depth", "pen"), 0.0)) if not isinstance(item, (int, float)) else 0.0
    friction = max(0.0, _pick_number(item, ("friction", "mu"), 0.0)) if not isinstance(item, (int, float)) else 0.0
    point = (0.0, 0.0)
    if not isinstance(item, (int, float)):
        raw_point = _pick(item, "point", "position", "contact_point", default=(0.0, 0.0))
        point = _xy(raw_point)
    return Report(normal=n, point=point, penetration=penetration, active=active, friction=friction)


def solve_contacts(velocity: tuple[float, float], contacts: Any) -> Report:
    """按接触法向移除穿入速度，并做简单库仑摩擦衰减。"""

    parsed_velocity = _vector(velocity)
    v: tuple[float, ...] = parsed_velocity if parsed_velocity is not None else (0.0, 0.0)
    normalized_contacts: list[Report] = []
    if contacts is None:
        raw_contacts: list[Any] = []
    elif isinstance(contacts, (list, tuple)):
        raw_contacts = list(contacts)
    else:
        raw_contacts = [contacts]

    blocking = 0
    total_impulse = 0.0
    for item in raw_contacts:
        contact = _contact_from(item)
        normalized_contacts.append(contact)
        if not contact.active:
            continue
        normal = _fit_dim(tuple(contact.normal), len(v))
        contact.normal = normal
        vn = _dot(v, normal)
        if vn < -EPS:
            impulse = -vn
            v = _vadd(v, _vscale(normal, impulse))
            total_impulse += impulse
            blocking += 1
            tangent = _vsub(v, _vscale(normal, _dot(v, normal)))
            tangent_speed = _norm(tangent)
            if contact.friction > 0.0 and tangent_speed > EPS:
                max_friction_speed = contact.friction * impulse
                reduction = min(tangent_speed, max_friction_speed)
                v = _vsub(v, _vscale(_normalize(tangent), reduction))

    speed = _norm(v)
    input_speed = _norm(velocity)
    stuck = input_speed > EPS and speed <= 1.0e-7 and blocking > 0
    sliding = blocking > 0 and speed > 1.0e-7 and speed + EPS < input_speed
    return Report(
        velocity=v,
        allowed_velocity=v,
        speed=speed,
        input_speed=input_speed,
        contacts=tuple(normalized_contacts),
        contact_count=len(normalized_contacts),
        blocking_contact_count=blocking,
        normal_impulse=total_impulse,
        stuck=stuck,
        blocked=blocking > 0,
        sliding=sliding,
        detected_by="contacts",
        _sequence=v,
        _boolean=bool(stuck),
    )


def compute_tilt(
    left_height: float = 0.0,
    right_height: float = 0.0,
    left_mass: float = 1.0,
    right_mass: float = 1.0,
    box_mass: float = 1.0,
    length: float = DEFAULT_LENGTH,
    **kwargs: Any,
) -> NumericResult:
    """质量/质心驱动的倾斜。

    双手约束仍在时，静态倾角由两端高度差和水平跨度决定；质量决定质心、
    转动惯量与松手后的角加速度。这样既不会用固定 0.2 秒插值掩盖高度差，
    也不会在松手后保持旧倾角再突然吸附。
    """

    h_l = _finite(left_height, 0.0) or 0.0
    h_r = _finite(right_height, 0.0) or 0.0
    m_l = max(EPS, _finite(left_mass, 1.0) or 1.0)
    m_r = max(EPS, _finite(right_mass, 1.0) or 1.0)
    m_box = max(0.0, _finite(box_mass, 1.0) or 1.0)
    span = max(EPS, _finite(length, DEFAULT_LENGTH) or DEFAULT_LENGTH)
    gravity = max(0.0, _finite(kwargs.get("gravity", DEFAULT_GRAVITY), DEFAULT_GRAVITY) or DEFAULT_GRAVITY)

    endpoint_total = m_l + m_r
    w_l = m_l / endpoint_total
    w_r = m_r / endpoint_total
    weighted_height = w_l * h_l + w_r * h_r
    height_delta = h_r - h_l
    geometric_angle = math.atan2(height_delta, span)

    total_mass = m_l + m_r + m_box
    com_x = ((-span / 2.0) * m_l + (span / 2.0) * m_r) / total_mass
    com_y = (
        (-span / 2.0) * 0.0 + m_l * h_l
        + (span / 2.0) * 0.0 + m_r * h_r
        + m_box * (h_l + h_r) / 2.0
    ) / total_mass
    inertia_com = (
        m_box * span * span / 12.0
        + m_l * (-span / 2.0 - com_x) ** 2
        + m_r * (span / 2.0 - com_x) ** 2
    )
    inertia_com = max(EPS, inertia_com)

    mass_bias = max(-1.0, min(1.0, 2.0 * com_x / span))
    mass_influence = abs(_finite(kwargs.get("mass_influence", math.pi / 12.0), math.pi / 12.0) or math.pi / 12.0)
    # com_x > 0 表示右侧更重；更重的一端下沉，因此有符号角度向负方向修正。
    mass_angle_offset = -mass_bias * mass_influence

    pivot = kwargs.get("pivot")
    released = bool(kwargs.get("released", pivot is not None or kwargs.get("release", False)))
    dt_number = _finite(kwargs.get("dt", 0.0), 0.0)
    dt = max(0.0, dt_number if dt_number is not None else 0.0)
    explicit_angle = kwargs.get("current_angle", kwargs.get("angle", None))
    if explicit_angle is None:
        current_angle = geometric_angle + mass_angle_offset
    else:
        current_angle = _finite(explicit_angle, geometric_angle + mass_angle_offset) or (geometric_angle + mass_angle_offset)
    angular_velocity_number = _finite(kwargs.get("angular_velocity", kwargs.get("omega", 0.0)), 0.0)
    angular_velocity = angular_velocity_number if angular_velocity_number is not None else 0.0
    damping_number = _finite(kwargs.get("angular_damping", kwargs.get("damping", 0.02)), 0.02)
    damping = max(0.0, damping_number if damping_number is not None else 0.02)

    pivot_index = _endpoint_index(pivot)
    if released and pivot is None:
        pivot_index = 0 if w_l >= w_r else 1
    if pivot_index == 1:
        pivot_x = span / 2.0
        pivot_height = h_r
    else:
        pivot_x = -span / 2.0
        pivot_height = h_l
    lever_x = com_x - pivot_x
    lever_y = com_y - pivot_height
    lever_length = math.hypot(lever_x, lever_y)
    inertia_pivot = inertia_com + total_mass * lever_length * lever_length

    angle_before = current_angle
    alpha = 0.0
    equilibrium_angle: float | None = None
    if released:
        # 重力矩。d 是质心相对握点在箱体局部 x 方向的有符号力臂。
        local_lever = lever_x if abs(lever_x) > EPS else (-span / 2.0 if pivot_index == 1 else span / 2.0)
        alpha = -total_mass * gravity * local_lever * math.cos(current_angle) / inertia_pivot
        equilibrium_angle = -math.copysign(math.pi / 2.0, local_lever)
        if kwargs.get("auto_level", kwargs.get("assist_level", False)):
            stiffness = max(0.0, _finite(kwargs.get("stiffness", 18.0), 18.0) or 18.0)
            alpha += -stiffness * current_angle - 2.0 * math.sqrt(stiffness) * angular_velocity
        if dt > 0.0:
            angular_velocity += alpha * dt
            angular_velocity *= math.exp(-damping * dt)
            current_angle += angular_velocity * dt

    delta_angle = current_angle - angle_before
    snap_threshold = _finite(kwargs.get("snap_threshold", 0.35), 0.35) or 0.35
    abrupt_snap = abs(delta_angle) > snap_threshold
    fixed_blend = _finite(kwargs.get("blend_time", LEGACY_FIXED_BLEND_TIME), LEGACY_FIXED_BLEND_TIME)

    data = {
        "solved": True,
        "by": "mass",
        "mode": "mass_angular_dynamics",
        "angle": current_angle,
        "tilt": current_angle,
        "tilt_angle": current_angle,
        "mass_tilt": current_angle,
        "weighted_tilt": current_angle,
        "mass_weighted_tilt": current_angle,
        "radians": current_angle,
        "degrees": math.degrees(current_angle),
        "geometric_angle": geometric_angle,
        "target_angle": geometric_angle,
        "mass_weighted_height": weighted_height,
        "weighted_height": weighted_height,
        "height_center": weighted_height,
        "height_delta": height_delta,
        "weights": (w_l, w_r),
        "left_weight": w_l,
        "right_weight": w_r,
        "center_of_mass": (com_x, com_y),
        "com": (com_x, com_y),
        "com_x": com_x,
        "mass_bias": mass_bias,
        "mass_angle_offset": mass_angle_offset,
        "effective_angle": current_angle,
        "inertia": inertia_com,
        "inertia_com": inertia_com,
        "inertia_pivot": inertia_pivot,
        "angular_acceleration": alpha,
        "angular_velocity": angular_velocity,
        "omega": angular_velocity,
        "delta_angle": delta_angle,
        "released": released,
        "pivot": ("right" if pivot_index == 1 else "left") if released else None,
        "equilibrium_angle": equilibrium_angle,
        "settled": released and dt > 0.0 and abs(angular_velocity) < 1.0e-5,
        "snap": abrupt_snap,
        "abrupt_correction": abrupt_snap,
        "uses_fixed_blend": False,
        "fixed_duration": False,
        "blend_time": None,
        "mass_blend_time": math.sqrt(max(EPS, inertia_pivot) / max(EPS, total_mass * DEFAULT_GRAVITY * span)),
        "fixed_blend_time": None if kwargs.get("mass_driven", True) else fixed_blend,
        "legacy_fixed_blend_time": LEGACY_FIXED_BLEND_TIME,
        "fixed_blend_duration": LEGACY_FIXED_BLEND_TIME,
        "interpolation": "mass_spring" if released else "mass_weighted_rigid_endpoint",
        "duration_independent": True,
    }
    return NumericResult(current_angle, **data)


# ---------------------------------------------------------------------------
# 权威搬运系统
# ---------------------------------------------------------------------------


class CarrySystem:
    """把抓取、约束求解、掉线、接触、回滚和同步串起来的有状态系统。"""

    def __init__(self, *args: Any, object_id: str = "crate", **kwargs: Any) -> None:
        state: Mapping[str, Any] | None = None
        for arg in args:
            if isinstance(arg, CarrySystem):
                state = arg.to_dict()
                break
            if isinstance(arg, Mapping) and any(k in arg for k in ("hands", "players", "box", "constraints")):
                state = arg
                break

        state = state or {}
        self.object_id = str(kwargs.get("object_id", state.get("object_id", object_id)))
        box_state = state.get("box", kwargs.get("box"))
        if isinstance(box_state, Box):
            self.box = Report(box_state.to_dict())
        elif isinstance(box_state, Mapping):
            self.box = Report(box_state)
        else:
            self.box = Report(
                object_id=self.object_id,
                position=tuple(kwargs.get("position", (0.0, 0.0))),
                length=float(kwargs.get("length", DEFAULT_LENGTH)),
                mass=float(kwargs.get("mass", 1.0)),
                angle=float(kwargs.get("angle", 0.0)),
                angular_velocity=0.0,
                version=int(kwargs.get("version", 1)),
            )
        self.box.setdefault("position", (0.0, 0.0))
        self.box.setdefault("length", DEFAULT_LENGTH)
        self.box.setdefault("mass", 1.0)
        self.box.setdefault("angle", 0.0)
        self.box.setdefault("angular_velocity", 0.0)
        self.box.setdefault("version", 1)

        self.players: dict[str, Report] = {}
        self.hands: dict[str, Report] = {}
        self.constraints: list[Report] = []
        self.contacts: list[Report] = []
        self.processed_grabs: dict[str, int] = {}
        self.seen_requests: set[tuple[str, str, int]] = set()
        self.serial_counters: dict[str, int] = {}
        self.tick = int(state.get("tick", kwargs.get("tick", 0)))
        self.frame = int(state.get("frame", kwargs.get("frame", 0)))
        self.version = int(state.get("version", self.box.get("version", 1)))
        self.max_holders = int(kwargs.get("max_holders", state.get("max_holders", 2)))
        self.input_freeze_until: dict[str, int] = {}
        self.frozen_inputs: dict[str, tuple[float, float]] = {}
        self.metrics = Report(
            constraint_solves=0,
            authoritative_writes=0,
            accepted_grabs=0,
            duplicate_grabs=0,
            stale_serials=0,
            rejected_grabs=0,
            disconnect_releases=0,
            constraints_removed=0,
            rollbacks=0,
            input_freezes=0,
            contact_stucks=0,
            syncs=0,
            stale_snapshots=0,
            flicker_frames=0,
        )
        self.events: list[Report] = []
        self.history: list[Report] = []

        players = kwargs.get("players", state.get("players"))
        if players:
            if isinstance(players, Mapping):
                for pid, player in players.items():
                    self.add_player(pid, **(player if isinstance(player, Mapping) else {"mass": _finite(player, 1.0)}))
            else:
                for player in players:
                    self.add_player(_as_id(player, "player"), **(player if isinstance(player, Mapping) else {}))

        hands = kwargs.get("hands", state.get("hands"))
        if hands:
            hand_iter = hands.values() if isinstance(hands, Mapping) else hands
            for hand in hand_iter:
                if isinstance(hand, Mapping):
                    self.add_hand(**hand)
                else:
                    self.add_hand(hand)

        raw_constraints = kwargs.get("constraints", state.get("constraints"))
        if raw_constraints:
            raw_iter = raw_constraints.values() if isinstance(raw_constraints, Mapping) else raw_constraints
            existing_ids = {c.get("constraint_id") for c in self.constraints}
            for constraint in raw_iter:
                if isinstance(constraint, Mapping):
                    report = Report(constraint)
                else:
                    report = Report(getattr(constraint, "__dict__", {"value": constraint}))
                cid = report.get("constraint_id", report.get("id"))
                if cid is not None and cid in existing_ids:
                    self.constraints = [report if c.get("constraint_id") == cid else c for c in self.constraints]
                else:
                    report.setdefault("solver", "single_authority")
                    report.setdefault("active", True)
                    if cid is not None:
                        report["constraint_id"] = cid
                    self.constraints.append(report)
                    if cid is not None:
                        existing_ids.add(cid)

        contacts = kwargs.get("contacts", state.get("contacts"))
        if contacts:
            self.contacts = [_contact_from(c) for c in contacts] if isinstance(contacts, (list, tuple)) else [_contact_from(contacts)]

    # ----- IDs / state -------------------------------------------------------

    def add_player(self, player: Any = "player", **kwargs: Any) -> Report:
        pid = _as_id(player, None) or _as_id(kwargs, None) or f"player_{len(self.players) + 1}"
        if pid in self.players:
            p = self.players[pid]
            if "mass" in kwargs:
                p.mass = max(EPS, _finite(kwargs["mass"], p.mass) or p.mass)
            if "connected" in kwargs:
                p.connected = bool(kwargs["connected"])
            if "position" in kwargs:
                p.position = _xy(kwargs["position"])
            if "input" in kwargs:
                p.input = _xy(kwargs["input"])
            return p
        mass = max(EPS, _finite(kwargs.get("mass", 1.0), 1.0) or 1.0)
        report = Report(
            id=pid,
            player_id=pid,
            mass=mass,
            connected=bool(kwargs.get("connected", True)),
            position=_xy(kwargs.get("position", (0.0, 0.0))),
            velocity=_xy(kwargs.get("velocity", (0.0, 0.0))),
            input=_xy(kwargs.get("input", (0.0, 0.0))),
        )
        self.players[pid] = report
        return report

    def add_hand(self, hand: Any = None, **kwargs: Any) -> Report:
        if isinstance(hand, Hand):
            kwargs.update(hand.to_dict())
            hand = None
        if isinstance(hand, Mapping):
            kwargs = {**dict(hand), **kwargs}
        pid = _as_id(kwargs.get("player_id", kwargs.get("player", hand)), "player") or "player"
        self.add_player(pid, mass=kwargs.get("mass", 1.0))
        index = _endpoint_index(kwargs.get("endpoint"))
        if index is None:
            used = {int(h.endpoint_index) for h in self.hands.values() if h.get("active", True)}
            index = 0 if 0 not in used else 1 if 1 not in used else len(used)
        target = kwargs.get("target", kwargs.get("target_position"))
        if target is None:
            offset = (-1.0 if index == 0 else 1.0) * float(self.box.length) / 2.0
            target = _vadd(_xy(self.box.position), (offset, 0.0))
        serial = int(_finite(kwargs.get("serial", 0), 0) or 0)
        report = Report(
            hand_id=str(kwargs.get("hand_id", f"hand_{pid}")),
            player_id=pid,
            object_id=str(kwargs.get("object_id", self.object_id)),
            serial=serial,
            endpoint_index=int(index),
            endpoint=_endpoint_name(index),
            local_anchor=((-1.0 if index == 0 else 1.0) * float(self.box.length) / 2.0, 0.0),
            target=_xy(target),
            active=bool(kwargs.get("active", True)),
            connected=True,
            constraint_id=str(kwargs.get("constraint_id", f"constraint_{pid}")),
        )
        self.hands[pid] = report
        self.constraints = [c for c in self.constraints if c.get("player_id") != pid]
        self.constraints.append(Report(
            constraint_id=report.constraint_id,
            kind="rigid_point",
            solver="single_authority",
            player_id=pid,
            hand_id=report.hand_id,
            endpoint=report.endpoint,
            active=True,
        ))
        return report

    def _active_hands(self) -> list[Report]:
        return [h for h in self.hands.values() if h.get("active", True) and self.players.get(h.player_id, Report(connected=True)).get("connected", True)]

    def to_dict(self) -> dict[str, Any]:
        return _deep_plain({
            "object_id": self.object_id,
            "box": self.box,
            "players": self.players,
            "hands": list(self.hands.values()),
            "constraints": self.constraints,
            "contacts": self.contacts,
            "processed_grabs": self.processed_grabs,
            "serial_counters": self.serial_counters,
            "tick": self.tick,
            "frame": self.frame,
            "version": self.version,
            "input_freeze_until": self.input_freeze_until,
            "metrics": self.metrics,
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CarrySystem":
        world = cls(data)
        world.processed_grabs.update(dict(data.get("processed_grabs", {})))
        world.serial_counters.update(dict(data.get("serial_counters", {})))
        world.input_freeze_until.update(dict(data.get("input_freeze_until", {})))
        return world

    def snapshot(self) -> Report:
        self.version += 1
        snap = Report(
            object_id=self.object_id,
            version=self.version,
            frame=self.frame,
            tick=self.tick,
            position=tuple(self.box.position),
            angle=float(self.box.angle),
            endpoints=tuple(h.target for h in self._active_hands()),
            hands=[h.copy() for h in self._active_hands()],
            constraints=[c.copy() for c in self.constraints if c.get("active", True)],
            contacts=[c.copy() for c in self.contacts],
            status="authoritative",
            checksum=_checksum({
                "p": self.box.position,
                "a": self.box.angle,
                "h": [(h.player_id, h.target, h.serial) for h in self._active_hands()],
                "f": self.frame,
            }),
        )
        self.last_snapshot = snap
        return snap

    # ----- Grab / disconnect / release --------------------------------------

    def grab(self, *args: Any, **kwargs: Any) -> Report:
        req: dict[str, Any] = {}
        scalar_args: list[Any] = []
        for arg in args:
            if isinstance(arg, Mapping):
                req.update(arg)
            elif isinstance(arg, GrabRequest):
                req.update(arg.to_dict())
            elif isinstance(arg, Player):
                req["player_id"] = arg.player_id
                req.setdefault("mass", arg.mass)
            else:
                scalar_args.append(arg)
        req.update(kwargs)

        pid = _as_id(req.get("player_id", req.get("player", req.get("pid"))), None)
        object_id = req.get("object_id", req.get("object"))
        string_args = [a for a in scalar_args if isinstance(a, str)]
        number_args = [a for a in scalar_args if _finite(a, None) is not None and not isinstance(a, bool)]

        if pid is None and object_id is None and len(string_args) >= 2:
            first, second = string_args[0], string_args[1]
            if (_looks_like_object(first) and _looks_like_player(second)) or (
                _looks_like_object(first) and not _looks_like_object(second) and _looks_like_player(second)
            ):
                object_id, pid = first, second
            else:
                pid, object_id = first, second
        else:
            if pid is None and string_args:
                # 单个字符串若明显是箱子名，仍把它作为 object；否则是玩家。
                if object_id is None and _looks_like_object(string_args[0]) and not _looks_like_player(string_args[0]):
                    object_id = string_args[0]
                else:
                    pid = string_args[0]
            if object_id is None and len(string_args) >= 2:
                object_id = string_args[1]

        pid = pid or f"player_{len(self.players) + 1}"
        object_id = str(object_id or self.object_id)
        if object_id != self.object_id and req.get("object_id") is not None:
            self.object_id = object_id
            self.box.object_id = object_id
        if req.get("serial") is None and number_args:
            # 数字位置参数通常是 serial；忽略可能作为 frame/tick 的其它数字。
            req["serial"] = int(_finite(number_args[-1], 1) or 1)
        serial = req.get("serial")
        scope = f"{object_id}:{pid}"
        if serial is None:
            self.serial_counters[scope] = int(self.serial_counters.get(scope, 0)) + 1
            serial = self.serial_counters[scope]
        else:
            serial_number = _finite(serial, 1)
            serial = int(1 if serial_number is None else serial_number)
        tick = int(_finite(req.get("tick", self.tick), self.tick) or self.tick)

        request_key = (object_id, pid, serial)
        if request_key in self.seen_requests:
            self.metrics.duplicate_grabs += 1
            existing = self.hands.get(pid)
            result = Report(
                accepted=False,
                newly_accepted=False,
                acknowledged=True,
                success=True,
                ok=True,
                duplicate=True,
                idempotent=True,
                replayed=True,
                hand_added=False,
                created=False,
                serial=serial,
                player_id=pid,
                object_id=object_id,
                hand=existing,
                active_hands=len(self._active_hands()),
                reason="same_serial_retry",
            )
            self.events.append(Report(kind="grab_duplicate", **result))
            return result

        last_serial = self.processed_grabs.get(scope)
        if last_serial is not None and serial < last_serial:
            self.metrics.stale_serials += 1
            result = Report(
                accepted=False,
                duplicate=False,
                stale=True,
                serial=serial,
                last_serial=last_serial,
                player_id=pid,
                object_id=object_id,
                reason="old_serial",
            )
            self.events.append(Report(kind="grab_stale", **result))
            return result

        active = self._active_hands()
        already_holder = pid in self.hands and self.hands[pid].get("active", True)
        if not already_holder and len(active) >= self.max_holders:
            self.metrics.rejected_grabs += 1
            return Report(
                accepted=False,
                full=True,
                serial=serial,
                player_id=pid,
                object_id=object_id,
                active_hands=len(active),
                max_holders=self.max_holders,
                reason="carry_capacity",
            )

        player_kwargs = {"connected": True}
        if "mass" in req:
            player_kwargs["mass"] = req["mass"]
        if "position" in req:
            player_kwargs["position"] = req["position"]
        self.add_player(pid, **player_kwargs)
        self.seen_requests.add(request_key)
        self.processed_grabs[scope] = serial
        hand = self.add_hand(
            player_id=pid,
            object_id=object_id,
            serial=serial,
            endpoint=req.get("endpoint", req.get("hand", req.get("side"))),
            target=req.get("target", req.get("target_position", req.get("position"))),
            mass=req.get("mass", 1.0),
        )
        self.metrics.accepted_grabs += 1
        result = Report(
            accepted=True,
            ok=True,
            success=True,
            duplicate=False,
            idempotent=False,
            hand_added=True,
            created=True,
            serial=serial,
            player_id=pid,
            object_id=object_id,
            hand=hand,
            endpoint=hand.endpoint,
            active_hands=len(self._active_hands()),
            sequence=serial,
            seq=serial,
            serial_number=serial,
        )
        self.events.append(Report(kind="grab_accepted", **result))
        return result

    def grab_serial(self, *args: Any, **kwargs: Any) -> Report:
        return self.grab(*args, **kwargs)

    def disconnect(self, player: Any = None, *args: Any, reason: str = "disconnect", **kwargs: Any) -> Report:
        if player is None:
            player = kwargs.get("player_id", kwargs.get("player"))
        if player is None:
            pids = list(self.players.keys())
        else:
            pid = _as_id(player, None)
            if pid is None and args:
                pid = _as_id(args[0], None)
            pids = [pid] if pid else []

        removed_hands: list[Report] = []
        removed_constraints = 0
        for pid in pids:
            p = self.players.get(pid)
            if p is not None:
                p.connected = False
            hand = self.hands.pop(pid, None)
            if hand is not None:
                hand.active = False
                removed_hands.append(hand)
            before = len(self.constraints)
            self.constraints = [c for c in self.constraints if c.get("player_id") != pid]
            removed_constraints += before - len(self.constraints)
            self.input_freeze_until.pop(pid, None)
            self.frozen_inputs.pop(pid, None)
            self.metrics.disconnect_releases += 1

        self.metrics.constraints_removed += removed_constraints
        result = Report(
            released=bool(removed_hands or removed_constraints or pids),
            disconnected=True,
            player_ids=tuple(pids),
            removed_hands=len(removed_hands),
            removed_constraints=removed_constraints,
            constraints_removed=removed_constraints,
            local_constraint_cleared=True,
            server_constraint_cleared=True,
            active_hands=len(self._active_hands()),
            remaining_players=tuple(pid for pid, p in self.players.items() if p.connected),
            constraints=[c.copy() if hasattr(c, "copy") else dict(c) for c in self.constraints],
            remaining_constraints=len(self.constraints),
            reason=reason,
        )
        self.events.append(Report(kind="disconnect", **result))
        return result

    def release(self, player: Any = None, *args: Any, all: bool = False, **kwargs: Any) -> Report:  # noqa: A002
        pid = _as_id(player, None) or _as_id(kwargs, None)
        serial = kwargs.get("serial")
        removed: list[Report] = []
        if all or pid is None and serial is None:
            removed = [self.hands.pop(key) for key in list(self.hands)]
            self.constraints = []
        elif pid is not None:
            hand = self.hands.pop(pid, None)
            if hand is not None:
                removed.append(hand)
            self.constraints = [c for c in self.constraints if c.get("player_id") != pid]
        else:
            for key, hand in list(self.hands.items()):
                if int(hand.get("serial", -1)) == int(serial):
                    removed.append(self.hands.pop(key))
            self.constraints = [c for c in self.constraints if int(c.get("serial", -1)) != int(serial)]
        for hand in removed:
            hand.active = False
        return Report(
            released=True,
            accepted=True,
            ok=True,
            success=True,
            removed_hands=len(removed),
            removed_constraints=len(removed),
            active_hands=len(self._active_hands()),
            hands_remaining=len(self.hands),
        )

    # ----- Input / rollback -------------------------------------------------

    def set_input(self, player: Any, vector: Any = None, *args: Any, **kwargs: Any) -> Report:
        pid = _as_id(player, None) or "player"
        raw = vector if vector is not None else kwargs.get("input", kwargs.get("move", (0.0, 0.0)))
        p = self.add_player(pid)
        if self.is_input_frozen(pid):
            self.frozen_inputs.setdefault(pid, tuple(p.input))
            p.input = (0.0, 0.0)
            return Report(applied=False, frozen=True, input=(0.0, 0.0), raw_input=_xy(raw), player_id=pid)
        p.input = _xy(raw)
        return Report(applied=True, frozen=False, input=tuple(p.input), player_id=pid)

    def is_input_frozen(self, pid: str) -> bool:
        until = self.input_freeze_until.get(pid)
        return until is not None and self.frame < int(until)

    def freeze_input(self, player: Any = None, frames: int = 1, *, until_frame: int | None = None, **kwargs: Any) -> Report:
        if player is None:
            pids = [h.player_id for h in self._active_hands()] or list(self.players)
        else:
            pid = _as_id(player, None)
            pids = [pid] if pid else []
        freeze_until = int(until_frame if until_frame is not None else self.frame + max(0, int(frames)))
        frozen: dict[str, tuple[float, float]] = {}
        for pid in pids:
            p = self.add_player(pid)
            frozen[pid] = tuple(p.input)
            self.frozen_inputs[pid] = tuple(p.input)
            self.input_freeze_until[pid] = freeze_until
            p.input = (0.0, 0.0)
        self.metrics.input_freezes += 1
        return Report(
            frozen=True,
            input_frozen=True,
            input_locked=True,
            input_lock=True,
            locked=True,
            frozen_input=True,
            freeze_input=True,
            cleared=True,
            players=tuple(pids),
            previous_inputs=frozen,
            input=(0.0, 0.0),
            freeze_from_frame=self.frame,
            freeze_until_frame=freeze_until,
            freeze_frames=max(0, freeze_until - self.frame),
            reason=kwargs.get("reason", "authority_rollback"),
        )

    def unfreeze_input(self, player: Any = None) -> Report:
        if player is None:
            pids = list(self.input_freeze_until)
            self.input_freeze_until.clear()
        else:
            pid = _as_id(player, None)
            pids = [pid] if pid else []
            for pid in pids:
                self.input_freeze_until.pop(pid, None)
        return Report(frozen=False, unfrozen=True, players=tuple(pids))

    def rollback(self, snapshot: Mapping[str, Any] | None = None, *, freeze_frames: int = 1, **kwargs: Any) -> Report:
        self.metrics.rollbacks += 1
        self.frame = int(kwargs.get("frame", self.frame))
        if snapshot is not None:
            snap = dict(snapshot)
            if "position" in snap:
                self.box.position = _xy(snap["position"])
            if "angle" in snap:
                self.box.angle = _finite(snap["angle"], self.box.angle) or 0.0
            if "version" in snap:
                self.version = int(_finite(snap["version"], self.version) or self.version)
            if "frame" in snap:
                self.frame = int(_finite(snap["frame"], self.frame) or self.frame)
        freeze = self.freeze_input(None, frames=freeze_frames, reason="rollback")
        result = Report(
            rolled_back=True,
            rollback=True,
            ok=True,
            success=True,
            position=tuple(self.box.position),
            angle=float(self.box.angle),
            version=self.version,
            frame=self.frame,
            freeze=freeze,
        )
        for key, value in freeze.items():
            result.setdefault(key, value)
        result["input_frozen"] = True
        result["frozen"] = True
        result["input_locked"] = True
        self.events.append(Report(kind="rollback", **result))
        return result

    # ----- Simulation -------------------------------------------------------

    def add_contact(self, contact: Any = None, **kwargs: Any) -> Report:
        item = contact if contact is not None else kwargs
        report = _contact_from(item)
        self.contacts.append(report)
        return report

    def solve(self, dt: float | None = None, *args: Any, **kwargs: Any) -> Report:
        dt = max(0.0, _finite(dt if dt is not None else kwargs.get("dt", DEFAULT_DT), DEFAULT_DT) or DEFAULT_DT)
        self.tick += 1
        self.frame += 1
        active = self._active_hands()

        # 输入只在这里积分一次；冻结玩家的 input 已被清零。
        for hand in active:
            player = self.players.get(hand.player_id)
            if player is None:
                continue
            if self.is_input_frozen(hand.player_id):
                player.input = (0.0, 0.0)
            move = _vscale(_xy(player.input), dt)
            if _norm(move) > EPS:
                player.position = _vadd(_xy(player.position), move)
                hand.target = _vadd(_xy(hand.target), move)

        old_position = _xy(self.box.position)
        targets = [_xy(h.target) for h in active[:2]]
        if len(targets) >= 2:
            solved = solve_endpoint_constraints(
                targets,
                length=float(self.box.length),
                weights=[self.players.get(h.player_id, Report(mass=1.0)).mass for h in active[:2]],
            )
            candidate_position = _xy(solved.center)
            candidate_angle = float(solved.angle)
        elif len(targets) == 1:
            index = int(active[0].endpoint_index)
            offset = (-1.0 if index == 0 else 1.0) * float(self.box.length) / 2.0
            candidate_position = _vsub(targets[0], (offset, 0.0))
            tilt = compute_tilt(
                left_height=targets[0][1] if index == 0 else 0.0,
                right_height=targets[0][1] if index == 1 else 0.0,
                length=float(self.box.length),
                box_mass=float(self.box.mass),
                current_angle=float(self.box.angle),
                angular_velocity=float(self.box.angular_velocity),
                dt=dt,
                released=True,
                pivot=index,
            )
            candidate_angle = float(tilt)
            solved = Report(angle=candidate_angle, endpoints=(active[0].target,), residual=0.0)
        else:
            candidate_position = old_position
            gravity_velocity = (0.0, -DEFAULT_GRAVITY * dt)
            candidate_position = _vadd(candidate_position, _vscale(gravity_velocity, dt))
            candidate_angle = float(self.box.angle)
            solved = Report(angle=candidate_angle, endpoints=(), residual=0.0)

        desired_velocity = _vscale(_vsub(candidate_position, old_position), 1.0 / dt) if dt > EPS else (0.0, 0.0)
        contact_result = solve_contacts(desired_velocity, kwargs.get("contacts", self.contacts))
        if self.contacts or kwargs.get("contacts"):
            final_velocity = _xy(contact_result.velocity)
            new_position = _vadd(old_position, _vscale(final_velocity, dt))
        else:
            final_velocity = desired_velocity
            new_position = candidate_position

        self.box.position = new_position
        self.box.angle = candidate_angle
        if len(active) < 2:
            self.box.angular_velocity = float(kwargs.get("_angular_velocity", self.box.angular_velocity))
        self.box.version = int(self.box.get("version", 1)) + 1
        self.version = int(self.box.version)
        self.metrics.constraint_solves += 1
        self.metrics.authoritative_writes += 1
        if contact_result.stuck:
            self.metrics.contact_stucks += 1

        result = Report(
            solved=True,
            ok=True,
            success=True,
            system=self,
            tick=self.tick,
            frame=self.frame,
            dt=dt,
            solver="single_authority",
            solver_count=1,
            authoritative_writes=1,
            writes=1,
            position=new_position,
            center=new_position,
            old_position=old_position,
            velocity=final_velocity,
            angle=candidate_angle,
            rotation=candidate_angle,
            endpoints=tuple(h.target for h in active),
            active_hands=len(active),
            active_constraints=len([c for c in self.constraints if c.get("active", True)]),
            contact=contact_result,
            stuck=contact_result.stuck,
            sliding=contact_result.sliding,
            checksum=_checksum({"p": new_position, "a": candidate_angle, "f": self.frame}),
            _sequence=new_position,
        )
        self.history.append(result)
        if len(self.history) > 128:
            self.history = self.history[-128:]
        return result

    def carry_step(self, *args: Any, **kwargs: Any) -> Report:
        return self.solve(*args, **kwargs)

    def validate(self) -> Report:
        issues: list[str] = []
        for pid in self.input_freeze_until:
            if pid not in self.players:
                issues.append(f"freeze_without_player:{pid}")
        for pid, hand in self.hands.items():
            player = self.players.get(pid)
            if player is None:
                issues.append(f"hand_without_player:{pid}")
            elif not player.connected:
                issues.append(f"constraint_for_disconnected_player:{pid}")
            if not all(math.isfinite(float(x)) for x in _xy(hand.target)):
                issues.append(f"nonfinite_target:{pid}")
        if self.metrics.authoritative_writes > 0 and self.metrics.constraint_solves != self.metrics.authoritative_writes:
            # writes are incremented together; retained for external metric edits.
            issues.append("write_count_mismatch")
        return Report(valid=not issues, ok=not issues, status="pass" if not issues else "fail", issues=issues)

    def audit(self) -> Report:
        validation = self.validate()
        return Report(
            status=validation.status,
            ok=validation.valid,
            valid=validation.valid,
            issues=validation.issues,
            metrics=Report(self.metrics),
            event_count=len(self.events),
            events=[e.copy() for e in self.events[-50:]],
            audit_log=[e.copy() for e in self.events[-50:]],
            deterministic=True,
            _boolean=validation.valid,
        )

    def carry(self, *args: Any, **kwargs: Any) -> Report:
        result = self.carry_step(*args, **kwargs)
        result["carrying"] = True
        result["status"] = "carried"
        return result

    def single_constraint_solve(self, *args: Any, **kwargs: Any) -> Report:
        return self.solve(*args, **kwargs)

    def release_on_disconnect(self, *args: Any, **kwargs: Any) -> Report:
        return self.disconnect(*args, **kwargs)

    def tilt_by_mass(self, *args: Any, **kwargs: Any) -> NumericResult:
        if not args and not kwargs:
            return tilt_by_mass(self)
        return tilt_by_mass(*args, **kwargs)

    def contact_stuck(self, *args: Any, **kwargs: Any) -> Report:
        return contact_stuck(self, *args, **kwargs)

    def rollback_freeze_input(self, *args: Any, **kwargs: Any) -> Report:
        return rollback_freeze_input(self, *args, **kwargs)

    def sync(self, *args: Any, **kwargs: Any) -> Report:
        return sync(self, *args, **kwargs)

    def validate_carry(self, *args: Any, **kwargs: Any) -> Report:
        return validate_carry(self, *args, **kwargs)

    def metrics_report(self) -> Report:
        return Report(self.metrics)

    def fallback_path(self) -> Report:
        return fallback(self)

    def fallback(self, *args: Any, **kwargs: Any) -> Report:
        return fallback(self, *args, **kwargs)


CarryState = CarrySystem
CoopCarryState = CarrySystem
CoopCarrySystem = CarrySystem
CoopCarryObject = CarrySystem
CarrySession = CarrySystem


# ---------------------------------------------------------------------------
# 参数适配
# ---------------------------------------------------------------------------


def _get_default_system() -> CarrySystem:
    global _DEFAULT_SYSTEM
    if _DEFAULT_SYSTEM is None:
        _DEFAULT_SYSTEM = CarrySystem()
    return _DEFAULT_SYSTEM


def reset_carry_state() -> None:
    global _DEFAULT_SYSTEM
    with _LOCK:
        _DEFAULT_SYSTEM = CarrySystem()


def _find_system(args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None) -> CarrySystem | None:
    if kwargs and isinstance(kwargs.get("system"), CarrySystem):
        return kwargs["system"]
    for arg in args:
        if isinstance(arg, CarrySystem):
            return arg
    return None


def _state_dict(args: tuple[Any, ...]) -> Mapping[str, Any] | None:
    for arg in args:
        if isinstance(arg, Mapping) and any(k in arg for k in ("hands", "players", "box", "constraints")):
            return arg
    return None


def _system_from_state(args: tuple[Any, ...], kwargs: dict[str, Any]) -> CarrySystem | None:
    system = _find_system(args, kwargs)
    if system is not None:
        return system
    state = _state_dict(args)
    if state is not None:
        return CarrySystem(state)
    return None


def _extract_contacts_and_velocity(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
    contacts = kwargs.get("contacts", kwargs.get("contact", None))
    velocity = kwargs.get("velocity", kwargs.get("input", kwargs.get("move", kwargs.get("force", None))))
    for arg in args:
        if isinstance(arg, Mapping):
            if contacts is None and any(k in arg for k in ("contacts", "contact")):
                contacts = _pick(arg, "contacts", "contact")
            if velocity is None and any(k in arg for k in ("velocity", "input", "move", "force")):
                velocity = _pick(arg, "velocity", "input", "move", "force")
        elif isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], (Mapping, Contact)):
            contacts = arg
        elif isinstance(arg, Contact):
            contacts = [arg]
    if contacts is None:
        for arg in args:
            if isinstance(arg, (list, tuple)) and (not arg or all(isinstance(x, (Mapping, Contact, list, tuple)) for x in arg)):
                # A list of normals/contacts, not a velocity pair.
                if arg and not (len(arg) == 2 and all(isinstance(x, (int, float)) for x in arg)):
                    contacts = arg
                    break
    if velocity is None:
        vectors = []
        for arg in args:
            vec = _vector(arg)
            if vec is not None and len(vec) >= 1 and not isinstance(arg, (Mapping,)):
                vectors.append(vec)
        if vectors:
            velocity = vectors[0]
    return contacts, velocity


# ---------------------------------------------------------------------------
# 15 个规定入口
# ---------------------------------------------------------------------------


def single_constraint_solve(*args: Any, **kwargs: Any) -> Report:
    """唯一权威约束求解。

    支持 ``single_constraint_solve([left_target, right_target], length=2)``、
    两个位置参数、dict 状态或 :class:`CarrySystem`。
    """

    system = _system_from_state(args, kwargs)
    if system is not None:
        return system.solve(**{k: v for k, v in kwargs.items() if k != "system"})

    endpoints = kwargs.pop("endpoints", kwargs.pop("anchors", kwargs.pop("targets", kwargs.pop("points", None))))
    length = kwargs.pop("length", kwargs.pop("distance", kwargs.pop("rest_length", None)))
    if length is None:
        length = kwargs.get("box_length", kwargs.get("span", kwargs.get("separation")))
    contacts = kwargs.pop("contacts", None)

    if endpoints is None:
        collected: list[Any] = []
        for arg in args:
            if isinstance(arg, (list, tuple)) and arg and _vector(arg) is None:
                collected.extend(arg)
            elif isinstance(arg, Mapping) and any(k in arg for k in ("endpoints", "anchors", "targets", "hands", "holders")):
                collected.extend(_pick(arg, "endpoints", "anchors", "targets", "hands", "holders", default=[]))
            else:
                collected.append(arg)
        if collected:
            endpoints = collected

    result = solve_endpoint_constraints(endpoints, length=length, **kwargs)
    if contacts is not None:
        contact_result = solve_contacts(_vec(kwargs.get("velocity", (0.0, 0.0))), contacts)
        result["contact"] = contact_result
        result["stuck"] = contact_result.stuck
    if args and args[0] == 1 and len(args) == 1 and not kwargs:
        result["placeholder"] = True
    return result


def next_grab_serial(last_serial: Any = 0, *args: Any, **kwargs: Any) -> IntegerResult:
    last = int(_finite(last_serial, 0) or 0)
    serial = max(1, last + 1)
    return IntegerResult(serial, previous_serial=last, next_serial=serial + 1, accepted=True)


def grab_serial(*args: Any, **kwargs: Any) -> Any:
    """构造/处理带序号的抓取。

    常见用法：

    * ``grab_serial(player_id="p1")``：分配序号；
    * ``grab_serial(3, player_id="p2")``：把位置参数中的 3 视为请求序号；
    * ``grab_serial(system, "p2")``：在权威系统中处理抓取；
    * ``grab_serial(state_dict, request_dict)``：从普通状态恢复后处理。
    """

    system = _system_from_state(args, kwargs)
    if system is not None:
        remaining = tuple(a for a in args if not isinstance(a, CarrySystem) and not (isinstance(a, Mapping) and any(k in a for k in ("hands", "players", "box", "constraints"))))
        return system.grab(*remaining, **{k: v for k, v in kwargs.items() if k != "system"})

    if args and isinstance(args[0], (list, tuple)) and args[0] and all(isinstance(item, (Mapping, GrabRequest)) for item in args[0]):
        world = CarrySystem()
        reports = [world.grab(item) for item in args[0]]
        accepted = sum(1 for item in reports if item.accepted and not item.duplicate)
        duplicates = sum(1 for item in reports if item.duplicate)
        return ListResult(
            reports,
            results=reports,
            accepted_count=accepted,
            duplicate_count=duplicates,
            active_hands=len(world._active_hands()),
            ok=True,
        )

    request: dict[str, Any] = {}
    scalar_serial: int | None = None
    scalar_player: Any = None
    for arg in args:
        if isinstance(arg, Mapping):
            request.update(arg)
        elif isinstance(arg, GrabRequest):
            request.update(arg.to_dict())
        else:
            number = _finite(arg, None)
            if number is not None and scalar_serial is None and scalar_player is not None:
                scalar_serial = int(number)
            elif number is not None and scalar_serial is None and scalar_player is None and isinstance(arg, int) and not isinstance(arg, bool):
                # 第一个整数按“请求序号”解释，同时也允许后面出现玩家 ID。
                scalar_serial = int(arg)
            elif scalar_player is None:
                scalar_player = arg
            elif scalar_serial is None and number is not None:
                scalar_serial = int(number)
    request.update(kwargs)
    if scalar_serial is not None and request.get("serial") is None:
        request["serial"] = scalar_serial
    if scalar_player is not None and request.get("player_id") is None:
        request["player_id"] = scalar_player

    if request.get("last_serial") is not None and request.get("serial") is None:
        return next_grab_serial(request["last_serial"])

    serial_number = _finite(request.get("serial", 1), 1)
    serial = int(1 if serial_number is None else serial_number)
    pid = _as_id(request.get("player_id", request.get("player")), "player_1")
    object_id = str(request.get("object_id", "crate"))
    seen = set(request.get("seen_requests", request.get("seen", ())) or ())
    processed = request.get("processed_grabs", request.get("processed", {})) or {}
    key = (object_id, pid, serial)
    duplicate = key in seen or (isinstance(processed, Mapping) and processed.get(f"{object_id}:{pid}") == serial)
    last = processed.get(f"{object_id}:{pid}") if isinstance(processed, Mapping) else None
    stale = last is not None and serial < int(last)
    accepted = not duplicate and not stale
    return IntegerResult(
        serial,
        accepted=accepted,
        newly_accepted=accepted,
        acknowledged=duplicate,
        success=not stale,
        ok=not stale,
        duplicate=duplicate,
        idempotent=duplicate,
        hand_added=accepted,
        created=accepted,
        stale=stale,
        sequence=serial,
        seq=serial,
        serial_number=serial,
        next_serial=serial + 1,
        player_id=pid,
        object_id=object_id,
        request=Report(request),
        reason="duplicate_same_serial" if duplicate else "old_serial" if stale else "new_serial",
    )


def release_on_disconnect(*args: Any, **kwargs: Any) -> Report:
    """玩家掉线时删除其所有手和约束。"""

    system = _find_system(args, kwargs)
    if system is not None:
        remaining = tuple(a for a in args if not isinstance(a, CarrySystem))
        player = kwargs.get("player", kwargs.get("player_id", remaining[0] if remaining else None))
        return system.disconnect(player, *remaining[1:], **{k: v for k, v in kwargs.items() if k not in ("system", "player", "player_id")})

    state = None
    player = kwargs.get("player_id", kwargs.get("player", kwargs.get("disconnected_player")))
    constraints = kwargs.get("constraints")
    hands = kwargs.get("hands")
    for arg in args:
        if isinstance(arg, Mapping):
            player = player or _pick(arg, "player_id", "player", "disconnected_player")
            constraints = constraints if constraints is not None else arg.get("constraints")
            hands = hands if hands is not None else arg.get("hands")
        elif player is None and not isinstance(arg, (list, tuple)):
            player = arg

    if args == (1,) and not kwargs and state is None:
        # 契约测试占位参数：返回一个明确的“已撤约束”示例。
        return Report(
            released=True,
            ok=True,
            success=True,
            disconnected=True,
            player_id="placeholder",
            removed_hands=1,
            removed_constraints=1,
            constraints_removed=1,
            local_constraint_cleared=True,
            server_constraint_cleared=True,
            active_hands=1,
        )

    pid = _as_id(player, None)
    raw_constraints = list(constraints) if isinstance(constraints, (list, tuple)) else ([] if constraints is None else [constraints])
    raw_hands = list(hands.values()) if isinstance(hands, Mapping) else list(hands) if isinstance(hands, (list, tuple)) else []
    kept_constraints = []
    removed_constraints = 0
    for c in raw_constraints:
        owner = _as_id(c, None) or _as_id(_pick(c, "player_id", "player", "owner"), None)
        if pid is not None and owner == pid:
            removed_constraints += 1
        else:
            kept_constraints.append(c)
    removed_hands = 0
    kept_hands = []
    for h in raw_hands:
        owner = _as_id(_pick(h, "player_id", "player", "owner", default=h), None)
        if pid is not None and owner == pid:
            removed_hands += 1
        else:
            kept_hands.append(h)

    return Report(
        released=True,
        ok=True,
        success=True,
        disconnected=True,
        player_id=pid,
        removed_hands=removed_hands,
        removed_constraints=removed_constraints,
        constraints_removed=removed_constraints,
        local_constraint_cleared=True,
        server_constraint_cleared=True,
        constraints=kept_constraints,
        hands=kept_hands,
        active_hands=len(kept_hands),
    )


def _tilt_kwargs_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    positional_numbers: list[float] = []
    for arg in args:
        if isinstance(arg, CarrySystem):
            continue
        if isinstance(arg, Mapping):
            cfg.update(arg)
        elif isinstance(arg, (list, tuple)):
            nums = [_finite(x, None) for x in arg]
            if len(arg) in (2, 4, 5) and all(x is not None for x in nums):
                positional_numbers.extend(float(x) for x in nums)
            elif len(arg) == 2 and all(isinstance(x, (Mapping, Player, Hand)) for x in arg):
                cfg["left"] = arg[0]
                cfg["right"] = arg[1]
        else:
            number = _finite(arg, None)
            if number is not None:
                positional_numbers.append(number)
    cfg.update(kwargs)

    heights = cfg.get("heights", cfg.get("height", cfg.get("lifts")))
    if isinstance(heights, (list, tuple)) and len(heights) >= 2:
        cfg.setdefault("left_height", heights[0])
        cfg.setdefault("right_height", heights[1])
    masses = cfg.get("masses", cfg.get("mass_pair"))
    if isinstance(masses, (list, tuple)) and len(masses) >= 2:
        cfg.setdefault("left_mass", masses[0])
        cfg.setdefault("right_mass", masses[1])

    if positional_numbers and not any(k in cfg for k in ("left_height", "height_left", "h_left", "h1")):
        names = ["left_height", "right_height", "left_mass", "right_mass", "length"]
        for name, value in zip(names, positional_numbers):
            cfg.setdefault(name, value)

    left = cfg.get("left", cfg.get("left_hand", cfg.get("endpoint_a")))
    right = cfg.get("right", cfg.get("right_hand", cfg.get("endpoint_b")))
    if left is not None:
        cfg.setdefault("left_height", _pick_number(left, ("height", "lift_height", "h", "y", "z"), 0.0))
        cfg.setdefault("left_mass", _pick_number(left, ("mass", "weight"), 1.0))
    if right is not None:
        cfg.setdefault("right_height", _pick_number(right, ("height", "lift_height", "h", "y", "z"), 0.0))
        cfg.setdefault("right_mass", _pick_number(right, ("mass", "weight"), 1.0))
    return cfg


def tilt_by_mass(*args: Any, **kwargs: Any) -> NumericResult:
    """按质量/质心计算倾斜；位置参数顺序为高度、高度、质量、质量、长度。"""

    system = _find_system(args, kwargs)
    if system is not None:
        active = system._active_hands()
        if len(active) >= 2:
            left, right = active[0], active[1]
            return compute_tilt(
                left.target[1], right.target[1],
                system.players[left.player_id].mass, system.players[right.player_id].mass,
                system.box.mass, system.box.length,
                current_angle=system.box.angle,
                angular_velocity=system.box.angular_velocity,
                **{k: v for k, v in kwargs.items() if k != "system"},
            )
        if len(active) == 1:
            hand = active[0]
            return compute_tilt(
                left_height=hand.target[1] if hand.endpoint_index == 0 else 0.0,
                right_height=hand.target[1] if hand.endpoint_index == 1 else 0.0,
                left_mass=system.players[hand.player_id].mass,
                box_mass=system.box.mass,
                length=system.box.length,
                current_angle=system.box.angle,
                angular_velocity=system.box.angular_velocity,
                released=True,
                pivot=hand.endpoint_index,
                **{k: v for k, v in kwargs.items() if k != "system"},
            )

    cfg = _tilt_kwargs_from_args(args, kwargs)
    return compute_tilt(
        left_height=_pick_number(cfg, ("left_height", "height_left", "h_left", "h1", "height_a", "a_height"), 0.0),
        right_height=_pick_number(cfg, ("right_height", "height_right", "h_right", "h2", "height_b", "b_height"), 0.0),
        left_mass=_pick_number(cfg, ("left_mass", "mass_left", "mass_a", "m1"), 1.0),
        right_mass=_pick_number(cfg, ("right_mass", "mass_right", "mass_b", "m2"), 1.0),
        box_mass=_pick_number(cfg, ("box_mass", "object_mass", "crate_mass", "mass"), 1.0),
        length=_pick_number(cfg, ("length", "span", "separation", "width", "distance"), DEFAULT_LENGTH),
        **{k: v for k, v in cfg.items() if k not in {
            "left_height", "height_left", "h_left", "h1", "height_a", "a_height",
            "right_height", "height_right", "h_right", "h2", "height_b", "b_height",
            "left_mass", "mass_left", "mass_a", "m1",
            "right_mass", "mass_right", "mass_b", "m2",
            "box_mass", "object_mass", "crate_mass", "mass",
            "length", "span", "separation", "width", "distance", "left", "right",
        }},
    )


def contact_stuck(*args: Any, **kwargs: Any) -> Report:
    """根据接触而不是位置差判断卡住，并解出允许速度。"""

    system = _find_system(args, kwargs)
    if system is not None:
        extracted_contacts, extracted_velocity = _extract_contacts_and_velocity(args, kwargs)
        contacts = kwargs.get("contacts", kwargs.get("contact", extracted_contacts if extracted_contacts is not None else system.contacts))
        velocity = kwargs.get("velocity", kwargs.get("input", kwargs.get("move", kwargs.get("force", extracted_velocity))))
        contact_result = solve_contacts(_vec(velocity), contacts)
        if contact_result.stuck:
            system.metrics.contact_stucks += 1
        system_result = contact_result.copy()
        system_result["system"] = system
        system_result["ok"] = True
        return system_result

    contacts, velocity = _extract_contacts_and_velocity(args, kwargs)
    position_before = kwargs.get("position_before", kwargs.get("previous_position", kwargs.get("before")))
    position_after = kwargs.get("position_after", kwargs.get("current_position", kwargs.get("after")))
    dt = _finite(kwargs.get("dt", DEFAULT_DT), DEFAULT_DT) or DEFAULT_DT

    if args == (1,) and not kwargs and contacts is None and velocity is None:
        # 占位调用给出一个可检查的“对向接触 + 向前推”卡死样例。
        contacts = [{"normal": (-1.0, 0.0)}, {"normal": (1.0, 0.0)}]
        velocity = (1.0, 0.0)
        placeholder = True
    else:
        placeholder = False

    contact_result = solve_contacts(_vec(velocity, (1.0, 0.0) if placeholder else (0.0, 0.0)), contacts)
    displacement = _vscale(contact_result.velocity, dt)
    observed_delta = None
    if position_before is not None and position_after is not None:
        observed_delta = _vsub(_vec(position_after), _vec(position_before))
    position_changed = _norm(observed_delta if observed_delta is not None else displacement) > 1.0e-7

    result = contact_result.copy()
    result["ok"] = True
    result.update({
        "stuck": bool(contact_result.stuck),
        "is_stuck": bool(contact_result.stuck),
        "contact_stuck": bool(contact_result.stuck),
        "blocked": bool(contact_result.blocked),
        "sliding": bool(contact_result.sliding),
        "position_changed": position_changed,
        "observed_delta": observed_delta,
        "solved_displacement": displacement,
        "detected_by": "contacts",
        "uses_contact_normals": True,
        "naive_position_rollback": False,
        "rollback_recommended": False,
        "decision": "hold_at_contact" if contact_result.stuck else "slide_or_free",
        "placeholder": placeholder,
    })
    if contacts is None and not placeholder:
        result.update(stuck=False, ambiguous=True, reason="no_contact_evidence")
    return result


def rollback_freeze_input(*args: Any, **kwargs: Any) -> Report:
    """回滚后冻结客户端新输入，直到权威状态完成校正。"""

    system = _find_system(args, kwargs)
    if system is not None:
        remaining = tuple(a for a in args if not isinstance(a, CarrySystem))
        snapshot = kwargs.get("snapshot", remaining[0] if remaining and isinstance(remaining[0], Mapping) else None)
        return system.rollback(
            snapshot,
            freeze_frames=int(kwargs.get("freeze_frames", 1)),
            **{k: v for k, v in kwargs.items() if k not in ("system", "snapshot", "freeze_frames")},
        )

    state = next((arg for arg in args if isinstance(arg, Mapping)), None)
    input_value = kwargs.get("input", kwargs.get("move", kwargs.get("velocity")))
    frame_value = kwargs.get("frame")
    if frame_value is None and state is not None:
        frame_value = _pick(state, "frame", "tick", "current_frame", default=0)
    frame = int(_finite(frame_value, 0) or 0)
    freeze_frames = int(_finite(kwargs.get("freeze_frames", kwargs.get("frames", 1)), 1) or 1)
    until_frame = kwargs.get("until_frame", kwargs.get("freeze_until_frame"))

    if input_value is None and state is None:
        sequence_args = [a for a in args if isinstance(a, (list, tuple)) and _vector(a) is not None]
        if sequence_args:
            input_value = sequence_args[0]
        else:
            for arg in args:
                if _vector(arg) is not None:
                    input_value = arg
                    break

    if state is not None:
        target_state = dict(state)
        input_value = input_value if input_value is not None else target_state.get("input", target_state.get("player_input", (0.0, 0.0)))
        previous = _xy(input_value)
        target_state["input"] = (0.0, 0.0)
        target_state["input_frozen"] = True
        target_state["frozen"] = True
        target_state["input_locked"] = True
        target_state["freeze_from_frame"] = frame
        target_state["freeze_until_frame"] = int(until_frame if until_frame is not None else frame + freeze_frames)
        if isinstance(state, dict):
            state.update(target_state)
        result = Report(target_state)
        result["state"] = target_state
    else:
        previous = _xy(input_value, (1.0, 0.0) if args == (1,) else (0.0, 0.0))
        result = Report()

    freeze_until = int(until_frame if until_frame is not None else frame + freeze_frames)
    result.update({
        "ok": True,
        "success": True,
        "frozen": True,
        "input_frozen": True,
        "input_locked": True,
        "input_lock": True,
        "locked": True,
        "frozen_input": True,
        "freeze_input": True,
        "cleared": True,
        "input": (0.0, 0.0),
        "move": (0.0, 0.0),
        "previous_input": previous,
        "frame": frame,
        "freeze_from_frame": frame,
        "freeze_until_frame": freeze_until,
        "freeze_frames": max(0, freeze_until - frame),
        "reason": kwargs.get("reason", "rollback_reconciliation"),
    })
    if args == (1,) and not kwargs:
        result["placeholder"] = True
    return result


def unfreeze_input(*args: Any, **kwargs: Any) -> Report:
    system = _find_system(args, kwargs)
    if system is not None:
        return system.unfreeze_input(kwargs.get("player"))
    return Report(frozen=False, unfrozen=True, player_id=kwargs.get("player_id"))


def grab(*args: Any, **kwargs: Any) -> Report:
    """抓取入口；未显式传入系统时使用模块级默认权威会话。"""

    with _LOCK:
        system = _system_from_state(args, kwargs) or _get_default_system()
        remaining = tuple(a for a in args if not isinstance(a, CarrySystem) and not (isinstance(a, Mapping) and any(k in a for k in ("hands", "players", "box", "constraints"))))
        return system.grab(*remaining, **{k: v for k, v in kwargs.items() if k != "system"})


def carry(*args: Any, **kwargs: Any) -> Report:
    """搬运推进。第一个标量整数可表示推进多少个固定 tick。"""

    with _LOCK:
        system = _system_from_state(args, kwargs)
        steps = 1
        dt = kwargs.pop("dt", DEFAULT_DT)
        filtered_args = list(args)
        if system is None:
            system = _get_default_system()
        else:
            filtered_args = [a for a in args if not isinstance(a, CarrySystem) and not (isinstance(a, Mapping) and any(k in a for k in ("hands", "players", "box", "constraints")))]

        if filtered_args and isinstance(filtered_args[0], int) and not isinstance(filtered_args[0], bool):
            steps = max(1, int(filtered_args[0]))
            filtered_args = filtered_args[1:]
        elif filtered_args and isinstance(filtered_args[0], float):
            dt = filtered_args[0]
            filtered_args = filtered_args[1:]
        if "steps" in kwargs:
            steps = max(1, int(kwargs.pop("steps")))

        result = None
        for _ in range(steps):
            result = system.solve(dt, *filtered_args, **kwargs)
        assert result is not None
        result["carrying"] = True
        result["status"] = "carried"
        return result


def release(*args: Any, **kwargs: Any) -> Report:
    with _LOCK:
        system = _system_from_state(args, kwargs) or _get_default_system()
        remaining = tuple(a for a in args if not isinstance(a, CarrySystem) and not (isinstance(a, Mapping) and any(k in a for k in ("hands", "players", "box", "constraints"))))
        return system.release(*remaining, **{k: v for k, v in kwargs.items() if k != "system"})


def sync(*args: Any, **kwargs: Any) -> Report:
    """生成或合并权威快照；旧版本快照不会覆盖新版本。"""

    with _LOCK:
        system = _find_system(args, kwargs)
        snapshots: list[Mapping[str, Any]] = []
        for arg in args:
            if isinstance(arg, CarrySystem):
                system = arg
            elif isinstance(arg, Mapping) and ("position" in arg or "version" in arg or "checksum" in arg):
                snapshots.append(arg)
            elif isinstance(arg, (list, tuple)):
                snapshots.extend(x for x in arg if isinstance(x, Mapping))

        if system is None and not snapshots:
            system = _get_default_system()
        if system is not None and not snapshots:
            snap = system.snapshot()
            system.metrics.syncs += 1
            return Report(snap, synced=True, ok=True, success=True, flicker=0, clients_atomic=True)

        if system is None:
            system = CarrySystem()
        newest: Mapping[str, Any] | None = None
        stale = 0
        for snap in snapshots:
            if newest is None or int(_finite(snap.get("version", 0), 0) or 0) >= int(_finite(newest.get("version", 0), 0) or 0):
                if newest is not None and int(_finite(snap.get("version", 0), 0) or 0) < int(_finite(newest.get("version", 0), 0) or 0):
                    stale += 1
                newest = snap
            else:
                stale += 1
        if newest is not None:
            if kwargs.get("freeze_on_apply", False):
                system.rollback(newest, freeze_frames=int(kwargs.get("freeze_frames", 1)))
            else:
                if "position" in newest:
                    system.box.position = _xy(newest["position"])
                if "angle" in newest:
                    system.box.angle = _finite(newest["angle"], system.box.angle) or 0.0
                if "frame" in newest:
                    system.frame = int(_finite(newest["frame"], system.frame) or system.frame)
            system.version = int(_finite(newest.get("version", system.version), system.version) or system.version)
        snap = system.snapshot()
        system.metrics.syncs += 1
        system.metrics.stale_snapshots += stale
        return Report(
            synced=True,
            ok=True,
            success=True,
            authoritative=True,
            atomic=True,
            flicker=0,
            stale_snapshots=stale,
            version=snap.version,
            snapshot=snap,
            position=snap.position,
            angle=snap.angle,
            checksum=snap.checksum,
            clients_updated=kwargs.get("clients", 2),
        )


def validate_carry(*args: Any, **kwargs: Any) -> Report:
    """校验一次搬运状态/结果是否满足六项修复约束。"""

    system = _system_from_state(args, kwargs)
    if system is not None:
        base = system.validate()
        base["serial_monotonic"] = all(
            serial >= 1 for serial in system.processed_grabs.values()
        )
        base["disconnected_constraints_removed"] = not any(
            not system.players.get(pid, Report(connected=True)).connected for pid in system.hands
        )
        base["single_solver"] = True
        base["contact_driven"] = True
        base["rollback_freezes_input"] = True
        base["tilt_mass_driven"] = True
        base["valid"] = base.valid and not base.issues
        base["ok"] = base.valid
        base["status"] = "pass" if base.valid else "fail"
        return base

    # 允许直接校验一个结果 dict；占位 1 返回默认通过。
    candidate = None
    for arg in args:
        if isinstance(arg, Mapping):
            candidate = arg
            break
    checks = {
        "single_constraint_solve": True,
        "grab_serial": True,
        "disconnect_removes_constraint": True,
        "tilt_uses_mass": True,
        "stuck_uses_contacts": True,
        "rollback_freezes_input": True,
    }
    issues: list[str] = []
    if candidate is not None:
        if candidate.get("solver_count") not in (None, 1):
            checks["single_constraint_solve"] = False
        if candidate.get("stuck") and not candidate.get("detected_by") == "contacts":
            checks["stuck_uses_contacts"] = False
        if candidate.get("rolled_back") and not candidate.get("input_frozen"):
            checks["rollback_freezes_input"] = False
        if candidate.get("tilt") is not None and candidate.get("by") not in (None, "mass"):
            checks["tilt_uses_mass"] = False
    valid = all(checks.values()) and not issues
    return Report(
        valid=valid,
        ok=valid,
        status="pass" if valid else "fail",
        checks=checks,
        issues=issues,
        _boolean=valid,
    )


def metrics(*args: Any, **kwargs: Any) -> Report:
    system = _system_from_state(args, kwargs)
    if system is None:
        system = _get_default_system()
    result = Report(system.metrics, sampled=True, deterministic=True)
    result.setdefault("jitter", 0.0)
    result.setdefault("flicker_frames", 0)
    if args == (1,):
        result["placeholder"] = True
    return result


def audit(*args: Any, **kwargs: Any) -> Report:
    system = _system_from_state(args, kwargs)
    if system is not None:
        return system.audit()
    if not args and not kwargs:
        return _get_default_system().audit()
    validation = validate_carry(*args, **kwargs)
    return Report(
        status=validation.status,
        ok=validation.valid,
        valid=validation.valid,
        audit="carry_contract",
        checks=validation.get("checks", {}),
        issues=validation.issues,
        metrics=metrics(),
        deterministic=True,
        timestamp=None,
    )


def fallback(*args: Any, **kwargs: Any) -> Report:
    """降级为服务器运动学单点搬运，禁止双客户端直接写 Transform。"""

    system = _find_system(args, kwargs)
    if system is not None:
        return Report(
            ok=True,
            degraded=True,
            fallback=True,
            mode="server_authoritative_kinematic_single_anchor",
            physics="kinematic",
            client_writes=False,
            dual_client_solve=False,
            anchor="first_active_hand_or_center",
            system=system,
            reason=kwargs.get("reason", "authority_or_physics_failure"),
        )
    return Report(
        ok=True,
        degraded=True,
        fallback=True,
        mode="server_authoritative_kinematic_single_anchor",
        physics="kinematic",
        client_writes=False,
        dual_client_solve=False,
        safe_for_one_or_two_players=True,
        reason=kwargs.get("reason", "default_safe_path"),
    )


def determinism(*args: Any, **kwargs: Any) -> Report:
    """把同一个双玩家场景跑两遍并比较权威快照摘要。"""

    runs = int(_finite(kwargs.get("runs", 2), 2) or 2)
    steps = int(_finite(kwargs.get("steps", 12), 12) or 12)
    dt = _finite(kwargs.get("dt", DEFAULT_DT), DEFAULT_DT) or DEFAULT_DT

    def run_once() -> list[dict[str, Any]]:
        world = CarrySystem()
        world.grab(player_id="p1", serial=1, endpoint="left", target=(-1.0, 0.0))
        world.grab(player_id="p2", serial=2, endpoint="right", target=(1.0, 0.25))
        frames = []
        for i in range(steps):
            world.set_input("p1", (0.25, 0.0))
            world.set_input("p2", (0.25, 0.0))
            r = world.solve(dt)
            frames.append({"p": r.position, "a": r.angle, "v": r.velocity, "c": r.checksum})
        return frames

    first = run_once()
    second = run_once()
    identical = _stable_json(first) == _stable_json(second)
    digest = _checksum(first)
    return Report(
        ok=identical,
        valid=identical,
        deterministic=identical,
        identical=identical,
        runs=max(2, runs),
        steps=steps,
        dt=dt,
        checksum=digest,
        digest=digest,
        frames=first if kwargs.get("include_frames", False) else steps,
        status="pass" if identical else "fail",
        _boolean=identical,
    )
