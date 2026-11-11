-- 틈(Teum) DB 초안 v0.2 (2026-11-11, docs/SPEC.md 경고 단계 반영) · PostgreSQL 16
-- 표 4개: devices(폰) → drives(주행) → events(아찔한 순간) → clips(영상)
-- 앱 지표는 맨 아래 뷰로 계산 (따로 저장 안 함)

-- 1) 기기: 로그인 없이 앱 설치마다 UUID 하나 (Play 비공개테스트 12명 구분용)
CREATE TABLE devices (
    id          UUID PRIMARY KEY,                 -- 앱이 처음 켤 때 만들어 보냄
    nickname    TEXT,
    camera_cfg  TEXT NOT NULL DEFAULT 'default',  -- configs/*.yaml 이름 (hood_ratio 등)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2) 주행: 실시간 1회 = 1행, 블랙박스 영상 분석 1건 = 1행
CREATE TABLE drives (
    id            BIGSERIAL PRIMARY KEY,
    device_id     UUID NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    source        TEXT NOT NULL CHECK (source IN ('live', 'upload')),
    video_name    TEXT,                           -- upload일 때 원본 파일명
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at      TIMESTAMPTZ,                    -- NULL = 주행 중
    distance_km   REAL,                           -- GPS 누적 (없으면 NULL)
    avg_speed_kmh REAL,
    max_speed_kmh REAL,
    frames        INTEGER,                        -- 처리한 프레임 수 (성능 평가용)
    avg_latency_ms REAL,                          -- 왕복 지연 평균 (성능 평가용)
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX drives_device_started ON drives (device_id, started_at DESC);

-- 3) 아찔한 순간: SPEC 기준 "경고"(TTC 3초 이하) 이상일 때만 기록 → level은 warning/danger 둘뿐
CREATE TABLE events (
    id             BIGSERIAL PRIMARY KEY,
    drive_id       BIGINT NOT NULL REFERENCES drives(id) ON DELETE CASCADE,
    occurred_at    TIMESTAMPTZ NOT NULL,          -- 실제 시각
    offset_s       REAL NOT NULL,                 -- 주행(영상) 시작부터 몇 초 (upload 영상 위치 찾기용)
    type           TEXT NOT NULL CHECK (type IN
                     ('lead',      -- 앞차 급접근 (TTC)
                      'cut_in',    -- 끼어들기
                      'brake',     -- 앞차 브레이크등
                      'lane_departure', -- 차선 이탈
                      'rear',      -- 뒷차 급접근
                      'two_wheeler')),  -- 이륜차
    level          TEXT NOT NULL CHECK (level IN ('warning', 'danger')),  -- 경고(TTC≤3s) / 위험(TTC≤1.5s), 이벤트 중 최고 단계
    min_ttc_s      REAL,                          -- 이벤트 동안 가장 낮았던 TTC
    distance_m     REAL,                          -- 그 순간 앞차 거리 (핀홀)
    lateral_m      REAL,                          -- 옆 거리 (끼어들기·이륜차)
    speed_kmh      REAL,                          -- 내 속도
    lat            DOUBLE PRECISION,
    lon            DOUBLE PRECISION,
    target_class   TEXT,                          -- car/truck/bus/motorcycle ...
    duration_s     REAL,                          -- 경고가 이어진 시간
    detail         JSONB NOT NULL DEFAULT '{}',   -- 타입별 추가값 (track_id, bbox 등) — 표 안 바꾸고 확장
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX events_drive_time ON events (drive_id, occurred_at);
CREATE INDEX events_type_level ON events (type, level);

-- 4) 클립: 이벤트 1개당 0~1개. 파일은 디스크/S3, DB엔 위치만
CREATE TABLE clips (
    id            BIGSERIAL PRIMARY KEY,
    event_id      BIGINT NOT NULL UNIQUE REFERENCES events(id) ON DELETE CASCADE,
    storage       TEXT NOT NULL DEFAULT 'local' CHECK (storage IN ('local', 's3')),
    object_key    TEXT NOT NULL,                  -- 예: clips/{device}/{drive}/{event}.mp4
    thumb_key     TEXT,                           -- 목록 화면 썸네일 jpg
    pre_s         REAL NOT NULL DEFAULT 5,        -- 이벤트 전 초 (실시간 모드는 경고 이후만 저장 → 0)
    post_s        REAL NOT NULL DEFAULT 5,        -- 이벤트 후 초
    fps           REAL,
    width         INTEGER,
    height        INTEGER,
    size_bytes    BIGINT,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'ready', 'failed')),  -- post_s 지나야 ready
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ===== 앱 지표용 뷰 =====

-- 주행 목록 화면: 주행마다 시간·거리·이벤트 수
CREATE VIEW drive_summary AS
SELECT d.id, d.device_id, d.source, d.started_at, d.ended_at,
       EXTRACT(EPOCH FROM (d.ended_at - d.started_at)) / 60 AS minutes,
       d.distance_km,
       COUNT(e.id)                                   AS events,
       COUNT(e.id) FILTER (WHERE e.level = 'danger') AS danger_events,
       MIN(e.min_ttc_s)                              AS worst_ttc_s
FROM drives d LEFT JOIN events e ON e.drive_id = d.id
GROUP BY d.id;

-- 통계 화면: 기기별 타입·단계별 횟수 + 100km당 (거리는 기기 전체 주행거리)
CREATE VIEW device_stats AS
WITH km AS (SELECT device_id, SUM(distance_km) AS km FROM drives GROUP BY device_id)
SELECT d.device_id, e.type, e.level, COUNT(*) AS n,
       COUNT(*) * 100.0 / NULLIF(MAX(km.km), 0) AS per_100km
FROM events e JOIN drives d ON d.id = e.drive_id JOIN km ON km.device_id = d.device_id
GROUP BY d.device_id, e.type, e.level;

-- 시간대별 (낮/밤 비교, 발표용)
CREATE VIEW events_by_hour AS
SELECT EXTRACT(HOUR FROM occurred_at AT TIME ZONE 'Asia/Seoul')::INT AS hour,
       type, COUNT(*) AS n
FROM events GROUP BY 1, 2;
