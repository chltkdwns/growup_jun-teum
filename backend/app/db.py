"""PostgreSQL 연결. 접속 주소는 .env의 DATABASE_URL에서 읽는다."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# 레포 루트의 .env를 찾아 읽는다 (이 파일 위치에서 위로 올라가며 찾음)
load_dotenv()

engine = create_engine(os.environ["DATABASE_URL"])
SessionLocal = sessionmaker(bind=engine)


def get_session():
    """요청 하나 동안 쓸 DB 세션을 만들고, 끝나면 닫는다.

    사용처: 5단계(기록). main.py의 API 함수가 Depends(get_session)으로 받는다.
    입력: 없음
    출력: SQLAlchemy Session (제너레이터)
    주의: 커밋은 API 함수에서 직접 한다.
    """
    session: Session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
