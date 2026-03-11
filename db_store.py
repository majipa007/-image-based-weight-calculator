import logging
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class GoatFrameStore:
    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        batch_size: int = 100,
    ):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        self.conn.autocommit = False
        self.batch_size = batch_size
        self._candidate_buffer: List[Tuple] = []

    def close(self):
        if self.conn:
            self.conn.close()

    def create_video_run(self, source_name: str, top_k: int) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO video_runs (source_name, top_k)
                VALUES (%s, %s)
                RETURNING run_id
                """,
                (source_name, top_k),
            )
            run_id = cur.fetchone()[0]
        self.conn.commit()
        return str(run_id)

    def add_candidate(
        self,
        run_id: str,
        goat_id: int,
        frame_index: int,
        mask_area: int,
        crop_image_jpg: bytes,
        mask_png: bytes,
    ):
        self._candidate_buffer.append(
            (run_id, goat_id, frame_index, mask_area, psycopg2.Binary(crop_image_jpg), psycopg2.Binary(mask_png))
        )
        if len(self._candidate_buffer) >= self.batch_size:
            self.flush_candidates()

    def flush_candidates(self):
        if not self._candidate_buffer:
            return
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO goat_frame_candidates
                    (run_id, goat_id, frame_index, mask_area, crop_image_jpg, mask_png)
                VALUES %s
                """,
                self._candidate_buffer,
                page_size=self.batch_size,
            )
        self.conn.commit()
        self._candidate_buffer = []

    def fetch_top_candidates(self, run_id: str, top_k: int) -> Dict[int, List[Dict]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT goat_id, frame_index, mask_area, crop_image_jpg, mask_png
                FROM (
                    SELECT
                        goat_id,
                        frame_index,
                        mask_area,
                        crop_image_jpg,
                        mask_png,
                        ROW_NUMBER() OVER (
                            PARTITION BY goat_id
                            ORDER BY mask_area DESC, frame_index ASC
                        ) AS row_num
                    FROM goat_frame_candidates
                    WHERE run_id = %s
                ) ranked
                WHERE row_num <= %s
                ORDER BY goat_id, mask_area DESC, frame_index ASC
                """,
                (run_id, top_k),
            )
            rows = cur.fetchall()

        grouped: Dict[int, List[Dict]] = {}
        for goat_id, frame_index, mask_area, crop_image_jpg, mask_png in rows:
            grouped.setdefault(goat_id, []).append(
                {
                    "frame_index": frame_index,
                    "mask_area": mask_area,
                    "crop_image_jpg": bytes(crop_image_jpg),
                    "mask_png": bytes(mask_png),
                }
            )
        return grouped

    def upsert_goat_result(self, run_id: str, goat_id: int, weight_proxy_kg: float, samples_used: int):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO goat_results (run_id, goat_id, weight_proxy_kg, samples_used)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id, goat_id)
                DO UPDATE SET
                    weight_proxy_kg = EXCLUDED.weight_proxy_kg,
                    samples_used = EXCLUDED.samples_used
                """,
                (run_id, goat_id, weight_proxy_kg, samples_used),
            )
        self.conn.commit()


def create_store_from_config(db_config: Dict) -> Optional[GoatFrameStore]:
    try:
        return GoatFrameStore(
            host=db_config["host"],
            port=int(db_config["port"]),
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
        )
    except Exception as e:
        logging.error(f"Failed to connect to Postgres: {e}")
        return None
