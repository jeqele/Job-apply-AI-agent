"""Tests for job folder move history and repository helpers."""

import os
import tempfile
import unittest
from unittest.mock import patch

from job_apply_ai.job_move_history import (
    can_redo_job_moves,
    can_undo_job_moves,
    pop_redo_job_moves,
    pop_undo_job_moves,
    record_job_moves,
    redo_job_move_label,
    undo_job_move_label,
)
from job_apply_ai.storage.database import init_db
from job_apply_ai.storage.job_repository import JobRepository


class JobMoveHistoryTests(unittest.TestCase):
    def setUp(self):
        self.session = {}

    def test_record_and_undo_redo_labels(self):
        changes = [{"job_id": 1, "from_status": "new", "to_status": "archived"}]
        record_job_moves(self.session, changes, "Archive 1 job")
        self.assertTrue(can_undo_job_moves(self.session))
        self.assertFalse(can_redo_job_moves(self.session))
        self.assertEqual(undo_job_move_label(self.session), "Archive 1 job")

        entry = pop_undo_job_moves(self.session)
        self.assertEqual(entry["label"], "Archive 1 job")
        self.assertTrue(can_redo_job_moves(self.session))
        self.assertEqual(redo_job_move_label(self.session), "Archive 1 job")

        entry = pop_redo_job_moves(self.session)
        self.assertEqual(entry["changes"][0]["to_status"], "archived")

    def test_record_clears_redo_stack(self):
        record_job_moves(
            self.session,
            [{"job_id": 1, "from_status": "new", "to_status": "applied"}],
            "First move",
        )
        pop_undo_job_moves(self.session)
        self.assertTrue(can_redo_job_moves(self.session))

        record_job_moves(
            self.session,
            [{"job_id": 1, "from_status": "applied", "to_status": "archived"}],
            "Second move",
        )
        self.assertFalse(can_redo_job_moves(self.session))


class JobRepositoryMoveTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmpdir.name, "moves.db")
        self._env_patch = patch.dict(os.environ, {"JOB_APPLY_AI_DB": self.db_path})
        self._env_patch.start()
        init_db(self.db_path)
        self.repo = JobRepository()

    def tearDown(self):
        self._env_patch.stop()
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    def test_move_jobs_status_returns_change_records(self):
        job_id = self.repo.create_job({"title": "Engineer", "company": "Acme"})
        changes = self.repo.move_jobs_status([job_id], "shortlisted")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["from_status"], "new")
        self.assertEqual(changes[0]["to_status"], "shortlisted")

        job = self.repo.get_job(job_id)
        self.assertEqual(job["workflow_status"], "shortlisted")

        no_op = self.repo.move_jobs_status([job_id], "shortlisted")
        self.assertEqual(no_op, [])

    def test_apply_job_status_changes_undo_and_redo(self):
        job_id = self.repo.create_job({"title": "Designer", "company": "Beta"})
        changes = self.repo.move_jobs_status([job_id], "archived")

        restored = self.repo.apply_job_status_changes(changes, use_from_status=True)
        self.assertEqual(restored, 1)
        self.assertEqual(self.repo.get_job(job_id)["workflow_status"], "new")

        reapplied = self.repo.apply_job_status_changes(changes, use_from_status=False)
        self.assertEqual(reapplied, 1)
        self.assertEqual(self.repo.get_job(job_id)["workflow_status"], "archived")

    def test_batch_move_only_changes_jobs_that_need_it(self):
        first_id = self.repo.create_job({"title": "One", "company": "A"})
        second_id = self.repo.create_job({"title": "Two", "company": "B"})
        self.repo.move_jobs_status([first_id], "applied")

        changes = self.repo.move_jobs_status([first_id, second_id], "archived")
        self.assertEqual(len(changes), 2)
        self.assertEqual(self.repo.get_job(first_id)["workflow_status"], "archived")
        self.assertEqual(self.repo.get_job(second_id)["workflow_status"], "archived")


if __name__ == "__main__":
    unittest.main()
