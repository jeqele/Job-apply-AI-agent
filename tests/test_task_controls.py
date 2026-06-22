import unittest

from job_apply_ai.ui.cv_tasks import (
    TaskStopped,
    create_task,
    get_task,
    pause_task,
    request_task_stop,
    resume_task,
    task_control_checkpoint,
    update_task,
)


class TaskControlTests(unittest.TestCase):
    def test_pause_and_resume(self):
        task_id = create_task('batch_search')
        update_task(task_id, status='running')

        self.assertTrue(pause_task(task_id))
        self.assertEqual(get_task(task_id)['status'], 'paused')
        self.assertFalse(pause_task(task_id))

        self.assertTrue(resume_task(task_id))
        self.assertEqual(get_task(task_id)['status'], 'running')

    def test_stop_raises_in_checkpoint(self):
        task_id = create_task('job_match_analyze')
        update_task(task_id, status='running')
        self.assertTrue(request_task_stop(task_id))

        with self.assertRaises(TaskStopped):
            task_control_checkpoint(task_id)

    def test_checkpoint_blocks_while_paused_then_resumes(self):
        task_id = create_task('batch_search')
        update_task(task_id, status='running')
        pause_task(task_id)

        seen = {'resumed': False}

        def resume_after_delay():
            import time
            time.sleep(0.05)
            resume_task(task_id)
            seen['resumed'] = True

        import threading
        threading.Thread(target=resume_after_delay, daemon=True).start()
        task_control_checkpoint(task_id)
        self.assertTrue(seen['resumed'])


if __name__ == '__main__':
    unittest.main()
