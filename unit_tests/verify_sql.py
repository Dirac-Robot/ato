import unittest
import tempfile
import os
from beacon.adict import ADict
from beacon.db_routers.sql.manager import SQLLogger, SQLFinder
from beacon.db_routers.sql.schema import Base, Project, Experiment, Metric


class SQLLoggerUnitTest(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = f'sqlite:///{self.temp_db.name}'
        self.config = ADict(
            experiment=ADict(
                project_name='test_project',
                sql=ADict(db_path=self.db_path)
            ),
            learning_rate=0.1,
            batch_size=32
        )
        self.logger = SQLLogger(self.config)

    def tearDown(self):
        self.logger.session.close()
        self.logger.engine.dispose()
        os.unlink(self.temp_db.name)

    def test_get_or_create_project(self):
        project = self.logger.get_or_create_project()
        self.assertIsNotNone(project)
        self.assertEqual(project.name, 'test_project')

        # Test that calling again returns the same project
        project2 = self.logger.get_or_create_project()
        self.assertEqual(project.id, project2.id)

    def test_run(self):
        run_id = self.logger.run(tags=['test', 'experiment'])
        self.assertIsNotNone(run_id)

        run = self.logger.get_current_run()
        self.assertEqual(run.id, run_id)
        self.assertEqual(run.status, 'running')
        self.assertEqual(run.tags, ['test', 'experiment'])

    def test_log_metric(self):
        run_id = self.logger.run()
        self.logger.log_metric('accuracy', 0.95, step=1)
        self.logger.log_metric('accuracy', 0.97, step=2)

        run = self.logger.get_current_run()
        self.assertEqual(len(run.metrics), 2)
        self.assertEqual(run.metrics[0].key, 'accuracy')
        self.assertEqual(run.metrics[0].value, 0.95)

    def test_log_artifact(self):
        run_id = self.logger.run()
        self.logger.log_artifact(run_id, '/path/to/model.pth', 'model', {'size': '100MB'})

        run = self.logger.get_current_run()
        self.assertEqual(len(run.artifacts), 1)
        self.assertEqual(run.artifacts[0].path, '/path/to/model.pth')
        self.assertEqual(run.artifacts[0].data_type, 'model')

    def test_update_status(self):
        self.logger.run()
        self.logger.update_status('failed')

        run = self.logger.get_current_run()
        self.assertEqual(run.status, 'failed')

    def test_finish(self):
        self.logger.run()
        self.logger.finish(status='completed')

        run = self.logger.get_current_run()
        self.assertEqual(run.status, 'completed')
        self.assertIsNotNone(run.end_time)


class SQLFinderUnitTest(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = f'sqlite:///{self.temp_db.name}'
        self.config = ADict(
            experiment=ADict(
                project_name='test_project',
                sql=ADict(db_path=self.db_path)
            ),
            learning_rate=0.1,
            batch_size=32
        )

        # Setup logger to create some test data
        self.logger = SQLLogger(self.config)
        self.run_id_1 = self.logger.run(tags=['exp1'])
        self.logger.log_metric('loss', 0.5, step=1)
        self.logger.log_metric('accuracy', 0.8, step=1)
        self.logger.finish()

        # Create second run with same structural hash
        self.logger._current_run_id = None
        self.run_id_2 = self.logger.run(tags=['exp2'])
        self.logger.log_metric('loss', 0.3, step=1)
        self.logger.log_metric('accuracy', 0.9, step=1)
        self.logger.finish()

        self.finder = SQLFinder(self.config)

    def tearDown(self):
        self.logger.session.close()
        self.logger.engine.dispose()
        os.unlink(self.temp_db.name)

    def test_find_project(self):
        project = self.finder.find_project('test_project')
        self.assertIsNotNone(project)
        self.assertEqual(project.name, 'test_project')

        non_existent = self.finder.find_project('non_existent')
        self.assertIsNone(non_existent)

    def test_find_run(self):
        run = self.finder.find_run(self.run_id_1)
        self.assertIsNotNone(run)
        self.assertEqual(run.id, self.run_id_1)

        non_existent = self.finder.find_run(99999)
        self.assertIsNone(non_existent)

    def test_get_runs_in_project(self):
        runs = self.finder.get_runs_in_project('test_project')
        self.assertEqual(len(runs), 2)

        empty_runs = self.finder.get_runs_in_project('non_existent')
        self.assertEqual(len(empty_runs), 0)

    def test_find_similar_runs(self):
        similar_runs = self.finder.find_similar_runs(self.run_id_1)
        self.assertEqual(len(similar_runs), 1)
        self.assertEqual(similar_runs[0].id, self.run_id_2)

    def test_find_best_run(self):
        # Test finding best run by max accuracy
        best_run = self.finder.find_best_run('test_project', 'accuracy', mode='max')
        self.assertIsInstance(best_run, Experiment)
        self.assertEqual(best_run.id, self.run_id_2)

        # Test finding best run by min loss
        best_run = self.finder.find_best_run('test_project', 'loss', mode='min')
        self.assertIsInstance(best_run, Experiment)
        self.assertEqual(best_run.id, self.run_id_2)

        # Test non-existent project
        result = self.finder.find_best_run('non_existent', 'accuracy')
        self.assertEqual(result, {'error': 'Project not found'})

    def test_get_trace_statistics(self):
        result = self.finder.get_trace_statistics('test_project', 'main_function')
        self.assertIn('project_name', result)
        self.assertIn('trace_id', result)
        self.assertIn('static_trace_versions', result)
        self.assertIn('runtime_trace_versions', result)

        # Test non-existent project
        result = self.finder.get_trace_statistics('non_existent', 'trace_id')
        self.assertEqual(result, {'error': 'Project not found'})


if __name__ == "__main__":
    unittest.main()