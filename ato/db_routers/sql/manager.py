import datetime
from typing import Optional

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session as SessionType

from ato.db_routers import BaseLogger, BaseFinder
from ato.db_routers.sql.schema import Base, Project, Experiment, Metric, Artifact, Fingerprint


class SQLLogger(BaseLogger):
    registry: set[str] = set()

    def __init__(self, config):
        super().__init__(config)
        db_path = self.config.experiment.sql.db_path
        self.engine = create_engine(db_path)
        registry = self.__class__.registry
        if db_path not in registry:
            Base.metadata.create_all(self.engine)  # Base.metadata is SQLAlchemy's internal attribute
            registry.add(db_path)
        self.session = sessionmaker(bind=self.engine)()
        self._current_run_id = None

    def get_current_run(self):
        return self.session.get(Experiment, self.get_current_run_id())

    def get_current_run_id(self):
        return self._current_run_id

    def get_or_create_project(self):
        project = self.session.query(Project).filter_by(name=self.config.experiment.project_name).first()
        if not project:
            project = Project(name=self.config.experiment.project_name)
            self.session.add(project)
            self.session.commit()
        return project

    def update_status(self, status):
        run = self.get_current_run()
        if run:
            run.status = status
            self.session.commit()

    def run(self, tags=None):
        project = self.get_or_create_project()
        structural_hash = self.config.get_structural_hash()
        run = Experiment(
            project_id=project.id,
            config=self.config.to_dict(),  # Recursively convert ADict to dict for JSON serialization
            structural_hash=structural_hash,
            tags=tags or []
        )
        self.session.add(run)
        self.session.commit()
        self._current_run_id = run.id
        return run.id

    def log_metric(self, key, value, step):
        metric = Metric(
            run_id=self.get_current_run_id(),
            key=key,
            value=value,
            step=step
        )
        self.session.add(metric)
        self.session.commit()

    def log_artifact(self, run_id, file_path, data_type, metadata=None):
        artifact = Artifact(
            run_id=run_id,
            path=file_path,
            data_type=data_type,
            data_info=metadata  # Column name is data_info in schema
        )
        self.session.add(artifact)
        self.session.commit()

    def finish(self, status='completed'):
        run = self.get_current_run()
        if run:
            run.status = status
            run.end_time = datetime.datetime.now(datetime.timezone.utc)
            self.session.commit()


class SQLFinder(BaseFinder):
    def __init__(self, config):
        super().__init__(config)
        self.engine = create_engine(self.config.experiment.sql.db_path)
        self.session_maker = sessionmaker(bind=self.engine)

    def _get_session(self) -> SessionType:
        return self.session_maker()

    def find_project(self, project_name) -> Optional[Project]:
        with self._get_session() as session:
            return session.query(Project).filter_by(name=project_name).first()

    def find_run(self, run_id: int) -> Optional[Experiment]:
        with self._get_session() as session:
            return session.get(Experiment, run_id)

    def get_runs_in_project(self, project_name) -> list[Experiment]:
        project = self.find_project(project_name)
        if not project:
            return []
        else:
            with self._get_session() as session:
                return session.query(Experiment).filter_by(project_id=project.id).all()

    def find_similar_runs(self, run_id: int) -> list[Experiment]:
        with self._get_session() as session:
            base_run = session.query(Experiment.structural_hash).filter_by(id=run_id).first()
            if not base_run:
                return []
            else:
                target_hash = base_run.structural_hash
                return session.query(Experiment).filter(
                    Experiment.structural_hash == target_hash,
                    Experiment.id != run_id
                ).all()

    def find_similar_runs_by_trace(
        self,
        run_id: int,
        trace_id: str,
        trace_type: str = 'static'
    ) -> list[Experiment]:
        with self._get_session() as session:
            target_fingerprint = session.query(Fingerprint.fingerprint).filter(
                Fingerprint.run_id == run_id,
                Fingerprint.trace_id == trace_id,
                Fingerprint.trace_type == trace_type
            ).scalar()
            if not target_fingerprint:
                return []
            query = session.query(Fingerprint.run_id).filter(
                Fingerprint.trace_id == trace_id,
                Fingerprint.trace_type == trace_type,
                Fingerprint.fingerprint == target_fingerprint,
                Fingerprint.run_id != run_id
            ).distinct()
            return session.query(Experiment).filter(Experiment.id.in_(query)).all()

    def find_best_run(self, project_name: str, metric_key: str, mode: str = 'max') -> Experiment | dict:
        project = self.find_project(project_name)
        if project:
            with self._get_session() as session:
                order_by_col = Metric.value.desc() if mode == 'max' else Metric.value.asc()
                best_run = session.query(Experiment, Metric.value).join(
                    Metric,
                    Experiment.id == Metric.run_id
                ).filter(
                    Experiment.project_id == project.id,
                    Metric.key == metric_key
                ).order_by(
                    order_by_col
                ).first()
                if best_run:
                    return best_run[0]
                else:
                    return {'error': 'Run not found'}
        return {'error': 'Project not found'}

    def get_trace_statistics(self, project_name: str, trace_id: str) -> dict:
        project = self.find_project(project_name)
        if not project:
            return {'error': 'Project not found'}
        else:
            with self._get_session() as session:
                run_ids_in_project = session.query(Experiment.id).filter_by(project_id=project.id)
                static_count = session.query(func.count(Fingerprint.fingerprint.distinct())).filter(
                    Fingerprint.run_id.in_(run_ids_in_project),
                    Fingerprint.trace_id == trace_id,
                    Fingerprint.trace_type == 'static'
                ).scalar()
                runtime_count = session.query(func.count(Fingerprint.fingerprint.distinct())).filter(
                    Fingerprint.run_id.in_(run_ids_in_project),
                    Fingerprint.trace_id == trace_id,
                    Fingerprint.trace_type == 'runtime'  # @runtime_trace의 타입
                ).scalar()
                return {
                    'project_name': project_name,
                    'trace_id': trace_id,
                    'static_trace_versions': static_count,
                    'runtime_trace_versions': runtime_count
                }
