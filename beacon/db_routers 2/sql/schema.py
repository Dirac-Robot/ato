import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# single project
class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    runs = relationship('Experiment', back_populates='project')  # makes 1:N relationship


# single experiment
class Experiment(Base):
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False, index=True)
    config = Column(JSON, nullable=False)

    structural_hash = Column(String(64), index=True)

    tags = Column(JSON, default=[])
    status = Column(String, default='running', index=True)  # current status of experiment
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime)

    project = relationship('Project', back_populates='runs')
    metrics = relationship('Metric', back_populates='run')
    artifacts = relationship('Artifact', back_populates='run')
    fingerprints = relationship('Fingerprint', back_populates='run')


# single metric
class Metric(Base):
    __tablename__ = 'metrics'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('experiments.id'), nullable=False, index=True)

    key = Column(String, nullable=False, index=True)  # name of metric
    value = Column(Float, nullable=False)  # value of metric
    step = Column(Integer, nullable=False)  # logged step
    timestamp = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))

    run = relationship('Experiment', back_populates='metrics')


# single artifact
class Artifact(Base):
    __tablename__ = 'artifacts'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('experiments.id'), nullable=False, index=True)

    path = Column(String, nullable=False)
    data_type = Column(String)
    data_info = Column(JSON)

    run = relationship('Experiment', back_populates='artifacts')


# single fingerprint
class Fingerprint(Base):
    __tablename__ = 'fingerprints'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('experiments.id'), nullable=False, index=True)
    trace_id = Column(String, nullable=False, index=True)

    trace_type = Column(String(32), nullable=False, index=True)  # 'static' or 'runtime'
    fingerprint = Column(String(128), nullable=False, index=True)  # hash value

    run = relationship('Experiment', back_populates='fingerprints')
