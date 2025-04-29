from sqlalchemy import ARRAY, JSON, Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from . import Base


class DatasetLinkType(Base):
    __tablename__ = "DatasetLinkType"
    id = Column(Integer, primary_key=True, autoincrement=True)
    value = Column(String, nullable=False)
    name = Column(String, nullable=False)
    isActive = Column(Boolean, nullable=False, default=True)


class DatasetBaseType(Base):
    __tablename__ = "DatasetBaseType"
    id = Column(Integer, primary_key=True, autoincrement=True)
    value = Column(String, nullable=False)
    name = Column(String, nullable=False)
    isActive = Column(Boolean, nullable=False, default=True)
    modelBaseTypes = relationship("ModelBaseType", back_populates="datasetType")
    datasets = relationship("Dataset", back_populates="datasetType")


class ModelBaseType(Base):
    __tablename__ = "ModelBaseType"
    id = Column(Integer, primary_key=True, autoincrement=True)
    value = Column(String, nullable=False)
    name = Column(String, nullable=False)
    datasetTypeId = Column(Integer, ForeignKey("DatasetBaseType.id"))
    datasetType = relationship("DatasetBaseType", back_populates="modelBaseTypes")
    isActive = Column(Boolean, nullable=False, default=True)
    modelResults = relationship("ModelResults", back_populates="modelType")
    yamlFile = Column(String, nullable=True)


class Dataset(Base):
    __tablename__ = "Dataset"
    id = Column(Integer, primary_key=True, autoincrement=True)
    datasetTypeId = Column(Integer, ForeignKey("DatasetBaseType.id"))
    datasetType = relationship("DatasetBaseType", back_populates="datasets")
    s3Key = Column(String, nullable=False)
    links = Column(ARRAY(String), nullable=False)
    tags = Column(ARRAY(String), nullable=False)
    modelResults = relationship("ModelResults", back_populates="dataset")
    name = Column(String, default="")
    checksumBlobS3Key = Column(String, default="")


class ModelResults(Base):
    __tablename__ = "ModelResults"
    id = Column(Integer, primary_key=True, autoincrement=True)
    datasetId = Column(Integer, ForeignKey("Dataset.id"))
    dataset = relationship("Dataset", back_populates="modelResults")
    modelTypeId = Column(Integer, ForeignKey("ModelBaseType.id"))
    modelType = relationship("ModelBaseType", back_populates="modelResults")
    params = Column(JSON, nullable=False)
    extras = Column(JSON, nullable=False, default={})
    iouScore = Column(Float)
    map50Score = Column(Float)
    map5095Score = Column(Float)
    inferenceTime = Column(Float)
    tags = Column(ARRAY(String), nullable=False)
    resultsS3Key = Column(String, nullable=False)
    modelS3Key = Column(String, nullable=False)
    tfjsS3Key = Column(String, nullable=False)
    isActive = Column(Boolean, nullable=False, default=True)
