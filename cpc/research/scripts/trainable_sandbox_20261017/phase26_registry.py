"""Trusted parent-process task registry; never import this module in a worker."""
import phase26_training as training
import phase25_training as parent

OUT=training.OUT
TABLE=parent.TABLE
ROLES=parent.ROLES
METHODS=training.METHODS
SPECS=training.SPECS
configs=training.configs
fit_path=training.path_for
units=parent.units
unit_column=parent.unit_column
subset=parent.subset
weights=parent.weights
digest=parent.digest
data=parent.data
