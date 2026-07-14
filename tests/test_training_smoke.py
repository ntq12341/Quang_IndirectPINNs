from src.losses import indirect_loss
from src.models import IndirectPINN
from src.problems import make_problem
from src.trainers import TrainConfig, make_batch, train


def test_short_indirect_training_runs_and_reduces_loss():
    problem = make_problem("linear_kkt")
    config = TrainConfig(epochs=5, n_interior=16, n_boundary=8, n_initial=8, resample_every=100, seed=3)
    weights = {"state": 1.0, "adjoint": 1.0, "stationarity": 1.0, "initial": 10.0}
    model = IndirectPINN(state_hidden=(8, 8), control_hidden=(8,), adjoint_hidden=(8, 8)).double()
    batch = make_batch(problem, config, config.seed, "cpu")
    before = float(indirect_loss(model, problem, batch, weights)[0].detach())
    train(model, problem, "indirect", config, weights)
    batch = make_batch(problem, config, config.seed, "cpu")
    after = float(indirect_loss(model, problem, batch, weights)[0].detach())
    assert after < before

