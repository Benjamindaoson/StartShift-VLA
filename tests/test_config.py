from startshift.config import ExperimentConfig, dump_config, load_config


def test_config_round_trip(tmp_path):
    path=tmp_path/"config.yaml"
    dump_config(ExperimentConfig(name="test"),path)
    loaded=load_config(path)
    assert loaded.name=="test"
    assert loaded.policy.state_dim==8
